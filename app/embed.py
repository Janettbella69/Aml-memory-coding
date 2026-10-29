"""Dense channel (v0.5): `text-embedding-v4` through an OpenAI-compatible `/embeddings` endpoint (Alibaba Model Studio).

Rule: on the academic (open-source) board the only embedding model allowed is `text-embedding-v4` (competition FAQ 05);
`EMBED_MODEL` therefore defaults to it and nothing else is tested. The service rejects batches above 10 inputs and
publishes 1,800 requests/min and ~1.2 M tokens/min per key, so the client batches by 10, paces itself with a token
bucket (`EMBED_RPS`) and retries 429/5xx/timeouts with backoff. Vectors are unit-normalised and stored as float16
(`EMBED_DIMS` x 2 bytes per turn). Failures degrade: the turn is stored without a vector (lexical retrieval still
covers it) and is back-filled later. No memory or query text is ever logged.

Idea credit: the hybrid dense + BM25 layout follows Cycle-1 leaders InvMem and Hybrid Episodic Memory (see
docs/CYCLE1_NOTES.md); the code here is our own. `FakeEmbedder` exists for tests only.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import logging
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from typing import Optional, Protocol

import numpy as np

log = logging.getLogger("aml_memory")

MAX_INPUT_CHARS = 6000       # text-embedding-v4 accepts 8,192 tokens per input; our turns are far shorter
MAX_BATCH = 10               # service limit ("batch size ... should not be larger than 10")
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class Embedder(Protocol):
    model: str
    dims: int

    def embed(self, texts: list[str]) -> list[Optional[np.ndarray]]: ...


def to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float16).tobytes()


def from_blob(blob: bytes, dims: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float16, count=dims).astype(np.float32)


def _normalise(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (rows / norms).astype(np.float32)


class _Bucket:
    """Token bucket: at most `rps` calls per second on average, bursts up to `rps`."""

    def __init__(self, rps: float) -> None:
        self.rps = max(rps, 0.1)
        self.tokens = self.rps
        self.stamp = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.rps, self.tokens + (now - self.stamp) * self.rps)
                self.stamp = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                wait = (1.0 - self.tokens) / self.rps
            time.sleep(wait)


class HttpEmbedder:
    def __init__(self, model: str, api_key: str, base_url: str = DEFAULT_BASE_URL, dims: int = 256,
                 timeout: float = 30.0, rps: float = 14.0, retries: int = 3, cache_path: Optional[str] = None) -> None:
        self.model = model
        self.dims = dims
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.bucket = _Bucket(rps)
        self.failures = 0            # calls that gave up (metrics only)
        # Optional on-disk cache (EMBED_CACHE=<sqlite file>): identical text -> identical vector, so repeated
        # evaluation runs stop paying for the same embeddings. Never set in production (it would retain text).
        self._cache: Optional[sqlite3.Connection] = None
        self._cache_lock = threading.Lock()
        if cache_path:
            self._cache = sqlite3.connect(cache_path, check_same_thread=False)
            self._cache.execute("CREATE TABLE IF NOT EXISTS vec (k TEXT PRIMARY KEY, v BLOB NOT NULL)")
            self._cache.commit()

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}:{self.dims}:{text}".encode("utf-8")).hexdigest()

    def _cache_get(self, texts: list[str]) -> dict[int, np.ndarray]:
        if self._cache is None:
            return {}
        out: dict[int, np.ndarray] = {}
        with self._cache_lock:
            for i, t in enumerate(texts):
                row = self._cache.execute("SELECT v FROM vec WHERE k = ?", (self._key(t),)).fetchone()
                if row is not None:
                    out[i] = from_blob(row[0], self.dims)
        return out

    def _cache_put(self, pairs: list[tuple[str, np.ndarray]]) -> None:
        if self._cache is None or not pairs:
            return
        with self._cache_lock:
            self._cache.executemany("INSERT OR IGNORE INTO vec(k, v) VALUES (?, ?)", [(self._key(t), to_blob(v)) for t, v in pairs])
            self._cache.commit()

    def _call(self, batch: list[str]) -> Optional[list[np.ndarray]]:
        body = json.dumps({"model": self.model, "input": batch, "dimensions": self.dims,
                           "encoding_format": "float"}).encode("utf-8")
        req = urllib.request.Request(self.base_url + "/embeddings", data=body, method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        delay = 0.5
        for attempt in range(self.retries + 1):
            self.bucket.acquire()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    out = json.load(resp)
                data = sorted(out["data"], key=lambda d: d["index"])
                if len(data) != len(batch):
                    raise ValueError("embedding count mismatch")
                rows = np.asarray([d["embedding"] for d in data], dtype=np.float32)
                if rows.shape != (len(batch), self.dims):
                    raise ValueError("embedding shape mismatch")
                return list(_normalise(rows))
            except urllib.error.HTTPError as err:
                retryable = err.code == 429 or err.code >= 500
                log.warning("embedding call failed: HTTP %s (attempt %d)", err.code, attempt + 1)  # no text in logs
                if not retryable or attempt == self.retries:
                    break
            except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException, ValueError, KeyError,
                    json.JSONDecodeError) as exc:
                log.warning("embedding call failed: %s (attempt %d)", type(exc).__name__, attempt + 1)
                if attempt == self.retries:
                    break
            time.sleep(delay)
            delay = min(delay * 2, 8.0)
        self.failures += 1
        return None

    def embed(self, texts: list[str]) -> list[Optional[np.ndarray]]:
        """One vector per text (None where the service could not be reached); empty texts get None."""
        out: list[Optional[np.ndarray]] = [None] * len(texts)
        clipped = [t.strip()[:MAX_INPUT_CHARS] if t else "" for t in texts]
        cached = self._cache_get(clipped)
        for i, v in cached.items():
            out[i] = v
        todo = [(i, t) for i, t in enumerate(clipped) if t and i not in cached]
        for start in range(0, len(todo), MAX_BATCH):
            chunk = todo[start:start + MAX_BATCH]
            vecs = self._call([t for _, t in chunk])
            if vecs is None:
                continue
            for (i, _), v in zip(chunk, vecs):
                out[i] = v
            self._cache_put([(t, v) for (_, t), v in zip(chunk, vecs)])
        return out

    def embed_one(self, text: str) -> Optional[np.ndarray]:
        return self.embed([text])[0]


class FakeEmbedder:
    """Deterministic test double: each word hashes to a fixed random direction, a text is the normalised sum,
    so texts sharing words are close. `aliases` maps a word to another so tests can fake paraphrases."""

    def __init__(self, dims: int = 32, aliases: Optional[dict[str, str]] = None, fail: bool = False) -> None:
        self.model = "fake"
        self.dims = dims
        self.aliases = aliases or {}
        self.fail = fail
        self.calls = 0

    def _word(self, w: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.blake2b(w.encode("utf-8"), digest_size=8).digest(), "little")
        return np.random.default_rng(seed).standard_normal(self.dims).astype(np.float32)

    def embed(self, texts: list[str]) -> list[Optional[np.ndarray]]:
        self.calls += 1
        if self.fail:
            return [None] * len(texts)
        out: list[Optional[np.ndarray]] = []
        for t in texts:
            words = [self.aliases.get(w, w) for w in re.findall(r"\w+", t.lower())]
            if not words:
                out.append(None)
                continue
            out.append(_normalise(np.sum([self._word(w) for w in words], axis=0)[None, :])[0])
        return out

    def embed_one(self, text: str) -> Optional[np.ndarray]:
        return self.embed([text])[0]


def embedder_from_env(model: str, dims: int, timeout: float, rps: float) -> Optional[HttpEmbedder]:
    if not model:
        return None
    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not key:
        log.warning("EMBED_MODEL=%s set but DASHSCOPE_API_KEY is empty; dense channel disabled", model)
        return None
    return HttpEmbedder(model, key, os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL,
                        dims=dims, timeout=timeout, rps=rps, cache_path=os.environ.get("EMBED_CACHE", "").strip() or None)
