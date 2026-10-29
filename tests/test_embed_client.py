"""HttpEmbedder: batching by 10, retry on 429/5xx, give-up on 4xx, pacing, normalisation. urllib is mocked."""
import io
import json
import urllib.error
from unittest import mock

import numpy as np

from app.embed import MAX_BATCH, HttpEmbedder, _Bucket


class _Resp(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _ok_response(req):
    body = json.loads(req.data.decode())
    n, dims = len(body["input"]), body["dimensions"]
    data = [{"index": i, "embedding": [float(i + 1)] * dims} for i in range(n)]
    return _Resp(json.dumps({"data": data, "usage": {"prompt_tokens": n}}).encode())


def test_batches_of_ten_and_unit_vectors():
    calls = []

    def fake_urlopen(req, timeout=0):
        calls.append(len(json.loads(req.data.decode())["input"]))
        return _ok_response(req)

    emb = HttpEmbedder("text-embedding-v4", "k", dims=8, rps=1000)
    with mock.patch("urllib.request.urlopen", fake_urlopen):
        vecs = emb.embed([f"text {i}" for i in range(23)] + ["", "   "])
    assert calls == [MAX_BATCH, MAX_BATCH, 3]           # 23 non-empty texts -> 10 + 10 + 3; blanks never sent
    assert len(vecs) == 25 and vecs[23] is None and vecs[24] is None
    assert all(abs(float(np.linalg.norm(v)) - 1.0) < 1e-5 for v in vecs[:23])


def test_retry_on_429_then_success_and_give_up_on_400():
    attempts = {"n": 0}

    def flaky(req, timeout=0):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise urllib.error.HTTPError(req.full_url, 429, "slow down", {}, io.BytesIO(b""))
        return _ok_response(req)

    emb = HttpEmbedder("text-embedding-v4", "k", dims=4, rps=1000, retries=2)
    with mock.patch("urllib.request.urlopen", flaky), mock.patch("time.sleep"):
        assert emb.embed_one("hello") is not None
    assert attempts["n"] == 2 and emb.failures == 0

    def bad_request(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 400, "bad", {}, io.BytesIO(b""))

    with mock.patch("urllib.request.urlopen", bad_request), mock.patch("time.sleep"):
        assert emb.embed_one("hello") is None
    assert emb.failures == 1


def test_bucket_paces_calls():
    b = _Bucket(rps=5.0)
    waited = []
    with mock.patch("time.sleep", lambda s: waited.append(s)):
        for _ in range(7):
            b.acquire()
    # 5 tokens burst, then every further call needs to wait about 1/5 s
    assert len(waited) >= 1 and all(0 < w <= 0.21 for w in waited)


def test_disk_cache_skips_repeat_calls(tmp_path):
    calls = {"n": 0}

    def counting(req, timeout=0):
        calls["n"] += 1
        return _ok_response(req)

    emb = HttpEmbedder("text-embedding-v4", "k", dims=8, rps=1000, cache_path=str(tmp_path / "c.sqlite"))
    with mock.patch("urllib.request.urlopen", counting):
        a = emb.embed(["alpha", "beta"])
        b = emb.embed(["beta", "gamma", "alpha"])
    assert calls["n"] == 2                     # second call only fetched "gamma"
    assert np.allclose(a[1], b[0], atol=1e-3) and np.allclose(a[0], b[2], atol=1e-3)
