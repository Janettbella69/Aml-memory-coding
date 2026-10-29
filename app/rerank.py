"""Optional cross-encoder reranker (v0.7): one rerank-API call per Search over the first RERANK_CANDIDATES blocks.
The academic-board rules fix the embedding (text-embedding-v4) and the LLM (gpt-4o-mini) but leave the reranker
free ("Reranker 不作限制"), so any hosted reranker may be used. Scores only reorder our own verbatim blocks; any
failure keeps the deterministic order.

Why: the locked answer model reads the list in order and mostly the head (window-arms experiment 2026-10-09:
evidence-first ordering lifts gpt-4o-mini accuracy 0.778 -> 0.819 on the same retrieved blocks; a cross-encoder
gets close to the oracle ordering on single-hop questions).

Providers (RERANK_PROVIDER): dashscope (gte-rerank-v2 | qwen3-rerank; DASHSCOPE_API_KEY), voyage (rerank-2.5 |
rerank-2.5-lite; VOYAGE_API_KEY), cohere (rerank-v3.5 | rerank-4-pro | rerank-4-fast; COHERE_API_KEY), jina
(jina-reranker-v3 | …; JINA_API_KEY). RERANK_API_KEY overrides the provider-specific key; RERANK_ENDPOINT overrides
the URL. Every provider returns {index, relevance_score} per document, so one client covers all of them.
"""
from __future__ import annotations

import http.client
import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Callable, Optional, Protocol

log = logging.getLogger("aml_memory.rerank")


class Reranker(Protocol):
    def scores(self, query: str, documents: list[str]) -> Optional[list[float]]: ...


def _dashscope_body(model: str, query: str, docs: list[str]) -> dict[str, Any]:
    return {"model": model, "input": {"query": query, "documents": docs},
            "parameters": {"return_documents": False, "top_n": len(docs)}}


def _voyage_body(model: str, query: str, docs: list[str]) -> dict[str, Any]:
    return {"model": model, "query": query, "documents": docs, "top_k": len(docs), "truncation": True}


def _cohere_body(model: str, query: str, docs: list[str]) -> dict[str, Any]:
    return {"model": model, "query": query, "documents": docs, "top_n": len(docs)}


def _jina_body(model: str, query: str, docs: list[str]) -> dict[str, Any]:
    return {"model": model, "query": query, "documents": docs, "top_n": len(docs), "return_documents": False}


PROVIDERS: dict[str, dict[str, Any]] = {
    # name: endpoint, body builder, path to the results list, env var of the key
    "dashscope": {"endpoint": "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
                  "body": _dashscope_body, "results": ("output", "results"), "key_env": "DASHSCOPE_API_KEY"},
    "voyage": {"endpoint": "https://api.voyageai.com/v1/rerank", "body": _voyage_body, "results": ("data",),
               "key_env": "VOYAGE_API_KEY"},
    "cohere": {"endpoint": "https://api.cohere.com/v2/rerank", "body": _cohere_body, "results": ("results",),
               "key_env": "COHERE_API_KEY"},
    "jina": {"endpoint": "https://api.jina.ai/v1/rerank", "body": _jina_body, "results": ("results",),
             "key_env": "JINA_API_KEY"},
}


class HttpReranker:
    """Bearer-token JSON call; the response's results list carries {index, relevance_score} per input document."""

    def __init__(self, provider: str, model: str, api_key: str, endpoint: str = "", timeout: float = 20.0,
                 doc_chars: int = 1500, query_chars: int = 2000) -> None:
        spec = PROVIDERS[provider]
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.endpoint = endpoint or spec["endpoint"]
        self.build: Callable[[str, str, list[str]], dict[str, Any]] = spec["body"]
        self.results_path: tuple[str, ...] = spec["results"]
        self.timeout = timeout
        self.doc_chars = doc_chars
        self.query_chars = query_chars

    def scores(self, query: str, documents: list[str]) -> Optional[list[float]]:
        if not documents:
            return None
        q = query if len(query) <= self.query_chars else query[:self.query_chars // 2] + "\n[...]\n" + query[-self.query_chars // 2:]
        body = self.build(self.model, q, [d[:self.doc_chars] for d in documents])
        req = urllib.request.Request(self.endpoint, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json", "Accept": "application/json",
                                              "Authorization": f"Bearer {self.api_key}"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                out = json.load(resp)
            items: Any = out
            for k in self.results_path:
                items = items[k]
            scores = [float("-inf")] * len(documents)
            for item in items:
                idx = int(item["index"])
                if 0 <= idx < len(documents):
                    scores[idx] = float(item["relevance_score"])
            return scores
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError, TypeError, json.JSONDecodeError, OSError,
                http.client.HTTPException) as exc:
            log.warning("rerank call failed (%s %s); deterministic order kept", self.provider, type(exc).__name__)  # no query text
            return None


def reranker_from_env(provider: str, model: str, timeout: float, doc_chars: int) -> Optional[Reranker]:
    if not model:
        return None
    if provider not in PROVIDERS:
        log.warning("RERANK_PROVIDER=%s unknown; reranker disabled", provider)
        return None
    key = (os.environ.get("RERANK_API_KEY", "").strip() or os.environ.get(PROVIDERS[provider]["key_env"], "").strip())
    if not key:
        log.warning("RERANK_MODEL=%s set but no %s / RERANK_API_KEY; reranker disabled", model, PROVIDERS[provider]["key_env"])
        return None
    return HttpReranker(provider, model, key, endpoint=os.environ.get("RERANK_ENDPOINT", "").strip(),
                        timeout=timeout, doc_chars=doc_chars)
