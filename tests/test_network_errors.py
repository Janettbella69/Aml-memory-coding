"""Every outbound client degrades instead of raising when the network fails mid-call. urllib wraps connection
errors in URLError, but a TLS alert, a reset or a short read while the response is being read surfaces raw
(ssl.SSLError / ConnectionResetError / http.client.IncompleteRead); one escaped from the planner on 2026-10-15
and turned a Search into an HTTP 500."""
import http.client
import ssl
import urllib.error
from unittest import mock

import pytest

from app.embed import HttpEmbedder
from app.planner import OpenAIPlanner
from app.rerank import HttpReranker
from app.selector import OpenAISelector

FAILURES = [
    ssl.SSLError("[SSL: TLSV1_ALERT_PROTOCOL_VERSION] tlsv1 alert protocol version"),
    ConnectionResetError("reset by peer"),
    http.client.RemoteDisconnected("closed without response"),
    http.client.IncompleteRead(b"{\"choi"),
    urllib.error.URLError("down"),
    TimeoutError("timed out"),
]


def _raising(exc):
    def urlopen(req, timeout=0):
        raise exc
    return urlopen


@pytest.mark.parametrize("exc", FAILURES, ids=lambda e: type(e).__name__)
def test_planner_falls_back(exc):
    with mock.patch("urllib.request.urlopen", _raising(exc)):
        assert OpenAIPlanner("gpt-4o-mini", "k").plan("when did I move?", None, ["a snippet"]) is None


@pytest.mark.parametrize("exc", FAILURES, ids=lambda e: type(e).__name__)
def test_selector_keeps_order(exc):
    with mock.patch("urllib.request.urlopen", _raising(exc)):
        assert OpenAISelector("gpt-4o-mini", "k").select("when did I move?", None, ["block one", "block two"]) is None


@pytest.mark.parametrize("exc", FAILURES, ids=lambda e: type(e).__name__)
def test_reranker_keeps_order(exc):
    with mock.patch("urllib.request.urlopen", _raising(exc)):
        assert HttpReranker("dashscope", "qwen3-rerank", "k").scores("when did I move?", ["a", "b"]) is None


@pytest.mark.parametrize("exc", FAILURES, ids=lambda e: type(e).__name__)
def test_embedder_gives_up_quietly(exc):
    emb = HttpEmbedder("text-embedding-v4", "k", dims=4, rps=1000, retries=1)
    with mock.patch("urllib.request.urlopen", _raising(exc)), mock.patch("time.sleep"):
        assert emb.embed(["hello", "world"]) == [None, None]
    assert emb.failures == 1
