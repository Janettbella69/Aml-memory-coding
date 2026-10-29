"""The optional cross-encoder reranker (v0.7): reorders the head by relevance (replace) or by RRF, keeps the tail,
re-issues monotonic scores, and leaves the order untouched on failure."""
import os
import tempfile

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.store import MemoryStore

T0 = 1_700_000_000_000
DAY = 86_400_000


class FakeReranker:
    def __init__(self, scores):
        self.scores_value = scores
        self.calls = []

    def scores(self, query, documents):
        self.calls.append((query, list(documents)))
        return self.scores_value


def make_client(reranker, **overrides):
    tmp = tempfile.TemporaryDirectory()
    settings = Settings(api_key="", data_dir=os.path.join(tmp.name, "d"), **overrides)
    app = create_app(settings=settings, store=MemoryStore(settings.data_dir), reranker=reranker)
    client = TestClient(app)
    client.__enter__()
    client._tmp = tmp
    return client


def seed(c, n=12):
    for i in range(n):
        c.post("/v1/memories/add", json={"request_id": f"c:chunk-{i}", "user_id": "u", "session_id": "s",
                                         "messages": [{"role": "user", "timestamp": T0 + i * DAY,
                                                       "content": f"Lin: tomato note number {i}, the plants grew again."}]})


def search(c):
    return c.post("/v1/memories/search", json={"query": "tomato plants grew", "user_id": "u", "top_k": 100}).json()["data"]


def test_replace_mode_orders_head_by_relevance_and_keeps_tail():
    rr = FakeReranker(None)
    c = make_client(rr, context_window=0, rerank_candidates=4, rerank_min_results=3)
    seed(c)
    before = search(c)
    rr.scores_value = [0.1, 0.9, 0.5, 0.7]
    after = search(c)
    assert len(rr.calls[-1][1]) == 4 and len(after) == len(before)
    assert [r["id"] for r in after[:4]] == [before[1]["id"], before[3]["id"], before[2]["id"], before[0]["id"]]
    assert [r["id"] for r in after[4:]] == [r["id"] for r in before[4:]]
    scores = [r["score"] for r in after]
    assert scores == sorted(scores, reverse=True) and scores[0] == 1.0
    assert {r["content"] for r in after} == {r["content"] for r in before}


def test_rrf_mode_fuses_retrieval_and_reranker_ranks():
    rr = FakeReranker(None)
    c = make_client(rr, context_window=0, rerank_candidates=4, rerank_min_results=3, rerank_mode="rrf")
    seed(c)
    before = search(c)
    rr.scores_value = [0.1, 0.9, 0.5, 0.7]   # reranker: 1, 3, 2, 0 ; retrieval: 0, 1, 2, 3
    after = search(c)
    # block 1 is 2nd by retrieval and 1st by reranker -> fused first; block 0 (1st/4th) and block 3 (4th/2nd) tie on RRF -> retrieval order breaks it
    assert after[0]["id"] == before[1]["id"]
    assert [r["id"] for r in after[:4]] == [before[1]["id"], before[0]["id"], before[3]["id"], before[2]["id"]]


def test_failure_short_or_bad_scores_keep_order():
    for scores in (None, [], [0.5] * 999):
        rr = FakeReranker(scores)
        c = make_client(rr, context_window=0, rerank_candidates=4, rerank_min_results=3)
        seed(c)
        before = search(c)
        after = search(c)
        assert [r["id"] for r in after] == [r["id"] for r in before]


def test_reranker_not_called_for_short_lists():
    rr = FakeReranker([0.5, 0.4])
    c = make_client(rr, context_window=0, rerank_min_results=50)
    seed(c, n=3)
    search(c)
    assert rr.calls == []


def test_http_reranker_maps_every_provider(monkeypatch):
    """Request bodies follow each provider's schema and {index, relevance_score} results map back to input order."""
    import io
    import json as _json

    import app.rerank as rr

    captured = {}

    class Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["body"] = _json.loads(req.data.decode())
        captured["auth"] = req.get_header("Authorization")
        p = captured["provider"]
        items = [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.2}]
        payload = {"output": {"results": items}} if p == "dashscope" else {"data": items} if p == "voyage" else {"results": items}
        return Resp(_json.dumps(payload).encode())

    monkeypatch.setattr(rr.urllib.request, "urlopen", fake_urlopen)
    for provider, model in (("dashscope", "gte-rerank-v2"), ("voyage", "rerank-2.5"), ("cohere", "rerank-v3.5"), ("jina", "jina-reranker-v3")):
        captured.clear(); captured["provider"] = provider
        r = rr.HttpReranker(provider, model, "k-" + provider, doc_chars=5)
        assert r.scores("q", ["aaaaaaaa", "bb"]) == [0.2, 0.9]
        assert captured["auth"] == "Bearer k-" + provider and captured["url"] == rr.PROVIDERS[provider]["endpoint"]
        b = captured["body"]
        docs = b["input"]["documents"] if provider == "dashscope" else b["documents"]
        assert docs == ["aaaaa", "bb"] and b["model"] == model
        assert (b["input"]["query"] if provider == "dashscope" else b["query"]) == "q"
        assert (b["parameters"]["top_n"] if provider == "dashscope" else b.get("top_k", b.get("top_n"))) == 2


def test_http_reranker_failure_returns_none(monkeypatch):
    import app.rerank as rr

    def boom(req, timeout=0):
        raise rr.urllib.error.URLError("down")

    monkeypatch.setattr(rr.urllib.request, "urlopen", boom)
    assert rr.HttpReranker("voyage", "rerank-2.5", "k").scores("q", ["a"]) is None
    assert rr.HttpReranker("voyage", "rerank-2.5", "k").scores("q", []) is None


def test_reranker_from_env_keys(monkeypatch):
    import app.rerank as rr

    monkeypatch.delenv("RERANK_API_KEY", raising=False)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    assert rr.reranker_from_env("voyage", "rerank-2.5", 20.0, 1500) is None
    monkeypatch.setenv("VOYAGE_API_KEY", "v-key")
    r = rr.reranker_from_env("voyage", "rerank-2.5", 20.0, 1500)
    assert isinstance(r, rr.HttpReranker) and r.api_key == "v-key" and r.endpoint == rr.PROVIDERS["voyage"]["endpoint"]
    monkeypatch.setenv("RERANK_API_KEY", "override")
    monkeypatch.setenv("RERANK_ENDPOINT", "https://proxy.example/rerank")
    r = rr.reranker_from_env("voyage", "rerank-2.5", 20.0, 1500)
    assert r.api_key == "override" and r.endpoint == "https://proxy.example/rerank"
    assert rr.reranker_from_env("nope", "x", 20.0, 1500) is None and rr.reranker_from_env("voyage", "", 20.0, 1500) is None


def test_hit_mode_sends_the_matched_turn_not_the_span():
    rr = FakeReranker([0.5, 0.4, 0.3, 0.2])
    c = make_client(rr, context_window=2, rerank_candidates=4, rerank_min_results=3, rerank_doc="hit")
    seed(c)
    search(c)
    docs = rr.calls[-1][1]
    assert all(d.count("\n") == 0 and "tomato note" in d for d in docs), docs   # one rendered turn each
    rr2 = FakeReranker([0.5, 0.4, 0.3, 0.2])
    c2 = make_client(rr2, context_window=2, rerank_candidates=4, rerank_min_results=3, rerank_doc="block")
    seed(c2)
    search(c2)
    assert any(d.count("\n") >= 2 for d in rr2.calls[-1][1])                   # spans carry neighbouring turns


def test_created_at_is_the_event_time_when_known():
    from datetime import datetime, timezone
    c = make_client(None, context_window=0)
    seed(c, n=3)
    c.post("/v1/memories/add", json={"request_id": "c:chunk-nots", "user_id": "u", "session_id": "s",
                                     "messages": [{"role": "user", "content": "Lin: tomato plants without a timestamp."}]})
    data = search(c)
    stamped = [r for r in data if "without a timestamp" not in r["content"]]
    assert stamped and all(r["created_at"].endswith("Z") and r["created_at"].startswith(datetime.fromtimestamp(T0 / 1000, tz=timezone.utc).strftime("%Y-%m-%d")[:4]) for r in stamped)
    assert stamped[0]["created_at"] == datetime.fromtimestamp((T0 + int(stamped[0]["content"].split("number ")[1].split(",")[0]) * DAY) / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    c.post("/v1/memories/add", json={"request_id": "x:chunk-0", "user_id": "bare", "session_id": "s",
                                     "messages": [{"role": "user", "content": "Juror 8: a line that came without any timestamp."}]})
    bare = c.post("/v1/memories/search", json={"query": "line without timestamp", "user_id": "bare", "top_k": 5}).json()["data"]
    assert bare and "created_at" not in bare[0] and bare[0]["id"] and bare[0]["score"] is not None   # omitted, not null, never the storage day


def test_default_document_is_the_matched_turn_followed_by_its_span(monkeypatch):
    from app.config import Settings
    monkeypatch.delenv("RERANK_DOC", raising=False)
    assert Settings().rerank_doc == "both" and Settings.from_env().rerank_doc == "both"
    rr = FakeReranker([0.5, 0.4, 0.3, 0.2])
    c = make_client(rr, context_window=2, rerank_candidates=4, rerank_min_results=3)
    seed(c)
    search(c)
    docs = rr.calls[-1][1]
    assert all("tomato note" in d.split("\n", 1)[0] for d in docs), docs      # every document opens with the matched turn
    assert any(d.count("\n") >= 2 for d in docs)                              # and carries its neighbours behind it


def test_rerank_document_modes():
    from app.search import rerank_document
    hit, block = "USER: the answer line", "USER: before\nUSER: the answer line\nUSER: after"
    assert rerank_document(hit, block, "block") == block
    assert rerank_document(hit, block, "hit") == hit
    assert rerank_document(hit, block, "both") == hit + "\n" + block      # the hit always opens the reranker's window
    assert rerank_document(hit, hit, "both") == hit
    assert rerank_document(None, block, "hit") == block and rerank_document(None, block, "both") == block
