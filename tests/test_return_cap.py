"""v0.9 RETURN_CHARS: the final cut runs after the reranker (which still sees the whole head), keeps whole blocks, keeps
standing rules that fall behind it, and with RETURN_FROM_CHARS leaves small memories whole."""
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


def make_client(reranker=None, **overrides):
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


def search(c, query="tomato plants grew"):
    return c.post("/v1/memories/search", json={"query": query, "user_id": "u", "top_k": 100}).json()["data"]


def test_cut_runs_after_the_reranker_and_keeps_whole_blocks():
    rr = FakeReranker(None)
    c = make_client(rr, context_window=0, rerank_candidates=4, rerank_min_results=3)
    seed(c)
    full = search(c)
    rr.scores_value = [0.1, 0.9, 0.5, 0.7]
    reranked = search(c)
    one = len(reranked[0]["content"])
    cut_client = make_client(rr, context_window=0, rerank_candidates=4, rerank_min_results=3, return_chars=one + 1)
    seed(cut_client)
    cut = search(cut_client)
    assert len(rr.calls[-1][1]) == 4                      # the reranker still saw the whole head
    assert [r["id"] for r in cut] == [r["id"] for r in reranked[:2]]
    assert [r["content"] for r in cut] == [r["content"] for r in reranked[:2]]   # whole blocks, never sliced
    assert len(full) == 12


def test_standing_rules_behind_the_cut_are_kept():
    c = make_client(context_window=0, return_chars=10)
    seed(c, n=8)
    c.post("/v1/memories/add", json={"request_id": "c:chunk-rule", "user_id": "u", "session_id": "s",
                                     "messages": [{"role": "user", "timestamp": T0 + 20 * DAY,
                                                   "content": "From now on always answer me in French."}]})
    data = search(c)
    assert len(data) == 2
    assert "tomato" in data[0]["content"] and "French" in data[1]["content"]


def test_small_memories_are_not_cut_with_return_from_chars():
    small = make_client(context_window=0, return_chars=10, return_from_chars=1_000_000)
    seed(small)
    assert len(search(small)) == 12
    large = make_client(context_window=0, return_chars=10, return_from_chars=100)
    seed(large)
    assert len(search(large)) == 1


def test_off_by_default():
    c = make_client(context_window=0)
    seed(c)
    assert len(search(c)) == 12
