"""The optional evidence selector (v0.6): selected blocks move to the front, 'fact' tails may be cut, rules survive,
failures and nonsense leave the deterministic order untouched, nothing is generated."""
import os
import tempfile

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.selector import Selection, parse_selection
from app.store import MemoryStore

T0 = 1_700_000_000_000
DAY = 86_400_000


class FakeSelector:
    def __init__(self, selection):
        self.selection = selection
        self.calls = []

    def select(self, query, options, blocks):
        self.calls.append((query, options, list(blocks)))
        return self.selection


def make_client(selector, **overrides):
    tmp = tempfile.TemporaryDirectory()
    settings = Settings(api_key="", data_dir=os.path.join(tmp.name, "d"), **overrides)
    app = create_app(settings=settings, store=MemoryStore(settings.data_dir), selector=selector)
    client = TestClient(app)
    client.__enter__()
    client._tmp = tmp
    return client


def seed(c, n=20):
    for i in range(n):
        c.post("/v1/memories/add", json={"request_id": f"c:chunk-{i}", "user_id": "u", "session_id": "s",
                                         "messages": [{"role": "user", "timestamp": T0 + i * DAY,
                                                       "content": f"Lin: tomato note number {i}, the plants grew again."}]})
    c.post("/v1/memories/add", json={"request_id": "c:chunk-rule", "user_id": "u", "session_id": "s",
                                     "messages": [{"role": "user", "timestamp": T0 + 40 * DAY,
                                                   "content": "Lin: From now on call me Captain when you reply."}]})


def search(c, **extra):
    return c.post("/v1/memories/search", json={"query": "tomato plants grew", "user_id": "u", "top_k": 100, **extra}).json()["data"]


def test_parse_selection_is_tolerant():
    sel = parse_selection('```json\n{"blocks": ["3", 1, 99, 3, "x"], "kind": "Fact"}\n```', 10, 8)
    assert sel.blocks == [2, 0] and sel.kind == "fact"
    assert parse_selection("no json", 10, 8).blocks == [] and parse_selection("{}", 10, 8).kind == "other"
    assert parse_selection('{"blocks": [1,2,3,4], "kind": "task"}', 10, 2).blocks == [0, 1]


def test_fact_selection_moves_blocks_first_and_cuts_tail_but_keeps_rules():
    selector = FakeSelector(None)   # None = deterministic order first
    c = make_client(selector, context_window=0, selector_tail=2, selector_min_results=3)
    seed(c)
    before = search(c)
    selector.selection = Selection(blocks=[4, 1], kind="fact")
    after = search(c)
    assert len(before) >= 12 and selector.calls and len(selector.calls[-1][2]) == min(40, len(before))
    assert after[0]["id"] == before[4]["id"] and after[1]["id"] == before[1]["id"]
    rule = [r for r in before if "call me Captain" in r["content"]]
    assert rule and any(r["id"] == rule[0]["id"] for r in after), "standing rule must survive the cut"
    non_rule_tail = [r for r in after[2:] if "call me Captain" not in r["content"]]
    assert len(non_rule_tail) == 2
    scores = [r["score"] for r in after]
    assert scores == sorted(scores, reverse=True) and scores[0] == 1.0
    assert {r["content"] for r in after} <= {r["content"] for r in before}


def test_task_kind_reorders_only():
    selector = FakeSelector(None)
    c = make_client(selector, context_window=0, selector_tail=0, selector_min_results=3)
    seed(c)
    before = search(c)
    selector.selection = Selection(blocks=[3], kind="task")
    after = search(c)
    assert len(after) == len(before)
    assert after[0]["id"] == before[3]["id"]
    assert [r["id"] for r in after[1:]] == [r["id"] for r in before if r["id"] != before[3]["id"]]


def test_reorder_only_when_tail_is_minus_one():
    selector = FakeSelector(None)
    c = make_client(selector, context_window=0, selector_tail=-1, selector_min_results=3)
    seed(c)
    before = search(c)
    selector.selection = Selection(blocks=[2], kind="fact")
    after = search(c)
    assert len(after) == len(before) and after[0]["id"] == before[2]["id"]


def test_failure_empty_and_invalid_selections_keep_order():
    for selection in (None, Selection(blocks=[], kind="fact"), Selection(blocks=[999], kind="fact")):
        c = make_client(FakeSelector(selection), context_window=0, selector_tail=0, selector_min_results=3)
        seed(c)
        before = search(c)
        after = search(c)
        assert [r["id"] for r in after] == [r["id"] for r in before]
        assert [r["score"] for r in after] == [r["score"] for r in before]


def test_selector_not_called_for_short_lists():
    selector = FakeSelector(Selection(blocks=[0], kind="fact"))
    c = make_client(selector, context_window=0, selector_min_results=50)
    seed(c, n=5)
    search(c)
    assert selector.calls == []
