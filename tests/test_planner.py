"""The optional query planner (v0.3): expansion queries are fused in, failures fall back, nothing is generated."""
import os
import tempfile

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.planner import Plan, parse_plan
from app.store import MemoryStore

DAY = 86_400_000
T0 = 1_700_000_000_000


class FakePlanner:
    def __init__(self, plan):
        self.plan_value = plan
        self.calls = []

    def plan(self, query, options, snippets):
        self.calls.append((query, options, snippets))
        return self.plan_value


def make_client(planner, **overrides):
    tmp = tempfile.TemporaryDirectory()
    settings = Settings(api_key="", data_dir=os.path.join(tmp.name, "d"), **overrides)
    app = create_app(settings=settings, store=MemoryStore(settings.data_dir), planner=planner)
    client = TestClient(app)
    client.__enter__()
    client._tmp = tmp
    return client


def seed(c):
    turns = [
        {"role": "user", "timestamp": T0, "content": "Lin: How is the game going?"},
        {"role": "assistant", "timestamp": T0 + 1, "content": "Tao: Slowly. Biscuit sleeps on my keyboard all day, he is a dog with opinions."},
        {"role": "user", "timestamp": T0 + 2, "content": "Lin: Classic. I am applying to a hospital in Hangzhou."},
        {"role": "assistant", "timestamp": T0 + 3, "content": "Tao: Hangzhou is forty minutes from Shanghai by train, I could visit every month."},
    ]
    for i, t in enumerate(turns):
        c.post("/v1/memories/add", json={"request_id": f"c:chunk-{i}", "user_id": "u", "session_id": "s", "messages": [t]})
    for i in range(4, 30):
        c.post("/v1/memories/add", json={"request_id": f"c:chunk-{i}", "user_id": "u", "session_id": "s",
                                         "messages": [{"role": "user", "timestamp": T0 + i * DAY, "content": f"Lin: gardening note {i} about tomatoes."}]})


def test_parse_plan_is_tolerant():
    plan = parse_plan('```json\n{"queries": ["Tao Shanghai", "Tao Shanghai", " Tao city "], "date_from": "2025-05-01", "date_to": "bad"}\n```', 3)
    assert plan.queries == ["Tao Shanghai", "Tao city"]
    assert plan.date_from_ms is not None and plan.date_to_ms is None
    assert parse_plan("no json here", 3).queries == []


def test_planner_queries_bring_in_second_hop_evidence():
    planner = FakePlanner(Plan(queries=["Tao Shanghai train"]))
    c = make_client(planner, context_window=0)
    seed(c)
    data = c.post("/v1/memories/search", json={"query": "In which city does the owner of Biscuit live?", "user_id": "u", "top_k": 5}).json()["data"]
    joined = " ".join(d["content"] for d in data)
    assert "Shanghai" in joined and "Biscuit" in joined
    assert planner.calls and planner.calls[0][0].startswith("In which city")
    assert any("Biscuit" in s for s in planner.calls[0][2])      # first-pass snippets were shown to the planner
    assert all(d["content"].split(": ", 1)[1].startswith(("Lin:", "Tao:")) for d in data)  # only stored turns are returned
    c.__exit__(None, None, None)


def test_planner_failure_falls_back_to_deterministic_results():
    c = make_client(FakePlanner(None), context_window=0)
    seed(c)
    data = c.post("/v1/memories/search", json={"query": "Biscuit keyboard", "user_id": "u", "top_k": 5}).json()["data"]
    assert data and "Biscuit" in data[0]["content"]
    c.__exit__(None, None, None)


def test_date_range_moves_in_range_hits_first():
    plan = Plan(queries=[])
    plan.date_from_ms, plan.date_to_ms = T0 + 20 * DAY, T0 + 25 * DAY
    c = make_client(FakePlanner(plan), context_window=0)
    seed(c)
    data = c.post("/v1/memories/search", json={"query": "gardening tomatoes", "user_id": "u", "top_k": 5}).json()["data"]
    assert all("note 2" in d["content"] for d in data[:3]), [d["content"][:40] for d in data]   # notes 20..25 first
    c.__exit__(None, None, None)


def test_no_planner_when_model_not_configured(tmp_path):
    settings = Settings(api_key="", data_dir=str(tmp_path), planner_model="")
    app = create_app(settings=settings, store=MemoryStore(settings.data_dir))
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200
