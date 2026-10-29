"""Ranking behaviours introduced in v0.2 (each maps to a line in docs/METHOD_CARD.md)."""
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.store import MemoryStore

DAY = 86_400_000


def make_client(**overrides):
    tmp = tempfile.TemporaryDirectory()
    settings = Settings(api_key="", data_dir=os.path.join(tmp.name, "d"), **overrides)
    app = create_app(settings=settings, store=MemoryStore(settings.data_dir))
    client = TestClient(app)
    client.__enter__()
    client._tmp = tmp  # keep the directory alive
    return client


@pytest.fixture()
def client():
    c = make_client()
    yield c
    c.__exit__(None, None, None)


def add(c, rid, user, msgs, session="s"):
    r = c.post("/v1/memories/add", json={"request_id": rid, "user_id": user, "session_id": session, "messages": msgs})
    assert r.status_code == 200, r.text
    return r


def search(c, user, query, top_k=100, options=None):
    body = {"query": query, "user_id": user, "top_k": top_k}
    if options:
        body["options"] = options
    r = c.post("/v1/memories/search", json=body)
    assert r.status_code == 200, r.text
    return r.json()["data"]


def test_porter_stemming_matches_inflections(client):
    add(client, "c:chunk-0", "u", [{"role": "user", "content": "After the book I've reset my target to 4,000 euros a month on average."},
                                  {"role": "user", "content": "Pixel watched me stretch with deep suspicion."}])
    data = search(client, "u", "What monthly income is she targeting?")
    assert data and "4,000 euros" in data[0]["content"]


def test_span_crosses_add_chunk_boundary(client):
    first = [{"role": "user", "content": f"filler message number {i} about gardening tomatoes."} for i in range(19)]
    first.append({"role": "user", "content": "The vet visit for the cat cost 85 euros in May."})
    add(client, "c:chunk-0", "u", first)
    add(client, "c:chunk-1", "u", [{"role": "user", "content": "And the second vet visit, a tooth extraction, cost 120 euros."},
                                  {"role": "assistant", "content": "Poor cat, two visits in one summer."}])
    data = search(client, "u", "tooth extraction cost")
    top = data[0]["content"]
    assert "120 euros" in top and "85 euros" in top, top  # previous turn comes from the earlier chunk


def test_overlapping_hits_are_not_returned_twice(client):
    msgs = [{"role": "user", "content": "Tomato seedlings need sun."},
            {"role": "assistant", "content": "Tomato plants want six hours of sun."},
            {"role": "user", "content": "My tomato bed faces east, is that enough sun?"},
            {"role": "assistant", "content": "East-facing beds get morning sun, usually fine for tomato."}]
    add(client, "c:chunk-0", "u", msgs)
    data = search(client, "u", "tomato sun")
    joined = "\n".join(d["content"] for d in data)
    assert len(data) < 4 and joined.count("seedlings need sun") == 1


def test_recency_band_prefers_newer_near_equal_match(client):
    t0 = 1_700_000_000_000
    add(client, "c:chunk-0", "u", [{"role": "user", "timestamp": t0, "content": "The Paperlantern contract covers 24 illustrations."}])
    add(client, "c:chunk-1", "u", [{"role": "user", "timestamp": t0 + 30 * DAY, "content": "The Paperlantern contract now covers 30 illustrations."}])
    data = search(client, "u", "How many illustrations does the Paperlantern contract cover?")
    assert "30 illustrations" in data[0]["content"]


def test_recency_band_uses_ordinal_when_no_timestamps(client):
    add(client, "c:chunk-0", "u", [{"role": "user", "content": "Budget for the lease is 60,000 rupees a month."}])
    add(client, "c:chunk-1", "u", [{"role": "user", "content": "Signed the lease today: 52,000 rupees a month, three years."}])
    data = search(client, "u", "How much is the monthly lease?")
    assert "52,000" in data[0]["content"]


def test_cjk_query_uses_trigram_index(client):
    add(client, "c:chunk-0", "u", [{"role": "user", "content": "我养了一只柯基，叫年糕，两岁了。"},
                                  {"role": "user", "content": "德国站的自动喂食器定价 59 欧元。"}])
    data = search(client, "u", "小雨养的那只柯基叫什么名字？")
    assert data and "年糕" in data[0]["content"]


def test_cjk_rule_is_tagged_and_offered():
    c = make_client()
    add(c, "c:chunk-0", "u", [{"role": "user", "content": "以后给我写的所有文案都用简体中文，不要夹杂英文缩写。"}])
    for i in range(10):
        add(c, f"c:chunk-{i+1}", "u", [{"role": "user", "content": f"第{i}批订单发往波兰，共 {i * 100} 台。"}])
    data = search(c, "u", "波兰订单的数量")
    assert any("简体中文" in d["content"] for d in data)
    c.__exit__(None, None, None)


def test_rules_sit_after_lead_hits_not_before():
    c = make_client(context_window=0)
    add(c, "c:chunk-0", "u", [{"role": "user", "content": "From now on always answer me in French."}])
    for i in range(1, 8):
        add(c, f"c:chunk-{i}", "u", [{"role": "user", "content": f"Gardening note {i}: tomatoes like warm soil and mulch."}])
    data = search(c, "u", "tomatoes soil")
    idx = next(i for i, d in enumerate(data) if "French" in d["content"])
    assert idx == 5, [d["content"][:30] for d in data]  # LEAD_HITS=5 tomato hits first, then the rule
    c.__exit__(None, None, None)


def test_matching_rule_outranks_newer_unrelated_rule(client):
    t0 = 1_700_000_000_000
    add(client, "c:chunk-0", "u", [{"role": "user", "timestamp": t0, "content": "When I ask for training plans, give them as a table with Day, Distance and Pace columns."}])
    add(client, "c:chunk-1", "u", [{"role": "user", "timestamp": t0 + DAY, "content": "From now on always answer me in French."}])
    add(client, "c:chunk-2", "u", [{"role": "user", "timestamp": t0 + 2 * DAY, "content": "Please never suggest shellfish recipes."}])
    data = search(client, "u", "Give me next week's training plan.")
    assert "Pace columns" in data[0]["content"]


def test_response_char_budget_caps_results():
    c = make_client(max_response_chars=10_000, context_window=0)
    for i in range(60):
        add(c, f"c:chunk-{i}", "u", [{"role": "user", "content": f"Entry {i}: " + "tomato " * 100}])
    data = search(c, "u", "tomato", top_k=100)
    assert 0 < len(data) < 60
    assert sum(len(d["content"]) for d in data) < 10_000 + 1_000
    c.__exit__(None, None, None)


def test_unknown_user_creates_no_file_and_old_schema_is_refused(tmp_path):
    import sqlite3
    store = MemoryStore(str(tmp_path))
    assert store.count("nobody") == 0 and not store.path_for("nobody").exists()
    path = store.path_for("someone")
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA user_version = 1")
    with pytest.raises(RuntimeError):
        with store.connect("someone"):
            pass


def test_users_live_in_separate_files(client):
    add(client, "a:chunk-0", "alice", [{"role": "user", "content": "alpha"}])
    add(client, "b:chunk-0", "bob", [{"role": "user", "content": "beta"}])
    store = client.app.state.store
    assert store.path_for("alice") != store.path_for("bob")
    assert store.path_for("alice").exists() and store.path_for("bob").exists()
