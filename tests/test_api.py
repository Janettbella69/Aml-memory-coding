import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.store import MemoryStore


@pytest.fixture()
def client():
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(api_key="", data_dir=os.path.join(tmp, "d"), context_window=1, rule_slots=4)
        app = create_app(settings=settings, store=MemoryStore(settings.data_dir))
        with TestClient(app) as c:
            yield c


def add(c, rid, user, session, msgs):
    return c.post("/v1/memories/add", json={"request_id": rid, "user_id": user, "session_id": session, "messages": msgs})


def test_health(client):
    assert client.get("/health").status_code == 200


def test_add_echo_and_search(client):
    r = add(client, "eval:r:conv-0:chunk-0", "u1", "s1",
            [{"role": "user", "timestamp": 1704067200000, "content": "I adopted a cat named Luna."},
             {"role": "assistant", "content": "Luna is a lovely name."}])
    assert r.status_code == 200
    assert r.json() == {"success": True, "request_id": "eval:r:conv-0:chunk-0", "user_id": "u1", "session_id": "s1"}
    s = client.post("/v1/memories/search", json={"query": "What is my cat called?", "user_id": "u1", "top_k": 5})
    data = s.json()["data"]
    assert data and "Luna" in data[0]["content"]
    assert data[0]["content"].startswith("[2024-01-01 00:00 UTC] USER:")


def test_idempotent_and_conflict(client):
    msgs = [{"role": "user", "content": "hello world"}]
    assert add(client, "rid-1", "u1", "s1", msgs).status_code == 200
    assert add(client, "rid-1", "u1", "s1", msgs).status_code == 200
    assert add(client, "rid-1", "u1", "s1", [{"role": "user", "content": "different"}]).status_code == 409


def test_user_isolation(client):
    add(client, "a", "alice", "s", [{"role": "user", "content": "My password hint is purple elephant."}])
    add(client, "b", "bob", "s", [{"role": "user", "content": "I like sushi."}])
    s = client.post("/v1/memories/search", json={"query": "purple elephant", "user_id": "bob", "top_k": 10})
    assert all("elephant" not in m["content"] for m in s.json()["data"])


def test_rules_ride_along_and_top_k_cap(client):
    add(client, "r0", "u", "s", [{"role": "user", "content": "From now on always answer me in French."}])
    for i in range(30):
        add(client, f"r{i+1}", "u", "s", [{"role": "user", "content": f"Note number {i} about gardening tomatoes."}])
    s = client.post("/v1/memories/search", json={"query": "tomatoes", "user_id": "u", "top_k": 8})
    data = s.json()["data"]
    assert len(data) <= 8
    assert any("French" in m["content"] for m in data)


def test_empty_user_returns_empty_list(client):
    s = client.post("/v1/memories/search", json={"query": "anything", "user_id": "nobody", "top_k": 3})
    assert s.status_code == 200 and s.json()["data"] == []


def test_options_and_multimodal_content(client):
    add(client, "m0", "u2", "s", [{"role": "user", "content": [{"type": "text", "text": "We met at the Berlin marathon."},
                                                             {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}])
    s = client.post("/v1/memories/search", json={"query": "Where did we meet?", "options": ["A. Berlin", "B. Paris"],
                                                 "user_id": "u2", "top_k": 5})
    assert "Berlin" in s.json()["data"][0]["content"]
