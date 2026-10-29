"""v0.5 dense channel with the deterministic FakeEmbedder: fusion, degradation, backfill, identity, date range."""
import os
import tempfile
from datetime import datetime, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.embed import FakeEmbedder, from_blob, to_blob
from app.main import create_app
from app.store import MemoryStore
from tests.test_search_v02 import add, search

TS = int(datetime(2025, 4, 27, 7, 50, tzinfo=timezone.utc).timestamp() * 1000)
DAY = 86_400_000


def make(embedder, **overrides):
    tmp = tempfile.TemporaryDirectory()
    settings = Settings(api_key="", data_dir=os.path.join(tmp.name, "d"), **overrides)
    store = MemoryStore(settings.data_dir)
    app = create_app(settings=settings, store=store, embedder=embedder)
    client = TestClient(app)
    client.__enter__()
    client._tmp = tmp
    client._store = store
    return client


def test_blob_roundtrip():
    v = np.random.default_rng(1).standard_normal(256).astype(np.float32)
    v /= np.linalg.norm(v)
    back = from_blob(to_blob(v), 256)
    assert back.shape == (256,) and np.allclose(v, back, atol=2e-3)


def test_dense_finds_paraphrase_without_lexical_overlap():
    # "puppy" and "dog" share no characters; the fake embedder maps puppy -> dog so the vectors are close.
    emb = FakeEmbedder(aliases={"puppy": "dog", "named": "called"})
    c = make(emb)
    try:
        add(c, "c:chunk-0", "u", [{"role": "user", "content": "Tao has a dog called Biscuit."},
                                 {"role": "user", "content": "The invoice table lock took the API down."},
                                 {"role": "user", "content": "Lin passed the critical care exam."}])
        data = search(c, "u", "What is the puppy named?")
        assert data and "Biscuit" in data[0]["content"]
        assert emb.calls >= 2   # one Add batch + one query
    finally:
        c.__exit__(None, None, None)


def test_without_embedder_nothing_changes():
    c = make(None)
    try:
        add(c, "c:chunk-0", "u", [{"role": "user", "content": "Tao has a dog called Biscuit."}])
        assert search(c, "u", "dog called")[0]["content"].count("Biscuit") == 1   # lexical path unchanged
        with c._store.connect("u") as conn:
            assert conn.execute("SELECT COUNT(*) FROM memories WHERE vec IS NULL").fetchone()[0] == 1
    finally:
        c.__exit__(None, None, None)


def test_failure_degrades_and_backfill_repairs():
    emb = FakeEmbedder(fail=True)
    c = make(emb)
    try:
        add(c, "c:chunk-0", "u", [{"role": "user", "content": "Tao has a dog called Biscuit."},
                                 {"role": "user", "content": "Lin passed the exam."}])
        assert "u" in c._store.vector_failures
        assert search(c, "u", "dog called")[0]["content"].count("Biscuit") == 1   # lexical path still works
        with c._store.connect("u") as conn:
            assert conn.execute("SELECT COUNT(*) FROM memories WHERE vec IS NULL").fetchone()[0] == 2
        emb.fail = False
        assert c._store.backfill_vectors(emb) == 2
        assert "u" not in c._store.vector_failures
        with c._store.connect("u") as conn:
            assert conn.execute("SELECT COUNT(*) FROM memories WHERE vec IS NULL").fetchone()[0] == 0
    finally:
        c.__exit__(None, None, None)


def test_replay_skips_embedding_and_identity_is_enforced():
    emb = FakeEmbedder()
    c = make(emb)
    try:
        msgs = [{"role": "user", "content": "Tao has a dog called Biscuit."}]
        add(c, "c:chunk-0", "u", msgs)
        calls = emb.calls
        add(c, "c:chunk-0", "u", msgs)          # idempotent replay
        assert emb.calls == calls               # not embedded again
        with c._store.connect("u") as conn:
            assert conn.execute("SELECT value FROM meta WHERE key='embed'").fetchone()[0] == "fake:32"
        other = FakeEmbedder(dims=16)
        with pytest.raises(RuntimeError):
            c._store.add(__import__("app.models", fromlist=["AddRequest"]).AddRequest(
                request_id="c:chunk-1", user_id="u", session_id="s", messages=msgs),
                vectors=[to_blob(np.ones(16, dtype=np.float32))], embed_identity=(other.model, other.dims))
    finally:
        c.__exit__(None, None, None)


def test_dense_search_respects_date_range_and_cache_invalidation():
    emb = FakeEmbedder()
    c = make(emb)
    try:
        add(c, "c:chunk-0", "u", [{"role": "user", "content": "Biscuit the dog chased birds.", "timestamp": TS},
                                 {"role": "user", "content": "Biscuit the dog slept all day.", "timestamp": TS + 10 * DAY}])
        q = emb.embed_one("Biscuit the dog")
        all_hits = c._store.dense_search("u", q, 10)
        assert len(all_hits) == 2
        late = c._store.dense_search("u", q, 10, event_range=(TS + 5 * DAY, None))
        assert [h[0].content for h in late] == ["Biscuit the dog slept all day."]
        add(c, "c:chunk-1", "u", [{"role": "user", "content": "Biscuit the dog ate a sleeve.", "timestamp": TS + 20 * DAY}])
        assert len(c._store.dense_search("u", q, 10)) == 3   # cache dropped on write
    finally:
        c.__exit__(None, None, None)


def test_unknown_user_dense_search_is_empty():
    c = make(FakeEmbedder())
    try:
        assert search(c, "nobody", "anything") == []
    finally:
        c.__exit__(None, None, None)


def test_dense_head_keeps_lexical_order_in_front():
    emb = FakeEmbedder(aliases={"puppy": "dog"})
    msgs = [{"role": "user", "content": "Tao has a dog called Biscuit."},
            {"role": "user", "content": "The puppy ate a cable sleeve."},
            {"role": "user", "content": "Lin passed the exam."}]
    for head in (0, 1):
        c = make(emb, dense_head=head)
        try:
            add(c, "c:chunk-0", "u", msgs)
            data = search(c, "u", "exam Lin passed", top_k=10)
            assert "exam" in data[0]["content"]          # the exact lexical hit leads either way here
        finally:
            c.__exit__(None, None, None)


def test_planner_expansion_keeps_lexical_head():
    from app.planner import Plan

    class FakePlanner:
        def plan(self, query, options, snippets):
            return Plan(queries=["cable sleeve"], date_from_ms=None, date_to_ms=None)

    tmp = tempfile.TemporaryDirectory()
    settings = Settings(api_key="", data_dir=os.path.join(tmp.name, "d"), dense_head=1)
    store = MemoryStore(settings.data_dir)
    app = create_app(settings=settings, store=store, planner=FakePlanner())
    c = TestClient(app)
    c.__enter__()
    try:
        add(c, "c:chunk-0", "u", [{"role": "user", "content": "Lin passed the critical care exam on 28 September."}])
        add(c, "c:chunk-1", "u", [{"role": "user", "content": "Biscuit ate a cable sleeve again."}])
        add(c, "c:chunk-2", "u", [{"role": "user", "content": "The cable sleeve was replaced by a metal cable cover."}])
        data = search(c, "u", "When did Lin pass the exam?", top_k=10)
        assert "exam" in data[0]["content"]      # the planner's extra query must not push the lexical head down
    finally:
        c.__exit__(None, None, None)
        tmp.cleanup()
