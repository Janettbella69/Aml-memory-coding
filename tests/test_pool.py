"""Connection cache bounds: the pool never exceeds its size, idle connections are closed, data survive eviction."""
import time

from app.models import AddRequest
from app.store import MemoryStore


def _add(store: MemoryStore, user: str, text: str, n: int = 0) -> None:
    store.add(AddRequest(request_id=f"{user}:chunk-{n}", user_id=user, session_id="s",
                         messages=[{"role": "user", "content": text}]))


def test_pool_is_bounded_and_evicted_users_stay_readable(tmp_path):
    store = MemoryStore(str(tmp_path), pool_size=4)
    for i in range(12):
        _add(store, f"user-{i}", f"note for user {i} about tomatoes")
    assert len(store._pool) <= 4
    assert store.count("user-0") == 1            # evicted long ago, reopened transparently
    hits = store.fts_search("user-0", '"tomatoes"', 5)
    assert hits and "user 0" in hits[0][0].content
    store.close()
    assert len(store._pool) == 0


def test_cache_size_pragma_is_applied(tmp_path):
    store = MemoryStore(str(tmp_path), cache_kb=128)
    _add(store, "u", "hello")
    with store.connect("u") as conn:
        assert conn.execute("PRAGMA cache_size").fetchone()[0] == -128
    store.close()


def test_idle_connections_are_closed(tmp_path):
    store = MemoryStore(str(tmp_path), idle_seconds=0)
    _add(store, "a", "alpha")
    _add(store, "b", "beta")
    time.sleep(0.01)
    with store._pool_lock:
        store._evict_idle_locked()
    assert len(store._pool) == 0
    assert store.count("a") == 1 and store.count("b") == 1
    store.close()
