"""v0.8 fit mode: when a user's whole memory fits into the answer window, Search returns all of it — at most top_k
contiguous segments, most relevant first, every turn exactly once. Off by default (FIT_CHARS=0)."""
import os
import tempfile

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.store import MemoryStore


def make_client(**overrides):
    tmp = tempfile.TemporaryDirectory()
    settings = Settings(api_key="", data_dir=os.path.join(tmp.name, "d"), **overrides)
    store = MemoryStore(settings.data_dir)
    client = TestClient(create_app(settings=settings, store=store))
    client.__enter__()
    client._tmp, client._store = tmp, store
    return client


def script(c, turns=600, session="s", user="u", start=0, special=None):
    """A play-like memory: one-line turns, no timestamps, twenty per Add chunk."""
    special = special or {}
    lines = [special.get(i, f"Juror {i % 12}: filler remark number {i} about the weather in the room.") for i in range(turns)]
    for n in range(0, turns, 20):
        r = c.post("/v1/memories/add", json={"request_id": f"eval:r:{session}:chunk-{start + n // 20}", "user_id": user, "session_id": session,
                                             "messages": [{"role": "user", "content": x} for x in lines[n:n + 20]]})
        assert r.status_code == 200, r.text
    return lines


def search(c, query, top_k=100, user="u"):
    return c.post("/v1/memories/search", json={"query": query, "user_id": user, "top_k": top_k}).json()["data"]


def lines_of(data):
    return [ln for d in data for ln in d["content"].split("\n")]


def test_off_by_default_and_counters_track_the_corpus():
    assert Settings().fit_chars == 0
    c = make_client()
    lines = script(c, turns=45)
    assert c._store.corpus_size("u") == (45, sum(len(x) for x in lines))
    c.post("/v1/memories/add", json={"request_id": "eval:r:s:chunk-0", "user_id": "u", "session_id": "s",       # idempotent replay
                                     "messages": [{"role": "user", "content": x} for x in lines[:20]]})
    c.post("/v1/memories/add", json={"request_id": "eval:r:s:chunk-9", "user_id": "u", "session_id": "s",
                                     "messages": [{"role": "user", "content": "   "}, {"role": "user", "content": "Juror 3: one more."}]})
    assert c._store.corpus_size("u") == (46, sum(len(x) for x in lines) + len("Juror 3: one more."))
    assert c._store.corpus_size("nobody") == (0, 0)


def test_fit_returns_the_whole_conversation_once_with_the_hit_first_and_centred():
    special = {311: "Juror 8: the old man could not have reached the door in fifteen seconds.",
               40: "Juror 3: my son hit me in the jaw when he was sixteen."}
    c = make_client(fit_chars=500_000)
    lines = script(c, turns=600, special=special)
    data = search(c, "old man reached the door fifteen seconds")
    got = lines_of(data)
    assert 40 <= len(data) <= 100                                    # 42k characters in segments of at least 800
    assert sorted(got) == sorted("USER: " + x for x in lines)        # everything, exactly once
    first = data[0]["content"].split("\n")
    where = first.index("USER: " + special[311])
    assert 0 < where < len(first) - 1                                # the hit sits inside its segment, context on both sides
    assert first == ["USER: " + x for x in lines[311 - where:311 - where + len(first)]]   # contiguous, in order
    ids = [d["id"] for d in data]
    assert len(set(ids)) == len(ids) and all(ids) and all(d["content"] for d in data)
    scores = [d["score"] for d in data]
    assert scores == sorted(scores, reverse=True) and scores[0] == 1.0
    # classic mode on the same memory shows only a part of it
    c2 = make_client()
    script(c2, turns=600, special=special)
    classic = lines_of(search(c2, "old man reached the door fifteen seconds"))
    assert len(set(classic)) < 0.9 * len(lines)


def test_fit_respects_small_top_k_and_sessions():
    c = make_client(fit_chars=500_000)
    a = script(c, turns=120, session="act-1", start=0, special={7: "Peter: the baths are the town's future."})
    b = script(c, turns=120, session="act-2", start=50, special={90: "Petra: I will not translate that story."})
    for k in (1, 3, 10):
        data = search(c, "Petra translate story", top_k=k)
        assert len(data) <= k
        assert sorted(lines_of(data)) == sorted("USER: " + x for x in a + b)
        assert "Petra: I will not translate that story." in data[0]["content"]
    data = search(c, "Petra translate story", top_k=100)
    for d in data:                                                    # with room to spare a segment never straddles two acts
        body = d["content"].split("\n")
        assert all(("USER: " + x) in set("USER: " + y for y in a) for x in [ln[6:] for ln in body]) or \
               all(("USER: " + x) in set("USER: " + y for y in b) for x in [ln[6:] for ln in body])


def test_memory_larger_than_fit_chars_keeps_the_classic_blocks():
    special = {311: "Juror 8: the old man could not have reached the door in fifteen seconds."}
    c_fit, c_off = make_client(fit_chars=5_000), make_client()
    for c in (c_fit, c_off):
        script(c, turns=600, special=special)
    q = "old man reached the door fifteen seconds"
    assert [d["content"] for d in search(c_fit, q)] == [d["content"] for d in search(c_off, q)]


def test_fit_without_any_lexical_hit_still_returns_everything():
    c = make_client(fit_chars=500_000)
    lines = script(c, turns=80)
    data = search(c, "zzzz qqqq")
    assert sorted(lines_of(data)) == sorted("USER: " + x for x in lines) and len(data) <= 100


def test_turn_marks_only_on_untimed_turns_and_off_by_default():
    assert Settings().turn_marks is False
    c = make_client(turn_marks=True, context_window=2)
    c.post("/v1/memories/add", json={"request_id": "eval:r:s:chunk-3", "user_id": "u", "session_id": "s", "messages": [
        {"role": "user", "content": "Juror 8: the knife was bought in a junk shop."},
        {"role": "user", "content": "Juror 3: you can not prove that.", "timestamp": 1704067200000},
        {"role": "user", "content": "Juror 8: here is one exactly like it."}]})
    body = search(c, "knife junk shop exactly like")[0]["content"].split("\n")
    assert body[0] == "[turn 3001] USER: Juror 8: the knife was bought in a junk shop."
    assert body[1].startswith("[2024-01-01 00:00 UTC] USER: Juror 3:") and "[turn" not in body[1]
    assert body[2] == "[turn 3003] USER: Juror 8: here is one exactly like it."
    plain = make_client(context_window=1)
    plain.post("/v1/memories/add", json={"request_id": "eval:r:s:chunk-3", "user_id": "u", "session_id": "s", "messages": [
        {"role": "user", "content": "Juror 8: the knife was bought in a junk shop."}]})
    assert search(plain, "knife junk shop")[0]["content"] == "USER: Juror 8: the knife was bought in a junk shop."
