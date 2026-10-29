"""v1.1 coding-track behaviours (CODE_TERMS, SESSION_CARDS); each maps to a line in docs/METHOD_CARD.md."""
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from app.codeterms import extra_words, index_text, split_identifier
from app.config import Settings
from app.main import create_app
from app.store import MemoryStore


def make_client(**overrides):
    tmp = tempfile.TemporaryDirectory()
    settings = Settings(api_key="", data_dir=os.path.join(tmp.name, "d"), **overrides)
    app = create_app(settings=settings, store=MemoryStore(settings.data_dir, code_terms=settings.code_terms))
    client = TestClient(app)
    client.__enter__()
    client._tmp = tmp  # keep the directory alive
    return client


def add(c, rid, user, msgs, session="s"):
    r = c.post("/v1/memories/add", json={"request_id": rid, "user_id": user, "session_id": session, "messages": msgs})
    assert r.status_code == 200, r.text


def search(c, user, query, top_k=100):
    r = c.post("/v1/memories/search", json={"query": query, "user_id": user, "top_k": top_k})
    assert r.status_code == 200, r.text
    return r.json()["data"]


def test_split_identifier():
    assert split_identifier("getUserName") == ["get", "user", "name"]
    assert split_identifier("HTTPResponse2") == ["http", "response", "2"]
    assert split_identifier("utf8") == ["utf", "8"]
    assert split_identifier("ndarithmetic") == []
    assert split_identifier("get_user_name") == []
    assert split_identifier("ABC") == []


def test_index_text_only_changes_turns_with_inner_words():
    assert index_text("plain words only") == "plain words only"
    assert index_text("call getUserName()") == "call getUserName()\nget user name"
    assert extra_words("x" * 50) == []
    assert len(extra_words(" ".join(f"a{i}B" for i in range(5000)))) == 2000   # bounded per turn


CAMEL = [{"role": "assistant", "content": "Renamed AuthService.getUserName to read the cached profile."},
         {"role": "assistant", "content": "Unrelated: bumped the version number in setup.cfg."}]


@pytest.mark.parametrize("code_terms", [True, False])
def test_code_terms_find_a_camel_case_name_by_its_words(code_terms):
    c = make_client(code_terms=code_terms)
    add(c, "c:chunk-0", "u", CAMEL)
    data = search(c, "u", "user name lookup")
    assert data
    if code_terms:
        assert "getUserName" in data[0]["content"]
    else:   # no term matches: the newest-turns fallback, newest first
        assert "getUserName" not in data[0]["content"]
    c.__exit__(None, None, None)


TASK_A = "Fix this bug: slugify keeps leading and trailing dashes."
TASK_B = "Fix this bug: the template tag renders dates in the wrong timezone."
SESSION_A = [{"role": "user", "content": TASK_A},
             {"role": "user", "content": "[tool result] def slugify(value): ... return re.sub(r'[-\\s]+', '-', value)"},
             {"role": "assistant", "content": "I changed slugify in django/utils/text.py so the dashes are stripped."}]
SESSION_B = [{"role": "user", "content": TASK_B},
             {"role": "user", "content": "[tool result] " + "slugify dashes " * 20},
             {"role": "assistant", "content": "The timezone handling lives in django/utils/timezone.py."}]


def test_session_cards_open_each_session_with_its_task_and_keep_sessions_together():
    c = make_client(session_cards=True, rule_slots=0, recency_band=0.0, context_window=0)
    add(c, "a:chunk-0", "u", SESSION_A, session="A")
    add(c, "b:chunk-0", "u", SESSION_B, session="B")
    data = search(c, "u", "slugify strip dashes")
    contents = [r["content"] for r in data]
    assert TASK_A in contents[0]                                  # the paired session's task opens the list
    assert any(TASK_B in t for t in contents)                     # B is a hit too, so its task is shown even though it matches nothing
    a_blocks = [i for i, t in enumerate(contents) if TASK_A in t or "django/utils/text.py" in t or "def slugify" in t]
    b_blocks = [i for i, t in enumerate(contents) if TASK_B in t or "timezone" in t or "slugify dashes slugify" in t]
    assert a_blocks and b_blocks and max(a_blocks) < min(b_blocks)   # one card after the other
    assert b_blocks[0] == min(i for i, t in enumerate(contents) if TASK_B in t)   # each card starts with its task
    scores = [r["score"] for r in data]
    assert scores == sorted(scores, reverse=True) and len(set(scores)) == len(scores)
    c.__exit__(None, None, None)


def test_session_cards_off_returns_hits_only():
    c = make_client(rule_slots=0, recency_band=0.0, context_window=0)
    add(c, "a:chunk-0", "u", SESSION_A, session="A")
    add(c, "b:chunk-0", "u", SESSION_B, session="B")
    data = search(c, "u", "slugify strip dashes")
    assert data and not any(TASK_B in r["content"] for r in data)   # B's task shares no word with the query: not returned
    c.__exit__(None, None, None)


def test_session_cards_card_limits():
    c = make_client(session_cards=True, rule_slots=0, recency_band=0.0, context_window=0, card_sessions=1, card_hits=1)
    add(c, "a:chunk-0", "u", SESSION_A, session="A")
    add(c, "b:chunk-0", "u", SESSION_B, session="B")
    data = search(c, "u", "slugify strip dashes")
    assert len(data) == 2 and TASK_A in data[0]["content"]        # one card: its task + one hit
    c.__exit__(None, None, None)
