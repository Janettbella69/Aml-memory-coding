"""The local eval set must stay well-formed: unique keys, resolvable evidence, adapter-sized chunks."""
from evalset.samples import ALL
from evalset.schema import CAPABILITIES, CHUNK_MESSAGES, CHUNK_WORDS, WORD_RE


def test_samples_validate_and_cover_every_capability():
    seen = set()
    for s in ALL:
        s.validate()
        for q in s.questions:
            seen.add(q.top_capability)
    assert seen == set(CAPABILITIES)


def test_chunks_follow_adapter_limits_and_ids():
    for s in ALL:
        payloads = s.add_payloads(run="t")
        assert payloads, s.name
        for i, p in enumerate(payloads):
            assert p["request_id"] == f"t:{s.name}:chunk-{i}"
            assert p["user_id"] == s.user_id and p["session_id"] == s.session_id
            assert 1 <= len(p["messages"]) <= CHUNK_MESSAGES
            assert sum(len(WORD_RE.findall(m["content"])) for m in p["messages"]) <= CHUNK_WORDS
            for m in p["messages"]:
                assert m["role"] in ("user", "assistant") and m["content"].strip()
                if s.sessions[0].start is None:
                    assert "timestamp" not in m


def test_dialogue_style_prefixes_speaker_names():
    lt = next(s for s in ALL if s.style == "dialogue")
    texts = lt.texts_by_key()
    assert texts["biscuit-dog"].startswith("Tao: ")
    assert texts["lin-job"].startswith("Lin: ")
