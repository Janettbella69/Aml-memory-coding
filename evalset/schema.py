from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, Literal, Optional

CAPABILITIES = (
    "fact_recall",      # single fact / entity-speaker attribution / structured set extraction
    "composition",      # cross-session aggregation / causal chain / evidence validity / common-sense completion
    "temporal",         # dates & intervals / event ordering / stance change over time
    "governance",       # new-value overwrite / contradiction / deletion & forgetting / long-history synthesis
    "personalization",  # preference recall / anti-stereotype robustness / background & care
    "execution",        # domain reasoning / rule-system application / workflow / empirical data / format constraints
    "safety",           # abstention on insufficient evidence / privacy minimization
)

CHUNK_MESSAGES = 20      # official adapter: "splits deterministically at either 20 messages
CHUNK_WORDS = 2000       # or 2,000 Adapter-counted words"
WORD_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class Message:
    key: Optional[str]      # unique within the sample when set; questions cite it as evidence
    speaker: str            # "user"/"assistant" in assistant style, a name in dialogue style
    text: str


@dataclass(frozen=True)
class Session:
    start: Optional[str]    # "YYYY-MM-DD HH:MM" (UTC); None = adapter sends no timestamps
    messages: tuple[Message, ...]
    gap_seconds: int = 40   # spacing between consecutive messages inside the session


def session(start: Optional[str], turns: Iterable[tuple], gap_seconds: int = 40) -> Session:
    msgs = []
    for turn in turns:
        if len(turn) == 2:
            speaker, text = turn
            key = None
        elif len(turn) == 3:
            speaker, text, key = turn
        else:
            raise ValueError(f"turn must be (speaker, text[, key]): {turn!r}")
        msgs.append(Message(key=key, speaker=speaker, text=text.strip()))
    return Session(start=start, messages=tuple(msgs), gap_seconds=gap_seconds)


@dataclass(frozen=True)
class Question:
    qid: str
    capability: str                 # "fact_recall/single_fact" -> capability "fact_recall", sub "single_fact"
    query: str
    evidence: tuple[str, ...]       # message keys; empty tuple = abstention question (no evidence exists)
    mode: Literal["all", "any"] = "any"   # all: every evidence message must be retrieved (multi-hop)
    trap: tuple[str, ...] = ()      # stale/misleading messages that must rank BELOW the evidence
    options: Optional[tuple[str, ...]] = None
    answer: str = ""                # for humans reading the set; never used by the metric
    note: str = ""

    @property
    def top_capability(self) -> str:
        return self.capability.split("/")[0]


def Q(qid: str, capability: str, query: str, evidence: Iterable[str] = (), *, mode: str = "any",
      trap: Iterable[str] = (), options: Optional[Iterable[str]] = None, answer: str = "", note: str = "") -> Question:
    cap = capability.split("/")[0]
    if cap not in CAPABILITIES:
        raise ValueError(f"{qid}: unknown capability {capability!r}")
    return Question(qid=qid, capability=capability, query=query, evidence=tuple(evidence), mode=mode,  # type: ignore[arg-type]
                    trap=tuple(trap), options=tuple(options) if options else None, answer=answer, note=note)


@dataclass(frozen=True)
class Sample:
    name: str
    style: Literal["assistant", "dialogue"]
    sessions: tuple[Session, ...]
    questions: tuple[Question, ...]
    speakers: tuple[str, str] = ("user", "assistant")   # dialogue style: first -> role user, second -> role assistant
    description: str = ""

    # ------------------------------------------------------------ derived
    @property
    def user_id(self) -> str:
        return f"local:{self.name}"

    @property
    def session_id(self) -> str:
        return f"local:{self.name}:sample:0"

    def validate(self) -> None:
        keys: dict[str, int] = {}
        for session_ in self.sessions:
            for m in session_.messages:
                if not m.text:
                    raise ValueError(f"{self.name}: empty message")
                if self.style == "assistant" and m.speaker not in ("user", "assistant"):
                    raise ValueError(f"{self.name}: speaker must be user/assistant, got {m.speaker!r}")
                if self.style == "dialogue" and m.speaker not in self.speakers:
                    raise ValueError(f"{self.name}: unknown speaker {m.speaker!r}")
                if m.key:
                    keys[m.key] = keys.get(m.key, 0) + 1
        dup = [k for k, n in keys.items() if n > 1]
        if dup:
            raise ValueError(f"{self.name}: duplicate message keys {dup}")
        qids: set[str] = set()
        for q in self.questions:
            if q.qid in qids:
                raise ValueError(f"{self.name}: duplicate qid {q.qid}")
            qids.add(q.qid)
            for k in (*q.evidence, *q.trap):
                if k not in keys:
                    raise ValueError(f"{self.name}/{q.qid}: unknown message key {k!r}")

    def texts_by_key(self) -> dict[str, str]:
        """Message text exactly as the adapter would send it (speaker prefix included in dialogue style)."""
        out: dict[str, str] = {}
        for session_ in self.sessions:
            for m in session_.messages:
                if m.key:
                    out[m.key] = self.render_text(m)
        return out

    def all_texts(self) -> list[str]:
        return [self.render_text(m) for s in self.sessions for m in s.messages]

    def render_text(self, m: Message) -> str:
        if self.style == "dialogue":
            return f"{m.speaker}: {m.text}"
        return m.text

    def role_of(self, m: Message) -> str:
        if self.style == "dialogue":
            return "user" if m.speaker == self.speakers[0] else "assistant"
        return m.speaker

    def add_payloads(self, run: str = "local") -> list[dict]:
        """Chunk like the official adapter (per session; 20 messages or 2,000 words) and build Add bodies."""
        payloads: list[dict] = []
        chunk_no = 0
        for session_ in self.sessions:
            base = None
            if session_.start:
                base = datetime.strptime(session_.start, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            buf: list[dict] = []
            words = 0

            def flush() -> None:
                nonlocal buf, words, chunk_no
                if not buf:
                    return
                payloads.append({
                    "request_id": f"{run}:{self.name}:chunk-{chunk_no}",
                    "user_id": self.user_id,
                    "session_id": self.session_id,
                    "messages": buf,
                })
                chunk_no += 1
                buf, words = [], 0

            for i, m in enumerate(session_.messages):
                text = self.render_text(m)
                n = len(WORD_RE.findall(text))
                if buf and (len(buf) >= CHUNK_MESSAGES or words + n > CHUNK_WORDS):
                    flush()
                msg: dict = {"role": self.role_of(m), "content": text}
                if base is not None:
                    msg["timestamp"] = int((base + timedelta(seconds=i * session_.gap_seconds)).timestamp() * 1000)
                buf.append(msg)
                words += n
            flush()
        return payloads
