"""Add-time session summary (coding-0.2.0, SUMMARY_MODEL): one gpt-4o-mini call per Add request turns the chunk's turns
into a short labelled note - task, root cause, files changed, fix, verification, pitfalls - that is stored as ONE extra
turn of the session (role `summary`, clearly labelled as written at Add time) NEXT TO the verbatim turns, never instead
of them. Search returns it inside the session's card, after the task statement. Why: SWEContextBench (arXiv:2602.08316
§3.3) measured that a coding agent gains from the correct session's concise summary (26 -> 34 %) and almost nothing from
its raw trajectory (26 -> 27 %); the cycle-1 textual lesson (docs/CYCLE1_NOTES.md) is that derived text may be added
behind verbatim text but must never replace it. The note is cached by the digest's hash (SUMMARY_CACHE), so a history
written again under another user_id costs nothing. Any failure (timeout, HTTP error, empty answer) stores no summary.
"""
from __future__ import annotations

import hashlib
import sqlite3
import threading
from typing import Optional, Protocol

from .llm import chat, credentials
from .models import MemoryMessage

LABEL = "[session summary, written at Add time by gpt-4o-mini]"
SYSTEM_PROMPT = (
    "You write the memory note of ONE past engineering session - an AI coding agent's transcript with the task, tool "
    "calls, tool output, reasoning and edits - for an engineer who will later work on a similar task in the same "
    "repository. Use only what the transcript shows; copy file paths, function and class names, commands and error "
    "messages exactly; never invent or generalise. Plain text, at most 1200 characters, exactly these labelled lines:\n"
    "TASK: what was asked (one sentence)\n"
    "ROOT CAUSE: what was actually wrong, or the design decision taken (one or two sentences)\n"
    "FILES CHANGED: comma-separated paths of the files the agent edited or created (from its edit/write calls or "
    "diff), or \"none\"\n"
    "FIX: what the change was (one or two sentences, naming the functions or classes)\n"
    "VERIFIED BY: the tests or commands that showed it worked, or \"not verified\"\n"
    "PITFALLS: wrong turns or constraints discovered on the way, or \"none\""
)
MAX_SUMMARY_CHARS = 2000
OPENING_CHARS, ASSISTANT_CHARS, TOOL_CHARS, CLOSING_CHARS = 3000, 700, 200, 4000


def digest(messages: list[MemoryMessage], limit: int) -> str:
    """The transcript as the model sees it: the opening turn, every later turn clipped (assistant text keeps more than
    tool output), the last turn in full; when that is over `limit` characters, middle turns are dropped first."""
    turns = [(m.role, m.text().strip()) for m in messages]
    turns = [(r, t) for r, t in turns if t]
    if not turns:
        return ""
    head = f"[{turns[0][0]}] {turns[0][1][:OPENING_CHARS]}"
    tail = f"[{turns[-1][0]}] {turns[-1][1][:CLOSING_CHARS]}" if len(turns) > 1 else ""
    body = [f"[{r}] {t[:ASSISTANT_CHARS if r == 'assistant' else TOOL_CHARS]}" for r, t in turns[1:-1]]
    budget = limit - len(head) - len(tail) - 2
    while body and sum(len(x) + 1 for x in body) > budget:
        body.pop(len(body) // 2)
    return "\n".join(x for x in [head, *body, tail] if x)


class Summarizer(Protocol):
    def summarize(self, messages: list[MemoryMessage]) -> Optional[str]: ...


class LLMSummarizer:
    def __init__(self, model: str, api_key: str, base_url: str, timeout: float = 60.0, input_chars: int = 30_000,
                 cache_path: Optional[str] = None) -> None:
        self.model, self.api_key, self.base_url = model, api_key, base_url
        self.timeout, self.input_chars = timeout, input_chars
        self.failures = 0
        self._cache: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        if cache_path:
            self._cache = sqlite3.connect(cache_path, check_same_thread=False)
            self._cache.execute("CREATE TABLE IF NOT EXISTS note (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
            self._cache.commit()

    def summarize(self, messages: list[MemoryMessage]) -> Optional[str]:
        text = digest(messages, self.input_chars)
        if not text:
            return None
        key = hashlib.sha256((self.model + "\x1f" + text).encode("utf-8")).hexdigest()
        if self._cache is not None:
            with self._lock:
                row = self._cache.execute("SELECT v FROM note WHERE k = ?", (key,)).fetchone()
            if row is not None:
                return row[0]
        answer = chat(self.model, self.api_key, self.base_url, SYSTEM_PROMPT, "TRANSCRIPT:\n" + text,
                      timeout=self.timeout, max_tokens=450)
        if not answer or not answer.strip():
            self.failures += 1
            return None
        note = LABEL + "\n" + answer.strip()[:MAX_SUMMARY_CHARS]
        if self._cache is not None:
            with self._lock:
                self._cache.execute("INSERT OR IGNORE INTO note(k, v) VALUES (?, ?)", (key, note))
                self._cache.commit()
        return note


def summarizer_from_env(model: str, timeout: float, input_chars: int, cache_path: str = "") -> Optional[LLMSummarizer]:
    if not model:
        return None
    key, base = credentials()
    if not key:
        return None
    return LLMSummarizer(model, key, base, timeout=timeout, input_chars=input_chars, cache_path=cache_path or None)
