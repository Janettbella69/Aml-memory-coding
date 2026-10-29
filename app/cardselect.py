"""Search-time card selector (coding-0.2.0, CARD_SELECT_MODEL): after the deterministic ranking has built the session cards,
one gpt-4o-mini call reads the new task and a brief of each leading card (the task as it was given, the Add-time summary
if any, the agent's closing message) and returns the numbers of the sessions that are useful, most useful first. Those
cards move to the front in that order; the rest keep their order behind them. It changes only the order of verbatim
cards - it adds no text and never sees more than the briefs. The call runs outside the SQLite gate; any failure or an
empty answer keeps the deterministic order. Idea: the evidence-selector step of the textual entry (0.6) at the level of
sessions, and ReFind's agent-controlled search reduced to one bounded decision."""
from __future__ import annotations

import json
from typing import Optional, Protocol

from .llm import chat, credentials

SYSTEM_PROMPT = (
    "You rank past engineering sessions from the same repository by how useful they are for a NEW task. Useful means: "
    "the session touched the same files, functions or classes; it fixed the same kind of bug or built the same kind of "
    "feature; or it explains the mechanism the new task is about. Reply with JSON only: "
    "{\"ranked\": [session numbers, most useful first]}. List only useful sessions; {\"ranked\": []} if none is useful."
)
QUERY_CHARS = 4000


class CardSelector(Protocol):
    def rank(self, query: str, briefs: list[str]) -> Optional[list[int]]: ...


def parse_ranked(text: str, n: int) -> list[int]:
    """0-based indices of the listed sessions, valid and distinct, in the model's order."""
    start, end = text.find("{"), text.rfind("}")
    try:
        data = json.loads(text[start:end + 1]) if start >= 0 and end > start else {}
    except json.JSONDecodeError:
        return []
    out: list[int] = []
    for v in data.get("ranked") or []:
        try:
            i = int(v) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= i < n and i not in out:
            out.append(i)
    return out


class LLMCardSelector:
    def __init__(self, model: str, api_key: str, base_url: str, timeout: float = 30.0) -> None:
        self.model, self.api_key, self.base_url, self.timeout = model, api_key, base_url, timeout
        self.failures = 0

    def rank(self, query: str, briefs: list[str]) -> Optional[list[int]]:
        if len(briefs) < 2:
            return None
        q = query if len(query) <= QUERY_CHARS else query[:QUERY_CHARS // 2] + "\n[...]\n" + query[-QUERY_CHARS // 2:]
        shown = "\n\n".join(f"[{i + 1}] {b}" for i, b in enumerate(briefs))
        answer = chat(self.model, self.api_key, self.base_url, SYSTEM_PROMPT,
                      f"NEW TASK:\n{q}\n\nPAST SESSIONS:\n{shown}", timeout=self.timeout, max_tokens=120, json_object=True)
        if answer is None:
            self.failures += 1
            return None
        return parse_ranked(answer, len(briefs))


def card_selector_from_env(model: str, timeout: float) -> Optional[LLMCardSelector]:
    if not model:
        return None
    key, base = credentials()
    if not key:
        return None
    return LLMCardSelector(model, key, base, timeout=timeout)
