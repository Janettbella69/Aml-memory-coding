"""Optional query planner: one small gpt-4o-mini call per Search that turns the question plus the first
search results into extra keyword queries (and a date range). It never writes answers; its output only
decides what else to look up. Idea from ReFind (agent-controlled search over raw chat logs, MIT); here
reduced to a single bounded step with a deterministic fallback.

Enable with PLANNER_MODEL=gpt-4o-mini and OPENAI_API_KEY (OPENAI_BASE_URL optional). Any failure
(timeout, bad JSON, HTTP error) silently falls back to the deterministic pipeline.
"""
from __future__ import annotations

import http.client
import json
import logging
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Protocol

log = logging.getLogger("aml_memory.planner")

SYSTEM_PROMPT = (
    "You help a memory-retrieval system find evidence in one user's past conversations. You never answer the "
    "question yourself. You receive the question (with answer options if any) and the first search results, "
    "each prefixed by a UTC timestamp and speaker role.\n"
    "Propose up to {max_queries} short keyword queries (2-6 words, no stop words) that would retrieve evidence "
    "that is still missing. Typical moves: resolve a reference in two hops (the results name the person, pet, "
    "place or project the question talks about -> search for that name together with the attribute asked); "
    "add synonyms or the concrete value words (numbers, units, dates); split a multi-part question; for "
    "questions about change over time, search both the old and the new wording.\n"
    "If the results already contain everything needed, return an empty list. If the question is about a "
    "specific period, give date_from/date_to as YYYY-MM-DD, else null.\n"
    "Reply with JSON only: {{\"queries\": [\"...\"], \"date_from\": null, \"date_to\": null}}"
)
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass
class Plan:
    queries: list[str] = field(default_factory=list)
    date_from_ms: Optional[int] = None
    date_to_ms: Optional[int] = None

    @property
    def has_dates(self) -> bool:
        return self.date_from_ms is not None or self.date_to_ms is not None

    def in_range(self, event_ms: Optional[int]) -> bool:
        if event_ms is None:
            return False
        if self.date_from_ms is not None and event_ms < self.date_from_ms:
            return False
        if self.date_to_ms is not None and event_ms > self.date_to_ms:
            return False
        return True


MAX_QUESTION_CHARS = 6000


def clip_question(text: str, limit: int = MAX_QUESTION_CHARS) -> str:
    """The opening and the closing of an over-long question. A question that carries a reference document can run to
    tens of thousands of tokens; the task itself is stated at one of its ends, and sending all of it to a helper
    model would cost seconds and cents on every Search (and time out)."""
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n[...]\n" + text[-half:]


class Planner(Protocol):
    def plan(self, query: str, options: list[str] | None, snippets: list[str]) -> Optional[Plan]: ...


def parse_plan(text: str, max_queries: int) -> Plan:
    """Tolerant parse of the model's JSON (code fences, stray prose)."""
    start, end = text.find("{"), text.rfind("}")
    data = json.loads(text[start:end + 1]) if start >= 0 and end > start else {}
    queries: list[str] = []
    for q in data.get("queries") or []:
        if isinstance(q, str) and q.strip() and q.strip() not in queries:
            queries.append(q.strip()[:120])
    plan = Plan(queries=queries[:max_queries])
    for key, attr, end_of_day in (("date_from", "date_from_ms", False), ("date_to", "date_to_ms", True)):
        value = data.get(key)
        if isinstance(value, str) and DATE_RE.match(value):
            try:
                dt = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                ms = int(dt.timestamp() * 1000) + (86_399_999 if end_of_day else 0)
                setattr(plan, attr, ms)
            except ValueError:
                pass
    return plan


class OpenAIPlanner:
    """Chat-completions call (OpenAI-compatible); the open-source division mandates gpt-4o-mini."""

    def __init__(self, model: str, api_key: str, base_url: str = "https://api.openai.com/v1",
                 timeout: float = 20.0, max_queries: int = 3, max_snippet_chars: int = 240) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_queries = max_queries
        self.max_snippet_chars = max_snippet_chars

    def plan(self, query: str, options: list[str] | None, snippets: list[str]) -> Optional[Plan]:
        shown = "\n".join(f"- {s[:self.max_snippet_chars]}" for s in snippets) or "- (no results)"
        question = clip_question(query + ("\nOptions:\n" + "\n".join(options) if options else ""))
        body = {
            "model": self.model, "temperature": 0, "max_tokens": 200,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT.format(max_queries=self.max_queries)},
                {"role": "user", "content": f"Question:\n{question}\n\nFirst results:\n{shown}"},
            ],
        }
        req = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                out = json.load(resp)
            return parse_plan(out["choices"][0]["message"]["content"], self.max_queries)
        # OSError/HTTPException: a TLS alert, reset or short read while the response is read is not wrapped in URLError
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException, KeyError, ValueError,
                json.JSONDecodeError) as exc:
            log.warning("planner call failed (%s); deterministic fallback", type(exc).__name__)  # no query text in logs
            return None


def planner_from_env(model: str, max_queries: int, timeout: float) -> Optional[Planner]:
    if not model:
        return None
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        log.warning("PLANNER_MODEL=%s set but OPENAI_API_KEY is empty; planner disabled", model)
        return None
    return OpenAIPlanner(model, key, os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                         timeout=timeout, max_queries=max_queries)
