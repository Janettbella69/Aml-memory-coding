"""Optional evidence selector (v0.6): one small gpt-4o-mini call per Search that looks at the question and the
first retrieved blocks and says which blocks hold the evidence. Those blocks are moved to the front of the
response (and, for plain fact questions, the long tail may be cut). It is a reranker in the competition's
sense: it never writes a word of the response, only chooses the order among verbatim blocks; any failure
(timeout, bad JSON, HTTP error, nonsense indices) leaves the deterministic order untouched.

Why: the locked answer model reads our whole list. On the organisers' public LoCoMo-Refined data the gold
evidence is inside our top-100 for 99 % of questions, yet gpt-4o-mini answers only ~80 % of single-hop
questions from the full list and ~95 % when shown the evidence blocks alone (window-arms experiment,
2026-10-09). Precision at the head of the list, not recall, is what the reader needs.

Enable with SELECTOR_MODEL=gpt-4o-mini and OPENAI_API_KEY (OPENAI_BASE_URL optional).
"""
from __future__ import annotations

import http.client
import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional, Protocol

from .planner import clip_question

log = logging.getLogger("aml_memory.selector")

KINDS = ("fact", "task", "other")

SYSTEM_PROMPT = (
    "You help a memory-retrieval system order the memory blocks it will show to a reader. You never answer the "
    "question yourself. You receive the question (with answer options if any) and numbered memory blocks; every "
    "line of a block is a conversation turn prefixed by a UTC timestamp and the speaker role.\n"
    "1. Pick the blocks that contain evidence needed to answer: the block that states the fact asked and, when the "
    "question needs several pieces (a change over time, a list, a comparison, a cause and its effect), every block "
    "holding one of them. When an earlier statement was later corrected or updated, pick the latest one too. "
    "List the block numbers in order of importance, at most {max_select}. If no block holds evidence, return [].\n"
    "2. Classify the question. 'fact': it asks for something that was said (a fact, date, preference, event, name, "
    "number, opinion, who said what). 'task': it asks to apply rules, follow a procedure or workflow, carry out steps, "
    "compute from data, or obey a format or instruction that the memories define. 'other': anything else.\n"
    "Reply with JSON only: {{\"blocks\": [3, 1], \"kind\": \"fact\"}}"
)


@dataclass
class Selection:
    blocks: list[int] = field(default_factory=list)   # 0-based indices into the candidate list, in importance order
    kind: str = "other"


class Selector(Protocol):
    def select(self, query: str, options: list[str] | None, blocks: list[str]) -> Optional[Selection]: ...


def parse_selection(text: str, n_blocks: int, max_select: int) -> Selection:
    """Tolerant parse of the model's JSON (code fences, stray prose, 1-based numbers as strings)."""
    start, end = text.find("{"), text.rfind("}")
    data = json.loads(text[start:end + 1]) if start >= 0 and end > start else {}
    picked: list[int] = []
    for raw in data.get("blocks") or []:
        try:
            idx = int(raw) - 1          # the prompt numbers blocks from 1
        except (TypeError, ValueError):
            continue
        if 0 <= idx < n_blocks and idx not in picked:
            picked.append(idx)
    kind = str(data.get("kind") or "other").strip().lower()
    return Selection(blocks=picked[:max_select], kind=kind if kind in KINDS else "other")


class OpenAISelector:
    """Chat-completions call (OpenAI-compatible); the open-source division mandates gpt-4o-mini."""

    def __init__(self, model: str, api_key: str, base_url: str = "https://api.openai.com/v1",
                 timeout: float = 20.0, max_select: int = 8, block_chars: int = 1200) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_select = max_select
        self.block_chars = block_chars

    def select(self, query: str, options: list[str] | None, blocks: list[str]) -> Optional[Selection]:
        if not blocks:
            return None
        shown = "\n\n".join(f"[block {i}]\n{b[:self.block_chars]}" for i, b in enumerate(blocks, 1))
        question = clip_question(query + ("\nOptions:\n" + "\n".join(options) if options else ""))
        body = {
            "model": self.model, "temperature": 0, "max_tokens": 120,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT.format(max_select=self.max_select)},
                {"role": "user", "content": f"Question:\n{question}\n\nMemory blocks:\n{shown}"},
            ],
        }
        req = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                out = json.load(resp)
            return parse_selection(out["choices"][0]["message"]["content"], len(blocks), self.max_select)
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError, OSError,
                http.client.HTTPException) as exc:
            log.warning("selector call failed (%s); deterministic order kept", type(exc).__name__)  # no query text in logs
            return None


def selector_from_env(model: str, timeout: float, max_select: int, block_chars: int) -> Optional[Selector]:
    if not model:
        return None
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        log.warning("SELECTOR_MODEL=%s set but OPENAI_API_KEY is empty; selector disabled", model)
        return None
    return OpenAISelector(model, key, os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                          timeout=timeout, max_select=max_select, block_chars=block_chars)
