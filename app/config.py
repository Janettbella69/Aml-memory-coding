from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, TypeVar

T = TypeVar("T", int, float)

PLACEHOLDER_PREFIX = "change-me"



def _choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
    raw = os.environ.get(name, "").strip().lower()
    return raw if raw in allowed else default

def _bounded(name: str, parse: Callable[[str], T], default: T, lo: T, hi: T) -> T:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = parse(raw.strip())
    except ValueError:
        raise RuntimeError(f"invalid {name}={raw!r}: expected a number in [{lo}, {hi}]") from None
    if value < lo or value > hi:
        raise RuntimeError(f"invalid {name}={raw!r}: expected a number in [{lo}, {hi}]")
    return value


@dataclass(frozen=True)
class Settings:
    api_key: str = ""                  # empty = auth disabled (local use only)
    data_dir: str = "data"             # one SQLite file per user under <data_dir>/users/
    retention_days: int = 30           # evaluation data must be deleted within 30 days of a run
    context_window: int = 2            # turns before/after each hit (sweep 10-07: 2 beats 1 on every metric)
    rule_slots: int = 6                # standing rules offered per search (also capped at top_k // 4)
    lead_hits: int = 5                 # BM25 hits placed before the rule slots
    recency_band: float = 0.2          # hits whose normalised score differs by less than this are ordered newest-first
    max_response_chars: int = 400_000  # ~100k tokens; the platform keeps a 117,760-token prefix in rank order
    fallback_results: int = 10         # newest memories returned when nothing matches
    sweep_hours: int = 6               # how often the retention sweep runs while the service is up
    search_parallelism: int = 1        # SQLite searches executed at once per process; soak 10-07: 1 -> 293 q/s, 16 -> 24 q/s (mutex contention)
    write_parallelism: int = 2         # Add transactions executed at once per process; soak 10-07: 2 -> 1,313 adds/s, 16 -> 627 adds/s
    db_pool_size: int = 128            # cached SQLite connections per process (3 file handles each)
    db_cache_kb: int = 256             # SQLite page cache per connection; bounds memory at ~pool x cache per process
    db_idle_seconds: int = 120         # cached connections unused this long are closed
    planner_model: str = ""           # "" = deterministic only; "gpt-4o-mini" enables the one-step query planner
    planner_max_queries: int = 3
    planner_timeout: float = 20.0
    planner_snippets: int = 8          # first-pass results shown to the planner
    time_notes: bool = True            # v0.4: render "(dates: yesterday = 2025-04-26)" after turns with relative-time phrases
    options_mode: str = "append"       # v0.4: how answer options enter retrieval: append (query+options), none, channel (own RRF channel)
    options_weight: float = 0.6        # v0.4: RRF weight of the options channel when options_mode=channel
    embed_model: str = ""              # v0.5: "text-embedding-v4" enables the dense channel ("" = lexical only)
    embed_dims: int = 256              # v0.5: vector size (text-embedding-v4 accepts 64..2048); float16 on disk
    embed_timeout: float = 30.0
    embed_rps: float = 14.0            # v0.5: client-side pacing PER WORKER PROCESS (the service sends up to UVICORN_WORKERS × this); published key limits differ by source (1,000–1,800 req/min)
    dense_weight: float = 1.0          # v0.5: RRF weight of the dense list against the lexical lists
    dense_head: int = 5                # v0.5: keep the first N lexical hits in front, fuse the dense list only behind them (0 = plain RRF; 5 chosen on public LoCoMo-Refined)
    vector_cache_users: int = 64       # v0.5: per-process LRU of in-memory vector matrices
    selector_model: str = ""          # v0.6: "gpt-4o-mini" enables the evidence selector (a reranker over our own blocks); "" = off
    selector_candidates: int = 40      # v0.6: how many leading blocks the selector sees
    selector_max_select: int = 8       # v0.6: blocks it may move to the front
    selector_tail: int = -1            # v0.6: for 'fact' questions, unselected blocks kept after the selected ones; -1 = keep all (reorder only)
    selector_block_chars: int = 1200   # v0.6: characters of each block shown to the selector
    selector_timeout: float = 20.0
    selector_min_results: int = 12     # v0.6: shorter result lists are returned as they are
    rerank_provider: str = "dashscope" # v0.7: dashscope | voyage | cohere | jina (hosted rerank APIs; see app/rerank.py)
    rerank_model: str = ""            # v0.7: "gte-rerank-v2" | "qwen3-rerank" enables the cross-encoder reranker (rules: rerankers unrestricted); "" = off
    rerank_candidates: int = 40        # v0.7: leading blocks that are reranked; the rest keep their order behind them
    rerank_mode: str = "replace"       # v0.7: replace (reranker order for the head) | rrf (fuse reranker rank with the retrieval rank, k = 60)
    rerank_timeout: float = 20.0
    rerank_doc_chars: int = 1500       # v0.7: characters of each document sent to the reranker (4,000-token limit per document)
    rerank_doc: str = "both"           # v0.7 (default since 2026-10-11, see METHOD_CARD): hit = score the matched turn itself; block = the rendered span (its first chars may be only the preceding turns — on long-turn data that hid the hit and scrambled the order); both = the matched turn first, then the span (the hit is always inside the reranker's window and short turns keep their context)
    rerank_min_results: int = 6        # v0.7: shorter lists are returned as they are
    fit_chars: int = 0                 # v0.8 fit mode: when the user's whole stored text is at most this many characters, return ALL of it in
                                       # relevance order (contiguous segments, top hits centred) instead of hit ± window blocks. 0 = off
    fit_fill: float = 0.8              # v0.8: share of the top_k result slots the segmentation aims to use
    fit_span_min: int = 800            # v0.8: smallest segment size in rendered characters
    fit_anchors: int = 8               # v0.8: leading hits that get a segment centred on them
    turn_marks: bool = False           # v0.8: render `[turn N]` on turns that came without a timestamp (N = place in the conversation)
    return_chars: int = 0              # v0.9: final cut AFTER the reranker/selector: whole blocks while the running total is below this many
                                       # characters, standing rules behind the cut kept. MAX_RESPONSE_CHARS still bounds assembly (what the
                                       # reranker chooses from). 0 = off
    return_from_chars: int = 0         # v0.9: apply RETURN_CHARS only to users whose stored text exceeds this many characters; smaller
                                       # memories go out whole. 0 = every user
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        key = os.environ.get("MEMORY_API_KEY", "").strip()
        if key and (len(key) < 16 or key.startswith(PLACEHOLDER_PREFIX)):
            raise RuntimeError("MEMORY_API_KEY must be a real secret of at least 16 characters (or empty to disable auth)")
        return cls(
            api_key=key,
            data_dir=os.environ.get("DATA_DIR", "data"),
            retention_days=_bounded("RETENTION_DAYS", int, 30, 1, 30),
            context_window=_bounded("CONTEXT_WINDOW", int, 2, 0, 5),
            rule_slots=_bounded("RULE_SLOTS", int, 6, 0, 50),
            lead_hits=_bounded("LEAD_HITS", int, 5, 0, 20),
            recency_band=_bounded("RECENCY_BAND", float, 0.2, 0.0, 1.0),
            max_response_chars=_bounded("MAX_RESPONSE_CHARS", int, 400_000, 10_000, 2_000_000),
            fallback_results=_bounded("FALLBACK_RESULTS", int, 10, 0, 100),
            sweep_hours=_bounded("SWEEP_HOURS", int, 6, 1, 168),
            search_parallelism=_bounded("SEARCH_PARALLELISM", int, 1, 1, 64),
            write_parallelism=_bounded("WRITE_PARALLELISM", int, 2, 1, 64),
            db_pool_size=_bounded("DB_POOL_SIZE", int, 128, 4, 4096),
            db_cache_kb=_bounded("DB_CACHE_KB", int, 256, 64, 65536),
            db_idle_seconds=_bounded("DB_IDLE_SECONDS", int, 120, 5, 86400),
            planner_model=os.environ.get("PLANNER_MODEL", "").strip(),
            planner_max_queries=_bounded("PLANNER_MAX_QUERIES", int, 3, 1, 6),
            planner_timeout=_bounded("PLANNER_TIMEOUT", float, 20.0, 1.0, 120.0),
            planner_snippets=_bounded("PLANNER_SNIPPETS", int, 8, 0, 30),
            time_notes=os.environ.get("TIME_NOTES", "1").strip().lower() not in ("0", "false", "no", "off", ""),
            options_mode=_choice("OPTIONS_MODE", "append", ("append", "none", "channel")),
            options_weight=_bounded("OPTIONS_WEIGHT", float, 0.6, 0.0, 2.0),
            embed_model=os.environ.get("EMBED_MODEL", "").strip(),
            embed_dims=_bounded("EMBED_DIMS", int, 256, 64, 2048),
            embed_timeout=_bounded("EMBED_TIMEOUT", float, 30.0, 1.0, 300.0),
            embed_rps=_bounded("EMBED_RPS", float, 14.0, 0.1, 100.0),
            dense_weight=_bounded("DENSE_WEIGHT", float, 1.0, 0.0, 3.0),
            dense_head=_bounded("DENSE_HEAD", int, 5, 0, 50),
            vector_cache_users=_bounded("VECTOR_CACHE_USERS", int, 64, 1, 4096),
            selector_model=os.environ.get("SELECTOR_MODEL", "").strip(),
            selector_candidates=_bounded("SELECTOR_CANDIDATES", int, 40, 5, 100),
            selector_max_select=_bounded("SELECTOR_MAX_SELECT", int, 8, 1, 20),
            selector_tail=_bounded("SELECTOR_TAIL", int, -1, -1, 1000),
            selector_block_chars=_bounded("SELECTOR_BLOCK_CHARS", int, 1200, 200, 6000),
            selector_timeout=_bounded("SELECTOR_TIMEOUT", float, 20.0, 1.0, 120.0),
            selector_min_results=_bounded("SELECTOR_MIN_RESULTS", int, 12, 2, 1000),
            rerank_provider=_choice("RERANK_PROVIDER", "dashscope", ("dashscope", "voyage", "cohere", "jina")),
            rerank_model=os.environ.get("RERANK_MODEL", "").strip(),
            rerank_candidates=_bounded("RERANK_CANDIDATES", int, 40, 2, 500),
            rerank_mode=_choice("RERANK_MODE", "replace", ("replace", "rrf")),
            rerank_timeout=_bounded("RERANK_TIMEOUT", float, 20.0, 1.0, 120.0),
            rerank_doc_chars=_bounded("RERANK_DOC_CHARS", int, 1500, 200, 12000),
            rerank_doc=_choice("RERANK_DOC", "both", ("hit", "block", "both")),
            rerank_min_results=_bounded("RERANK_MIN_RESULTS", int, 6, 2, 1000),
            fit_chars=_bounded("FIT_CHARS", int, 0, 0, 2_000_000),
            fit_fill=_bounded("FIT_FILL", float, 0.8, 0.1, 1.0),
            fit_span_min=_bounded("FIT_SPAN_MIN", int, 800, 50, 100_000),
            fit_anchors=_bounded("FIT_ANCHORS", int, 8, 0, 100),
            turn_marks=os.environ.get("TURN_MARKS", "0").strip().lower() in ("1", "true", "yes", "on", "untimed"),
            return_chars=_bounded("RETURN_CHARS", int, 0, 0, 2_000_000),
            return_from_chars=_bounded("RETURN_FROM_CHARS", int, 0, 0, 1_000_000_000),
            log_level=os.environ.get("LOG_LEVEL", "INFO").strip().upper() or "INFO",
        )
