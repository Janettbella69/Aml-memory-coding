"""Search = evidence assembly. No generation anywhere: every returned string is conversation text
rendered with its timestamp and role.

Pipeline (v0.2):
  1. word index  : porter-stemmed BM25 over the query + answer options (Latin-script terms)
  2. trigram index: BM25 over CJK character trigrams, fused with (1) by reciprocal rank fusion
  3. recency band: hits whose score is within RECENCY_BAND of each other are ordered newest-first,
                   so a value that was later overwritten does not outrank its replacement
  4. standing rules: the user's tagged instructions/preferences, query-matching ones first, then newest,
                   placed after the first LEAD_HITS results
  5. spans      : each returned hit carries CONTEXT_WINDOW turns before/after it (crossing Add chunks);
                   a hit already visible inside an earlier span is skipped
  6. fallback   : nothing matched -> the newest FALLBACK_RESULTS turns
  7. budget     : stop at top_k results or MAX_RESPONSE_CHARS characters
v0.5: an optional dense channel (text-embedding-v4 cosine neighbours, app/embed.py) is fused by RRF into step 1.
v0.6: an optional evidence selector (app/selector.py, SELECTOR_MODEL) looks at the first SELECTOR_CANDIDATES blocks
and moves the ones holding evidence to the front; for 'fact' questions the unselected tail may be cut to
SELECTOR_TAIL blocks (standing rules survive the cut). It changes only the order of verbatim blocks.
v0.7: an optional cross-encoder reranker (app/rerank.py, RERANK_MODEL) re-orders the first RERANK_CANDIDATES blocks by
query-block relevance (replace) or by RRF with the retrieval rank, before the selector; verbatim blocks only.
v0.4: turns are rendered with resolved relative-time notes (app/timex.py, TIME_NOTES) and answer options enter
retrieval per OPTIONS_MODE (append | none | channel with OPTIONS_WEIGHT).
v0.9: an optional final cut (RETURN_CHARS, default off) after every reordering; standing rules survive it. Measured with
the organisers' answer and judge prompts it did not help (PLAN.md 10-15), so the whole assembled list goes out.
"""
from __future__ import annotations

import re
from datetime import timedelta

from contextlib import nullcontext
from typing import Any, ContextManager, Optional

from .config import Settings
from .models import MemoryResult
from .embed import Embedder
from .planner import Plan, Planner
from .rerank import Reranker
from .selector import Selection, Selector
from .store import EPOCH, Chunk, MemoryStore, cjk_only, render

def event_time(chunk: Chunk) -> Optional[str]:
    """`created_at` of a result = the hit turn's own timestamp (ISO 8601 UTC) when the adapter sent one, else nothing.
    The organisers' pipelines render each selected memory as "- [created_at] text", so this is "the memory timestamp" the
    answer prompt tells the reader to use for relative-time conversion (seen 2026-10-09 in data/clbench/pipeline.py).
    A memory that came without a timestamp (three of ScriptMem's four scripts) gets no `created_at` at all: the field is
    optional, and the day we happened to store it would read as the day the line was spoken."""
    if chunk.event_ms is not None:
        try:
            return (EPOCH + timedelta(milliseconds=chunk.event_ms)).strftime("%Y-%m-%dT%H:%M:%SZ")
        except (OverflowError, OSError, ValueError):
            pass
    return None


WORD_RE = re.compile(r"\w+", re.UNICODE)
CJK_CHAR = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
# Function words that carry no retrieval signal in English questions. Own list; alphabetical.
STOPWORDS = frozenset("""
a about after again against all also am an and any are as at be because been before being between both but by can
could did do does doing done down during each few for from further had has have having he her here hers herself him
himself his how however i if in into is it its itself just let may me might more most much must my myself no nor not
now of off on once only or other our ours ourselves out over own same she should so some still such than that the
their theirs them themselves then there these they this those through to too under until up us very was we were what
when where whether which while who whom why will with would yes yet you your yours yourself yourselves
""".split())
PREFIX_MIN = 5   # terms at least this long also match as prefixes ("illustrat"* covers what the stemmer misses)
RRF_K = 60


def word_terms(query: str, options: list[str] | None) -> list[str]:
    """Distinct query terms in first-seen order. Linear in the query length: a question may carry a whole reference
    document (the organisers' CL-bench prompt says "The question includes a reference document"), i.e. tens of
    thousands of words."""
    terms: list[str] = []
    seen: set[str] = set()
    for text in (query, *(options or ())):
        for raw in WORD_RE.findall(text.lower()):
            tok = raw.strip("_")
            if len(tok) < 2 or tok in seen or tok in STOPWORDS or CJK_CHAR.search(tok):
                continue
            seen.add(tok)
            terms.append(tok)
    return terms


def cjk_terms(query: str, options: list[str] | None) -> list[str]:
    """Overlapping character trigrams of the CJK text (same normalisation as the index)."""
    grams: list[str] = []
    for text in (query, *(options or ())):
        run = cjk_only(text)
        for i in range(len(run) - 2):
            g = run[i:i + 3]
            if g not in grams:
                grams.append(g)
    return grams


def fts_expression(terms: list[str], prefixes: bool = True) -> str:
    """FTS5 MATCH expression: quoted terms joined by OR so BM25 can rank partial matches."""
    parts = []
    for t in terms:
        quoted = '"' + t.replace('"', '""') + '"'
        parts.append(quoted + "*" if prefixes and len(t) >= PREFIX_MIN else quoted)
    return " OR ".join(parts)


def fuse(primary: list[tuple[Chunk, float]], secondary: list[tuple[Chunk, float]],
         weight: float = 1.0) -> list[tuple[Chunk, float]]:
    """Reciprocal rank fusion (the secondary list's contributions scaled by `weight`); a no-op when only one list has hits."""
    if not secondary:
        return primary
    if not primary:
        return secondary
    score: dict[str, float] = {}
    seen: dict[str, Chunk] = {}
    for ranked, w in ((primary, 1.0), (secondary, weight)):
        for position, (chunk, _) in enumerate(ranked, start=1):
            score[chunk.id] = score.get(chunk.id, 0.0) + w / (RRF_K + position)
            seen[chunk.id] = chunk
    return sorted(((seen[i], s) for i, s in score.items()), key=lambda pair: -pair[1])


def newest_first_within_band(hits: list[tuple[Chunk, float]], band: float) -> list[tuple[Chunk, float]]:
    """Group hits into bands of near-equal normalised score; inside a band the newer memory comes first."""
    if len(hits) < 2 or band <= 0:
        return hits
    top = hits[0][1] or 1.0

    def key(pair: tuple[Chunk, float]) -> tuple[int, int, int]:
        chunk, s = pair
        return (-int(round((s / top) / band)), -chunk.order_key[0], -chunk.order_key[1])

    return sorted(hits, key=key)


OPTION_PREFIX = re.compile(r"^\s*(?:\(?[A-Za-z]\)|[A-Za-z][\.\:：、])\s*")


def dedupe_options(query: str, options: list[str] | None) -> list[str] | None:
    """Drop answer options whose text is already inside the question. ScriptMem-style questions carry the lettered
    options in the question text AND in the request's `options` field (seen 2026-10-09 in the public release), so
    appending them again would double the distractor wording in the lexical and dense queries."""
    if not options:
        return options
    q = " ".join(query.lower().split())
    kept = []
    for opt in options:
        body = " ".join(OPTION_PREFIX.sub("", str(opt)).lower().split())
        if body and body not in q:
            kept.append(opt)
    return kept or None


OPTION_LINE = re.compile(r"^\s*\(?([A-H])[\.\):：]\s+(\S.*)$")


def split_question(query: str) -> tuple[str, list[str]]:
    """(stem, lettered options) when the question text itself carries the option list — ScriptMem's questions read
    "stem, blank line, A. … to F. …, blank line, 'Please provide the option … enclosed in parentheses, e.g., (X).'" and
    83 % of their words are option text (public release, 2026-10-09). The answer-format sentence after the list never
    describes the memory, so it is dropped from retrieval. Anything else comes back unchanged with no options."""
    lines = query.split("\n")
    first = next((i for i, ln in enumerate(lines) if (m := OPTION_LINE.match(ln)) and m.group(1) == "A"), None)
    if first is None:
        return query, []
    found: list[str] = []
    for ln in lines[first:]:
        m = OPTION_LINE.match(ln)
        if not m or ord(m.group(1)) != ord("A") + len(found):
            break
        found.append(ln.strip())
    stem = "\n".join(lines[:first]).strip()
    if len(found) < 2 or not stem:
        return query, []
    return stem, found


def retrieval_text(query: str, options: list[str] | None, settings: Settings) -> str:
    """What the dense channel embeds: the question, plus the options when OPTIONS_MODE=append."""
    if options and settings.options_mode == "append":
        return query + "\n" + "\n".join(options)
    return query


def search(store: MemoryStore, settings: Settings, query: str, options: list[str] | None,
           user_id: str, top_k: int, planner: Optional[Planner] = None,
           gate: Optional[ContextManager] = None, embedder: Optional[Embedder] = None,
           selector: Optional[Selector] = None, reranker: Optional[Reranker] = None) -> list[MemoryResult]:
    """Gated SQLite passes around the network calls (query embedding, planner, reranker, selector); network time never holds the gate."""
    meta: dict[str, Any] = {}
    stem, embedded = split_question(query)
    if embedded:                                  # options written into the question: retrieve with the stem + options
        query = stem
        options = embedded + [o for o in (dedupe_options("\n".join(embedded), options) or [])]
    options = dedupe_options(query, options)
    results = _retrieve(store, settings, query, options, user_id, top_k, planner, gate, embedder, meta)
    if reranker is not None and len(results) >= settings.rerank_min_results:
        head = results[:settings.rerank_candidates]
        hit_text = meta.get("hit_text", {})
        docs = [rerank_document(hit_text.get(r.id), r.content, settings.rerank_doc) for r in head]
        scores = reranker.scores(retrieval_text(query, options, settings), docs)   # outside the gate
        results = apply_rerank(results, scores, settings)
    if selector is not None and len(results) >= settings.selector_min_results:
        candidates = results[:settings.selector_candidates]
        selection = selector.select(query, options, [r.content for r in candidates])   # outside the gate
        results = apply_selection(results, selection, meta.get("rule_ids", set()), settings)
    return cap_response(results, settings, meta)


def cap_response(results: list[MemoryResult], settings: Settings, meta: dict[str, Any]) -> list[MemoryResult]:
    """v0.9 RETURN_CHARS: the last step, after every reordering, so the reranker and selector still choose from the full
    assembled list. Whole blocks are kept while the running total is below the cap (the same rule as the assembly
    budget); standing rules that fall behind the cut are kept after it, as the selector keeps them. With
    RETURN_FROM_CHARS, users whose whole stored text is at most that size are not cut at all."""
    cap = settings.return_chars
    if cap <= 0 or not results:
        return results
    if settings.return_from_chars > 0 and meta.get("corpus_chars", 0) <= settings.return_from_chars:
        return results
    kept: list[MemoryResult] = []
    used = 0
    for r in results:
        if used >= cap:
            break
        kept.append(r)
        used += len(r.content)
    rule_ids = meta.get("rule_ids", set())
    return kept + [r for r in results[len(kept):] if r.id in rule_ids]


def _retrieve(store: MemoryStore, settings: Settings, query: str, options: list[str] | None,
              user_id: str, top_k: int, planner: Optional[Planner], gate: Optional[ContextManager],
              embedder: Optional[Embedder], meta: dict[str, Any]) -> list[MemoryResult]:
    gate = gate or nullcontext()
    qvec: Any = None
    if not query.strip() and not options:
        planner = None                                   # nothing to plan or embed: the newest-turns fallback answers
    elif embedder is not None:
        qvec = embedder.embed_one(retrieval_text(query, options, settings))   # None on failure -> lexical only
    with gate, store.connect(user_id, create=False) as conn:   # one connection per Search; None = unknown user
        if conn is None:
            return []
        hits = _first_pass(store, conn, settings, query, options, user_id, top_k, qvec)
        if planner is None:
            return _assemble(store, conn, settings, query, options, user_id, top_k, hits, meta)
        snippets = [render(c, settings.time_notes, settings.turn_marks)[:400] for c, _ in hits[:settings.planner_snippets]]
    plan = planner.plan(query, options, snippets)          # outside the gate
    with gate, store.connect(user_id, create=False) as conn:
        if conn is None:
            return []
        if plan is not None and (plan.queries or plan.has_dates):
            # The first DENSE_HEAD lexical hits stay in front of everything the planner adds. Measured 2026-10-09 on
            # public LoCoMo-Refined (200 q): letting the expansion re-rank the head took hit@1 0.405 -> 0.140 and the
            # reader's accuracy went down with it even though recall@100 reached 1.0 - the head is what gets read.
            head = hits[:settings.dense_head]
            head_ids = {c.id for c, _ in head}
            tail = _expand(store, conn, settings, user_id, top_k, query, options, hits[len(head):], plan, qvec)
            hits = head + [h for h in tail if h[0].id not in head_ids]
            if plan.has_dates:   # a date range in the question outranks the head: in-range hits first, each group in order
                hits = _prefer_dates(hits, plan)
        return _assemble(store, conn, settings, query, options, user_id, top_k, hits, meta)


def rerank_document(hit: Optional[str], block: str, mode: str) -> str:
    """What the cross-encoder scores for one result. `block` is the rendered span, `hit` the matched turn inside it.
    The reranker reads only the first RERANK_DOC_CHARS characters, so on long turns the span's opening (the turns
    BEFORE the hit) can fill the whole window; on short turns the hit alone loses the neighbour that holds the answer."""
    if mode == "block" or not hit:
        return block
    if mode == "hit":
        return hit
    return hit if hit == block else hit + "\n" + block


def apply_rerank(results: list[MemoryResult], scores: Optional[list[float]], settings: Settings) -> list[MemoryResult]:
    """Re-order the first len(scores) blocks by the reranker (replace) or by RRF of retrieval rank and reranker rank;
    blocks behind the reranked head keep their order. Scores are re-issued so that order and score agree."""
    if not scores or len(scores) > len(results):
        return results
    n_head = len(scores)
    by_rerank = sorted(range(n_head), key=lambda i: (-scores[i], i))
    if settings.rerank_mode == "rrf":
        rr = {i: r for r, i in enumerate(by_rerank)}
        fused = sorted(range(n_head), key=lambda i: (-(1.0 / (RRF_K + i) + 1.0 / (RRF_K + rr[i])), i))
        order = fused
    else:
        order = by_rerank
    ordered = [results[i] for i in order] + results[n_head:]
    n = len(ordered)
    return [MemoryResult(id=r.id, content=r.content, created_at=r.created_at, score=round(1.0 - i / n, 4))
            for i, r in enumerate(ordered)]


def apply_selection(results: list[MemoryResult], selection: Optional[Selection], rule_ids: set[str],
                    settings: Settings) -> list[MemoryResult]:
    """Selected blocks first (in the selector's order), then the rest in their original order. For 'fact'
    questions with SELECTOR_TAIL >= 0 the unselected non-rule blocks are cut to that many; standing rules are
    always kept. Scores are re-issued so that order and score agree (the platform treats order as priority)."""
    if selection is None or not selection.blocks:
        return results
    picked = [i for i in selection.blocks if 0 <= i < len(results)]
    if not picked:
        return results
    chosen = set(picked)
    head = [results[i] for i in picked]
    rest = [r for j, r in enumerate(results) if j not in chosen]
    if selection.kind == "fact" and settings.selector_tail >= 0:
        rest = [r for r in rest if r.id not in rule_ids][:settings.selector_tail] + [r for r in rest if r.id in rule_ids]
    ordered = head + rest
    n = len(ordered)
    return [MemoryResult(id=r.id, content=r.content, created_at=r.created_at, score=round(1.0 - i / n, 4))
            for i, r in enumerate(ordered)]


def _first_pass(store: MemoryStore, conn, settings: Settings, query: str, options: list[str] | None,
                user_id: str, top_k: int, qvec: Any = None) -> list[tuple[Chunk, float]]:
    pool = top_k * 2
    opts = options if settings.options_mode == "append" else None
    by_words = store.fts_search(user_id, fts_expression(word_terms(query, opts)), pool, conn)
    by_trigrams = store.trigram_search(user_id, fts_expression(cjk_terms(query, opts), prefixes=False), pool, conn)
    hits = fuse(by_words, by_trigrams)
    if settings.options_mode == "channel" and options:
        # v0.4: the options as their own channel (idea: Hybrid Episodic Memory, Cycle-1 #6), so the question's terms
        # keep the lead and a distractor option alone cannot outrank evidence for the question itself
        o_words = word_terms("", options)
        o_grams = cjk_terms("", options)
        by_opts = fuse(store.fts_search(user_id, fts_expression(o_words), pool, conn) if o_words else [],
                       store.trigram_search(user_id, fts_expression(o_grams, prefixes=False), pool, conn) if o_grams else [])
        hits = fuse(hits, by_opts, weight=settings.options_weight)
    lexical = newest_first_within_band(hits, settings.recency_band)
    if qvec is None:
        return lexical
    # v0.5 dense channel: cosine neighbours of the question, fused by RRF (lexical first). This is what
    # separated the Cycle-1 top 10 from plain BM25: evidence that shares no words with the question.
    # DENSE_HEAD keeps the first N lexical hits in place (exact name/number matches are the most precise
    # signal we have) and lets the dense list reorder only what comes after them. The recency band is applied
    # to the lexical list and to the fused tail separately: BM25 and RRF scores live on different scales, and
    # banding them together sorted the whole tail newest-first (measured 10-09: hit@100 0.975 -> 0.944).
    dense = store.dense_search(user_id, qvec, pool, conn)
    head = lexical[:settings.dense_head]
    head_ids = {c.id for c, _ in head}
    tail = fuse([h for h in lexical if h[0].id not in head_ids], [d for d in dense if d[0].id not in head_ids],
                weight=settings.dense_weight)
    tail = newest_first_within_band(tail, settings.recency_band)
    head = [(c, 1.0 + 0.001 * (len(head) - i)) for i, (c, _) in enumerate(head)]   # scores above any RRF value
    return head + tail


def _expand(store: MemoryStore, conn, settings: Settings, user_id: str, top_k: int, query: str,
            options: list[str] | None, hits: list[tuple[Chunk, float]], plan: Plan, qvec: Any = None) -> list[tuple[Chunk, float]]:
    """Fold the planner's extra queries (and, with a date range, date-restricted searches of every query)
    into the first pass by reciprocal rank fusion, first pass first."""
    pool = top_k * 2
    fused = hits
    rng = (plan.date_from_ms, plan.date_to_ms) if plan.has_dates else None
    queries: list[tuple[str, list[str] | None]] = [(extra, None) for extra in plan.queries]
    if rng is not None:
        opts = options if settings.options_mode == "append" else None
        queries = [(query, opts)] + queries   # the original question, restricted to the period
    for text, opts in queries:
        words = fts_expression(word_terms(text, opts))
        grams = fts_expression(cjk_terms(text, opts), prefixes=False)
        more = fuse(store.fts_search(user_id, words, pool, conn), store.trigram_search(user_id, grams, pool, conn))
        if rng is not None:
            ranged = fuse(store.fts_search(user_id, words, pool, conn, rng), store.trigram_search(user_id, grams, pool, conn, rng))
            if qvec is not None and text == query:   # only the original question has an embedding
                ranged = fuse(ranged, store.dense_search(user_id, qvec, pool, conn, rng), weight=settings.dense_weight)
            more = fuse(ranged, more)
        fused = fuse(fused, more)
    return newest_first_within_band(fused, settings.recency_band)


def _prefer_dates(hits: list[tuple[Chunk, float]], plan: Plan) -> list[tuple[Chunk, float]]:
    """Hits inside the planner's date range come first, each group keeping its order."""
    inside = [h for h in hits if plan.in_range(h[0].event_ms)]
    outside = [h for h in hits if not plan.in_range(h[0].event_ms)]
    return inside + outside


def _assemble(store: MemoryStore, conn, settings: Settings, query: str, options: list[str] | None,
              user_id: str, top_k: int, hits: list[tuple[Chunk, float]],
              meta: Optional[dict[str, Any]] = None) -> list[MemoryResult]:
    top_score = hits[0][1] if hits else 1.0
    score_of = {chunk.id: s for chunk, s in hits}

    rule_budget = min(settings.rule_slots, top_k // 4)
    rules = store.standing_rules(user_id, rule_budget * 4, conn) if rule_budget > 0 else []
    rules.sort(key=lambda c: (-score_of.get(c.id, 0.0), -c.order_key[0], -c.order_key[1]))
    rules = rules[:rule_budget]
    if meta is not None:
        meta["rule_ids"] = {c.id for c in rules}
        if settings.return_chars > 0 and settings.return_from_chars > 0:
            meta["corpus_chars"] = store.corpus_size(user_id, conn)[1]

    queue: list[tuple[Chunk, float]] = []
    queued: set[str] = set()

    def enqueue(chunk: Chunk, s: float) -> None:
        if chunk.id not in queued:
            queued.add(chunk.id)
            queue.append((chunk, s))

    for chunk, s in hits[:settings.lead_hits]:
        enqueue(chunk, s)
    for chunk in rules:
        enqueue(chunk, score_of.get(chunk.id, top_score))
    for chunk, s in hits[settings.lead_hits:]:
        enqueue(chunk, s)
    if not hits:
        for chunk in store.newest(user_id, settings.fallback_results, conn):
            enqueue(chunk, 0.0)

    if settings.fit_chars > 0:
        _, stored_chars = store.corpus_size(user_id, conn)
        if 0 < stored_chars <= settings.fit_chars:
            return _assemble_fit(store, conn, settings, user_id, top_k, hits, queue, meta)

    results: list[MemoryResult] = []
    visible: set[tuple[str, int]] = set()   # (session, position) already shown inside an earlier span
    used = 0
    for chunk, s in queue:
        if len(results) >= top_k or used >= settings.max_response_chars:
            break
        if (chunk.session_id, chunk.pos) in visible and not chunk.standing:
            continue
        turns = store.span(chunk, settings.context_window, conn) if s > 0 or chunk.standing else [chunk]
        visible.update((t.session_id, t.pos) for t in turns)
        text = "\n".join(render(t, settings.time_notes, settings.turn_marks) for t in turns)
        used += len(text)
        results.append(MemoryResult(id=chunk.id, content=text, created_at=event_time(chunk),
                                    score=round(min(s / top_score, 1.0), 4) if top_score else None))
        if meta is not None:
            meta.setdefault("hit_text", {})[chunk.id] = render(chunk, settings.time_notes, settings.turn_marks)   # the matched turn alone, for the reranker
    return results


def _assemble_fit(store: MemoryStore, conn, settings: Settings, user_id: str, top_k: int,
                  hits: list[tuple[Chunk, float]], queue: list[tuple[Chunk, float]],
                  meta: Optional[dict[str, Any]] = None) -> list[MemoryResult]:
    """v0.8 fit mode: the user's whole memory is small enough for the answer window, so return ALL of it, most
    relevant first. The conversation is cut into at most top_k contiguous segments: the leading hits get a segment
    centred on them, the rest is split evenly; segments come out in the order their best turn is ranked, the
    untouched ones after that in conversation order. Every turn appears exactly once.

    Why: with fixed hit ± window blocks the share of a conversation we return depends on how long its turns are —
    about three quarters of a LoCoMo conversation, about a quarter of a play script with one-line turns — while the
    reader has room for all of it and answers better with more of the same conversation in view (public LoCoMo-Refined,
    2026-10-09: full list 0.819, first 30 blocks 0.750)."""
    turns = store.all_turns(user_id, conn)
    n = len(turns)
    if n == 0:
        return []
    text = [render(t, settings.time_notes, settings.turn_marks) for t in turns]
    size = [len(x) + 1 for x in text]
    total = sum(size)
    span = max(settings.fit_span_min, -(-total // max(1, int(top_k * settings.fit_fill))))
    at = {t.id: i for i, t in enumerate(turns)}
    owner = [-1] * n
    segs: dict[int, list[int]] = {}    # segment number -> [first turn index, last turn index]

    def claim(lo: int, hi: int) -> None:
        k = len(segs)
        segs[k] = [lo, hi]
        for j in range(lo, hi + 1):
            owner[j] = k

    def free(i: int, ref: int) -> bool:
        return 0 <= i < n and owner[i] < 0 and turns[i].session_id == turns[ref].session_id

    # 1. the leading hits each get a segment grown around them, alternately backwards and forwards
    anchored = 0
    for chunk, s in hits:
        if anchored >= settings.fit_anchors:
            break
        i = at.get(chunk.id)
        if i is None or owner[i] >= 0:
            continue
        lo = hi = i
        acc = size[i]
        backwards = True
        while acc < span:
            can_back, can_fwd = free(lo - 1, i), free(hi + 1, i)
            if not can_back and not can_fwd:
                break
            if (backwards and can_back) or not can_fwd:
                lo -= 1
                acc += size[lo]
            else:
                hi += 1
                acc += size[hi]
            backwards = not backwards
        claim(lo, hi)
        anchored += 1

    # 2. everything else: each run of unclaimed turns is cut into pieces of about `span` characters; a sliver next
    #    to an anchored segment joins that segment instead of becoming a block of its own
    i = 0
    while i < n:
        if owner[i] >= 0:
            i += 1
            continue
        j = i
        while j + 1 < n and owner[j + 1] < 0 and turns[j + 1].session_id == turns[i].session_id:
            j += 1
        run = sum(size[i:j + 1])
        before = owner[i - 1] if i > 0 and owner[i - 1] >= 0 and turns[i - 1].session_id == turns[i].session_id else -1
        after = owner[j + 1] if j + 1 < n and owner[j + 1] >= 0 and turns[j + 1].session_id == turns[i].session_id else -1
        if run < span // 4 and (before >= 0 or after >= 0):
            k = before if before >= 0 else after
            segs[k] = [min(segs[k][0], i), max(segs[k][1], j)]
            for t in range(i, j + 1):
                owner[t] = k
        else:
            pieces = max(1, -(-run // span))
            target = run / pieces
            start, acc, made = i, 0, 0
            for t in range(i, j + 1):
                acc += size[t]
                if acc >= target and made < pieces - 1 and t < j:
                    claim(start, t)
                    start, acc, made = t + 1, 0, made + 1
            claim(start, j)
        i = j + 1

    # 3. never more segments than results allowed: merge the smallest neighbours (inside a session while possible)
    def seg_size(k: int) -> int:
        return sum(size[segs[k][0]:segs[k][1] + 1])

    while len(segs) > top_k:
        order = sorted(segs, key=lambda k: segs[k][0])
        pairs = [(a, b) for a, b in zip(order, order[1:])]
        same = [(a, b) for a, b in pairs if turns[segs[a][1]].session_id == turns[segs[b][0]].session_id]
        a, b = min(same or pairs, key=lambda ab: seg_size(ab[0]) + seg_size(ab[1]))
        segs[a] = [segs[a][0], segs[b][1]]
        for t in range(segs[b][0], segs[b][1] + 1):
            owner[t] = a
        del segs[b]

    # 4. order: by the rank of the best turn inside each segment, then the untouched segments in conversation order
    best: dict[int, Chunk] = {}
    ranked: list[int] = []
    for chunk, _ in queue:
        i = at.get(chunk.id)
        if i is None:
            continue
        k = owner[i]
        if k not in best:
            best[k] = chunk
            ranked.append(k)
    rest = [k for k in sorted(segs, key=lambda k: segs[k][0]) if k not in best]

    picked: list[tuple[Chunk, str]] = []
    used = 0
    for k in ranked + rest:
        if len(picked) >= top_k or used >= settings.max_response_chars:
            break
        lo, hi = segs[k]
        body = "\n".join(text[lo:hi + 1])
        used += len(body)
        head = best.get(k) or turns[lo]
        picked.append((head, body))
        if meta is not None and k in best:
            meta.setdefault("hit_text", {})[head.id] = text[at[head.id]]
    count = len(picked)
    return [MemoryResult(id=head.id, content=body, created_at=event_time(head), score=round(1.0 - i / count, 4))
            for i, (head, body) in enumerate(picked)]
