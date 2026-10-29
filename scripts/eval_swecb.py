"""Local retrieval proxy for the coding track: SWEContextBench (github.com/jiayuanz3/SWEContextBench, MIT on Hugging Face,
arXiv:2602.08316) run through the service in-process. The organisers list it as the coding track's reference catalog; the
scored suite (CAMBench Coding) is private, so this is a retrieval check, never a board prediction.

    PLANNER_MODEL= EMBED_MODEL= uv run python scripts/eval_swecb.py --cases ".../SWEContextBench Lite Past Experience" \
        --related ".../SWEContextBench Lite" --relationship relationship.json --label v1.0.0-lexical

Data: the "Lite Past Experience" folder holds 300 Claude Code transcripts (JSONL, one past task each; the first user prompt
names the instance_id), the "Lite" folder 99 related tasks (SWE-bench JSON: repo, instance_id, problem_statement), and
relationship.json pairs them ([{related_instance_id, experience_instance_id}], exported from
data/SWEContextBench_Relationship.parquet). Everything stays outside the repository.

Mapping, after the API guide ("Coding preserves repository/session trajectories"; messages[].role is user or assistant;
content is a string): one Add per transcript (--chunk N splits it every N messages instead); user_id = repository,
session_id = transcript id; the user prompt and tool results (role user in the transcript) -> role user, tool results
prefixed "[tool result]"; assistant text -> assistant; thinking -> assistant "[thinking] ..." (--no-thinking drops it);
tool calls -> assistant "[tool call] <name> <input json>"; timestamp = the record's own timestamp.

Search: the related task's problem_statement as the query (no options), top_k 100, scope = its repository. A question is
"hit" at rank r when the r-th result belongs to a paired past session; "statement" when the paired session's opening turn
(the task description) is itself among the results. Chars = text returned.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

INSTANCE_RE = re.compile(r"instance_id:\s*(\S+)")
HEAD_CHARS = 60_000   # "what the reader sees first" window for the head-share metric


def repo_of(instance_id: str) -> str:
    """'scikit-learn__scikit-learn-13142' -> 'scikit-learn/scikit-learn'."""
    owner, rest = instance_id.split("__", 1)
    return f"{owner}/{rest.rsplit('-', 1)[0]}"


def load_transcript(path: Path, thinking: bool) -> tuple[str | None, list[dict]]:
    from app.models import parse_timestamp

    instance = None
    messages: list[dict] = []
    for raw in path.open(encoding="utf-8"):
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            continue
        msg = rec.get("message")
        if rec.get("type") not in ("user", "assistant") or not isinstance(msg, dict):
            continue
        role = "assistant" if rec["type"] == "assistant" else "user"
        content = msg.get("content")
        texts: list[str] = []
        if isinstance(content, str):
            texts.append(content)
            if instance is None and role == "user":
                m = INSTANCE_RE.search(content)
                instance = m.group(1) if m else None
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                kind = part.get("type")
                if kind == "text":
                    texts.append(part.get("text") or "")
                elif kind == "thinking" and thinking:
                    texts.append("[thinking] " + (part.get("thinking") or ""))
                elif kind == "tool_use":
                    texts.append(f"[tool call] {part.get('name')} " + json.dumps(part.get("input", {}), ensure_ascii=False))
                elif kind == "tool_result":
                    body = part.get("content")
                    if isinstance(body, list):
                        body = "\n".join(x.get("text", "") for x in body if isinstance(x, dict))
                    texts.append("[tool result]\n" + ("" if body is None else str(body)))
        text = "\n".join(t for t in texts if t and t.strip())
        if text.strip():
            messages.append({"role": role, "content": text, "timestamp": parse_timestamp(rec.get("timestamp"))})
    return instance, messages


def chunks(messages: list[dict], size: int) -> list[list[dict]]:
    if size <= 0:
        return [messages]
    return [messages[i:i + size] for i in range(0, len(messages), size)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", required=True, help="folder of past-experience transcripts (*.jsonl)")
    ap.add_argument("--related", required=True, help="folder of related tasks (*.json, SWE-bench fields)")
    ap.add_argument("--relationship", required=True, help="JSON list of {related_instance_id, experience_instance_id}")
    ap.add_argument("--label", default="run")
    ap.add_argument("--chunk", type=int, default=0, help="messages per Add; 0 = one Add per transcript (default)")
    ap.add_argument("--no-thinking", action="store_true", help="drop the transcript's thinking blocks")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0, help="only the first N related tasks")
    ap.add_argument("--data-dir", default="", help="reuse a built index (default: a fresh temporary directory)")
    ap.add_argument("--out", default="", help="append one JSON line with the summary to this file")
    ap.add_argument("--details", default="", help="write one JSON line per question with its first 10 results (kind, size, paired)")
    ap.add_argument("-v", "--verbose", action="store_true", help="one line per question")
    args = ap.parse_args()

    from app.config import Settings
    from app.embed import embedder_from_env, to_blob
    from app.models import AddRequest
    from app.planner import planner_from_env
    from app.rerank import reranker_from_env
    from app.search import search
    from app.selector import selector_from_env
    from app.store import MemoryStore

    tmp = tempfile.TemporaryDirectory() if not args.data_dir else None
    data_dir = args.data_dir or os.path.join(tmp.name, "d")
    settings = replace(Settings.from_env(), api_key="", data_dir=data_dir, log_level="WARNING")
    store = MemoryStore(settings.data_dir, code_terms=settings.code_terms)
    planner = planner_from_env(settings.planner_model, settings.planner_max_queries, settings.planner_timeout)
    embedder = embedder_from_env(settings.embed_model, settings.embed_dims, settings.embed_timeout, settings.embed_rps)
    selector = selector_from_env(settings.selector_model, settings.selector_timeout, settings.selector_max_select, settings.selector_block_chars)
    reranker = reranker_from_env(settings.rerank_provider, settings.rerank_model, settings.rerank_timeout, settings.rerank_doc_chars)

    # ---- past experience -> Add
    transcripts = sorted(Path(args.cases).glob("*.jsonl"))
    session_of_instance: dict[str, str] = {}
    user_of_session: dict[str, str] = {}
    t0 = time.time()
    adds = turns = chars = 0
    fresh = not args.data_dir or not any(Path(data_dir).rglob("*.db"))
    for path in transcripts:
        instance, messages = load_transcript(path, thinking=not args.no_thinking)
        if instance is None or not messages:
            continue
        sid = path.stem
        user = repo_of(instance)
        session_of_instance[instance] = sid
        user_of_session[sid] = user
        if not fresh:
            continue
        for i, part in enumerate(chunks(messages, args.chunk)):
            payload = AddRequest(request_id=f"swecb:{sid}:chunk-{i}", messages=part, user_id=user, session_id=sid)
            vectors = identity = None
            if embedder is not None:
                vectors = [to_blob(v) if v is not None else None for v in embedder.embed([m.text() for m in payload.messages])]
                identity = (embedder.model, embedder.dims)
            turns += store.add(payload, vectors, identity)
            adds += 1
            chars += sum(len(m["content"]) for m in part)
    add_seconds = time.time() - t0

    # result id -> (session, pos); the opening turn of every session; the kind and size of every turn
    session_of_id: dict[str, tuple[str, int]] = {}
    opening: dict[str, str] = {}
    kind_of_id: dict[str, str] = {}
    role_of_id: dict[str, str] = {}
    chars_of_id: dict[str, int] = {}
    for user in sorted(set(user_of_session.values())):
        with store.connect(user, create=False) as conn:
            if conn is None:
                continue
            for row in conn.execute("SELECT id, session_id, pos, role, substr(content, 1, 14) AS head, length(content) AS n"
                                    " FROM memories ORDER BY session_id, pos"):
                session_of_id[row["id"]] = (row["session_id"], row["pos"])
                opening.setdefault(row["session_id"], row["id"])
                head = row["head"]
                kind_of_id[row["id"]] = ("tool_result" if head.startswith("[tool result]") else "tool_call" if head.startswith("[tool call]")
                                        else "thinking" if head.startswith("[thinking]") else "prompt" if row["role"] == "user" else "assistant")
                role_of_id[row["id"]] = row["role"]
                chars_of_id[row["id"]] = int(row["n"])
    details = open(args.details, "w", encoding="utf-8") if args.details else None

    # ---- related tasks -> Search
    pairs: dict[str, set[str]] = defaultdict(set)
    for row in json.loads(Path(args.relationship).read_text(encoding="utf-8")):
        sid = session_of_instance.get(row["experience_instance_id"])
        if sid is not None:
            pairs[row["related_instance_id"]].add(sid)
    related = sorted(Path(args.related).glob("*.json"))
    if args.limit:
        related = related[:args.limit]
    rows = []
    skipped = 0
    t1 = time.time()
    for path in related:
        task = json.loads(path.read_text(encoding="utf-8"))
        expected = pairs.get(task["instance_id"], set())
        if not expected:
            skipped += 1
            continue
        results = search(store, settings, task["problem_statement"], None, task["repo"], args.top_k,
                         planner=planner, embedder=embedder, selector=selector, reranker=reranker)
        sessions = [session_of_id.get(r.id, ("?", 0))[0] for r in results]
        rank = next((i + 1 for i, s in enumerate(sessions) if s in expected), None)
        order: list[str] = []
        for s in sessions:
            if s not in order:
                order.append(s)
        session_rank = next((i + 1 for i, s in enumerate(order) if s in expected), None)   # sessions in order of first appearance
        ids = {r.id for r in results}
        statement = any(opening.get(s) in ids for s in expected)
        total = sum(len(r.content) for r in results)
        paired = sum(len(r.content) for r, s in zip(results, sessions) if s in expected)
        head_chars = head_paired = 0          # what a reader sees first: the paired share of the first HEAD_CHARS characters
        for r, s in zip(results, sessions):
            if head_chars >= HEAD_CHARS:
                break
            take = min(len(r.content), HEAD_CHARS - head_chars)
            head_chars += take
            head_paired += take if s in expected else 0
        rows.append({"instance_id": task["instance_id"], "repo": task["repo"], "rank": rank, "session_rank": session_rank,
                     "statement": statement, "results": len(results), "chars": total,
                     "paired_share": (paired / total) if total else 0.0,
                     "head_share": (head_paired / head_chars) if head_chars else 0.0,
                     "sessions_top10": len(set(sessions[:10]))})
        if details is not None:
            top = [{"rank": i + 1, "paired": s in expected, "session": s, "kind": kind_of_id.get(r.id, "?"),
                    "role": role_of_id.get(r.id, "?"), "opening": opening.get(s) == r.id,
                    "turn_chars": chars_of_id.get(r.id, 0), "block_chars": len(r.content)}
                   for i, (r, s) in enumerate(zip(results, sessions))]
            details.write(json.dumps({"instance_id": task["instance_id"], "rank": rank, "statement": statement,
                                      "expected": sorted(expected), "top": top}) + "\n")
        if args.verbose:
            print(f"{task['instance_id']:34} rank={rank!s:5} statement={int(statement)} results={len(results):3} "
                  f"chars={total:7,} paired={100 * rows[-1]['paired_share']:4.0f}% sessions@10={rows[-1]['sessions_top10']}")
    search_seconds = time.time() - t1

    def hit(k: int, subset=None, key: str = "rank") -> float:
        rs = subset if subset is not None else rows
        return sum(1 for r in rs if r[key] is not None and r[key] <= k) / len(rs) if rs else 0.0

    def mrr(subset=None, key: str = "rank") -> float:
        rs = subset if subset is not None else rows
        return sum(1 / r[key] for r in rs if r[key] is not None) / len(rs) if rs else 0.0

    n = len(rows)
    summary = {
        "label": args.label, "questions": n, "skipped_unpaired": skipped, "transcripts": len(session_of_instance),
        "adds": adds, "turns": turns, "chars_added": chars, "chunk": args.chunk, "thinking": not args.no_thinking,
        "hit@1": hit(1), "hit@5": hit(5), "hit@10": hit(10), f"hit@{args.top_k}": hit(args.top_k), "mrr": mrr(),
        "session_hit@1": hit(1, key="session_rank"), "session_hit@3": hit(3, key="session_rank"),
        "session_hit@5": hit(5, key="session_rank"), "session_mrr": mrr(key="session_rank"),
        "statement": sum(1 for r in rows if r["statement"]) / n if n else 0.0,
        "results_median": statistics.median(r["results"] for r in rows) if rows else 0,
        "chars_median": statistics.median(r["chars"] for r in rows) if rows else 0,
        "paired_share_median": statistics.median(r["paired_share"] for r in rows) if rows else 0.0,
        "head_share_median": statistics.median(r["head_share"] for r in rows) if rows else 0.0,
        "sessions_top10_median": statistics.median(r["sessions_top10"] for r in rows) if rows else 0,
        "add_seconds": round(add_seconds, 1), "search_seconds": round(search_seconds, 1),
        "settings": {k: v for k, v in vars(settings).items() if k not in ("api_key", "data_dir")},
    }
    print(f"\n[{args.label}] {n} questions ({skipped} without a loaded pair), {len(session_of_instance)} transcripts, "
          f"{turns:,} turns, {chars:,} chars added in {add_seconds:.0f}s; search {search_seconds:.0f}s")
    print(f"  paired session by block: hit@1 {hit(1):.3f}  hit@5 {hit(5):.3f}  hit@10 {hit(10):.3f}  hit@{args.top_k} {hit(args.top_k):.3f}  MRR {mrr():.3f}")
    print(f"  paired session by session order: hit@1 {summary['session_hit@1']:.3f}  hit@3 {summary['session_hit@3']:.3f}  "
          f"hit@5 {summary['session_hit@5']:.3f}  MRR {summary['session_mrr']:.3f}")
    print(f"  its task statement returned: {summary['statement']:.3f}   results/question median {summary['results_median']:.0f}   "
          f"chars median {summary['chars_median']:,.0f}   paired share: whole list median {100 * summary['paired_share_median']:.0f}%, "
          f"first {HEAD_CHARS // 1000}k chars median {100 * summary['head_share_median']:.0f}%   "
          f"distinct sessions in top 10 median {summary['sessions_top10_median']:.0f}")
    by_repo: dict[str, list] = defaultdict(list)
    for r in rows:
        by_repo[r["repo"]].append(r)
    print(f"  {'repository':28} {'n':>3} {'s@1':>6} {'s@3':>6} {'sMRR':>6} {'stmt':>5} {'head%':>6}")
    for repo, rs in sorted(by_repo.items(), key=lambda kv: -len(kv[1])):
        print(f"  {repo:28} {len(rs):3} {hit(1, rs, 'session_rank'):6.2f} {hit(3, rs, 'session_rank'):6.2f} {mrr(rs, 'session_rank'):6.2f} "
              f"{sum(1 for r in rs if r['statement']) / len(rs):5.2f} {100 * statistics.median(r['head_share'] for r in rs):6.0f}")
    if args.out:
        with open(args.out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(summary, ensure_ascii=False) + "\n")
    if details is not None:
        details.close()
    store.close()
    if tmp is not None:
        tmp.cleanup()


if __name__ == "__main__":
    main()
