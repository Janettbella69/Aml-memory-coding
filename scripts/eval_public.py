"""Run the organisers' PUBLIC LoCoMo-Refined data (github.com/mem-eval-suite/LoCoMo_refined, data/public/) through
the service in-process and report evidence hit@k, optionally with gpt-4o-mini answer accuracy.

    PLANNER_MODEL= uv run --env-file .env python scripts/eval_public.py --data /path/to/data/public --label lexical
    EMBED_MODEL=text-embedding-v4 PLANNER_MODEL= uv run --env-file .env python scripts/eval_public.py --data ... --label dense
    ... --judge --limit 40          # answer + grade a 40-questions-per-conversation subsample

Rules (competition FAQ Q9/Q10): public subsets are for understanding the task, interface pre-checks and reproduction,
never for training, fine-tuning or fitting to the questions. We use this script for one-shot configuration checks
and record every run in docs/METHOD_CARD.md. The data stays outside the repository (CC BY-NC 4.0).

Mapping: one Add per 20 messages of a session; content "Speaker: text" plus " [photo: <blip caption>]" when the turn
carried an image (the textual track sees captions, not pixels); role as given; timestamp = session time + 1 min per
turn. Categories per the LoCoMo paper: 1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop (no adversarial in the
public set). A question's evidence is "hit" at rank r when the evidence turn's text appears inside the r-th result.
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
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CATEGORY = {"1": "multi_hop", "2": "temporal", "3": "open_domain", "4": "single_hop", "5": "adversarial"}
WS = re.compile(r"\s+")


def norm(s: str) -> str:
    return WS.sub(" ", s.strip().lower())


def parse_when(s: str) -> int | None:
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %B %Y", "%d %B, %Y"):
        try:
            return int(datetime.strptime(s.strip(), fmt).replace(tzinfo=timezone.utc).timestamp() * 1000)
        except ValueError:
            continue
    return None


def message_content(m: dict) -> str:
    text = f"{m.get('speaker', '')}: {m.get('text', '')}".strip()
    if m.get("images") and m.get("blip_caption"):
        text += f" [photo: {m['blip_caption']}]"
    return text


def load(data_dir: Path):
    convs = [json.loads(l) for l in (data_dir / "conversations.jsonl").open(encoding="utf-8") if l.strip()]
    qs = [json.loads(l) for l in (data_dir / "questions.jsonl").open(encoding="utf-8") if l.strip()]
    by_sample: dict[str, list[dict]] = defaultdict(list)
    for q in qs:
        by_sample[q["sample_id"]].append(q)
    return convs, by_sample


def add_payloads(conv: dict, run: str):
    user = f"public:{run}:locomo_refined:{conv['sample_id']}"
    n = 0
    for s in conv["sessions"]:
        when = parse_when(s.get("date_time") or "")
        msgs_all = [{"role": m.get("role") or "user", "content": message_content(m),
                     **({"timestamp": when + i * 60_000} if when is not None else {})}
                    for i, m in enumerate(s["messages"])]
        for start in range(0, len(msgs_all), 20):
            yield {"request_id": f"{user}:chunk-{n}", "user_id": user,
                   "session_id": f"{user}:session:{s['session_index']}", "messages": msgs_all[start:start + 20]}
            n += 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="directory with conversations.jsonl and questions.jsonl (outside the repo)")
    ap.add_argument("--label", default="public")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--limit", type=int, help="first N questions per conversation")
    ap.add_argument("--conversations", help="comma-separated conversation indexes to run (default: all)")
    ap.add_argument("--judge", action="store_true", help="also answer + grade with gpt-4o-mini (OPENAI_* env)")
    ap.add_argument("--max-context-chars", type=int, default=400_000)
    args = ap.parse_args()

    from fastapi.testclient import TestClient

    from app.config import Settings
    from app.main import create_app
    from app.store import MemoryStore

    tmp = tempfile.TemporaryDirectory()
    settings = replace(Settings.from_env(), api_key="", data_dir=os.path.join(tmp.name, "d"), log_level="WARNING")
    app = create_app(settings=settings, store=MemoryStore(settings.data_dir))
    client = TestClient(app)
    client.__enter__()
    run = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

    judge = None
    if args.judge:
        from scripts.judge_local import ANSWER_SYSTEM, JUDGE_SYSTEM, chat
        key = os.environ.get("OPENAI_API_KEY") or sys.exit("OPENAI_API_KEY is not set")
        judge = (ANSWER_SYSTEM, JUDGE_SYSTEM, chat, key)

    convs, by_sample = load(Path(args.data))
    wanted = {int(x) for x in args.conversations.split(",")} if args.conversations else None
    rows: list[dict] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    t_start = time.time()
    for idx, conv in enumerate(convs):
        if wanted is not None and idx not in wanted:
            continue
        t0 = time.time()
        for payload in add_payloads(conv, run):
            r = client.post("/v1/memories/add", json=payload)
            assert r.status_code == 200, r.text
        add_s = time.time() - t0
        user = f"public:{run}:locomo_refined:{conv['sample_id']}"
        questions = by_sample.get(conv["sample_id"], [])
        questions = questions[: args.limit] if args.limit else questions
        for q in questions:
            t0 = time.time()
            data = client.post("/v1/memories/search", json={"query": q["question"], "user_id": user, "top_k": args.top_k}).json()["data"]
            latency = time.time() - t0
            contents = [norm(str(r.get("content", ""))) for r in data]
            ranks: dict[str, int | None] = {}
            for ev in q.get("evidence_messages", []):
                et = norm(ev.get("text", ""))
                if not et:
                    continue
                ranks[ev["dia_id"]] = next((pos for pos, c in enumerate(contents, start=1) if et in c), None)
            found = [r for r in ranks.values() if r is not None]
            first = min(found) if found else None
            all20 = all(r is not None and r <= 20 for r in ranks.values()) if ranks else None
            row = {"qa_id": q["qa_id"], "category": CATEGORY.get(str(q["category"]), str(q["category"])),
                   "multimodal": bool(q.get("is_multi_modality")), "n_evidence": len(ranks), "first_rank": first,
                   "all_in_20": all20, "latency_s": round(latency, 3),
                   "chars": sum(len(str(r.get("content", ""))) for r in data), "results": len(data)}
            if judge is not None:
                ANSWER_SYSTEM, JUDGE_SYSTEM, chat, key = judge
                context = "\n\n".join(f"[memory {i}]\n{d['content']}" for i, d in enumerate(data, 1))[: args.max_context_chars]
                try:
                    answer, u1 = chat([{"role": "system", "content": ANSWER_SYSTEM},
                                       {"role": "user", "content": f"Memories:\n{context}\n\nQuestion: {q['question']}"}], 120, key)
                    reference = " | ".join(str(a) for a in q.get("answer", [])) + "  (any one of these is a complete correct answer)"
                    verdict, u2 = chat([{"role": "system", "content": JUDGE_SYSTEM},
                                        {"role": "user", "content": f"Question: {q['question']}\nReference: {reference}\nAnswer: {answer}"}], 3, key)
                except Exception as exc:   # a gateway refusal (e.g. HTTP 403 moderation) must not kill a 200-question run
                    row["answer"] = None
                    row["correct"] = False
                    row["error"] = type(exc).__name__ + ": " + str(exc)[:120]
                    print(f"  {q['qa_id']}: judge call failed ({row['error']})", flush=True)
                    rows.append(row)
                    continue
                for u in (u1, u2):
                    usage["prompt_tokens"] += u.get("prompt_tokens", 0)
                    usage["completion_tokens"] += u.get("completion_tokens", 0)
                verdict = verdict.upper().split()[0] if verdict else "INCORRECT"
                row["answer"] = answer
                row["correct"] = verdict == "CORRECT"
            rows.append(row)
        print(f"{conv['sample_id']}: add {add_s:.1f}s, {len(questions)} questions, elapsed {time.time() - t_start:.0f}s", flush=True)

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["category"]].append(r)
    groups["multimodal"] = [r for r in rows if r["multimodal"]]
    groups["ALL"] = rows
    print(f"\n### eval_public · LoCoMo-Refined · label={args.label} · questions={len(rows)} · top_k={args.top_k}")
    head = "| category | n | hit@1 | hit@5 | hit@10 | hit@20 | hit@100 | all@20 | mrr | chars_avg | latency p50"
    print(head + (" | accuracy" if judge else "") + " |")
    print("|---" * (12 if judge else 11) + "|")
    for cat in ["single_hop", "multi_hop", "temporal", "open_domain", "multimodal", "ALL"]:
        g = [r for r in groups.get(cat, []) if r["n_evidence"] > 0]
        if not g:
            continue
        def hit(k: int) -> float:
            return sum(1 for r in g if r["first_rank"] is not None and r["first_rank"] <= k) / len(g)
        mrr = sum(1.0 / r["first_rank"] for r in g if r["first_rank"]) / len(g)
        all20 = [r["all_in_20"] for r in g if r["all_in_20"] is not None]
        line = (f"| {cat} | {len(g)} | {hit(1):.3f} | {hit(5):.3f} | {hit(10):.3f} | {hit(20):.3f} | {hit(100):.3f} | "
                f"{(sum(all20) / len(all20)) if all20 else 0:.3f} | {mrr:.3f} | {int(statistics.mean(r['chars'] for r in g))} | "
                f"{statistics.median(r['latency_s'] for r in g):.2f}s")
        if judge:
            line += f" | {sum(1 for r in g if r.get('correct')) / len(g):.3f}"
        print(line + " |")
    if judge:
        cost = usage["prompt_tokens"] * 0.15 / 1e6 + usage["completion_tokens"] * 0.60 / 1e6
        print(f"\njudge tokens: {usage['prompt_tokens']} in / {usage['completion_tokens']} out · ≈ ${cost:.2f}")
    out = Path("evalset/results") / f"public-locomo-refined-{args.label}-{run}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"label": args.label, "rows": rows, "usage": usage}, indent=1, ensure_ascii=False))
    print(f"wrote {out}")
    client.__exit__(None, None, None)
    tmp.cleanup()


if __name__ == "__main__":
    main()
