"""Run OUR script-style proxy through the service and score it the way the organisers score ScriptMem.

Corpus  : a public-domain play parsed into ScriptMem's conversation schema (dialogue + narration lines, no timestamps).
Questions: written by us in ScriptMem's public format (stem + lettered options + answer-format sentence inside the question,
           options also sent in the top-level `options` array as the API guide describes).
Answer  : the organisers' ScriptMem prompt (data/scriptmem/pipeline.py, CHOICE_ANSWER_TEMPLATE), gpt-4o-mini, temperature 0.
Score   : the organisers' exact-letter scorer (predicted_letters + score_item) from the same file.
This is not ScriptMem (its source text is not published and its questions are not used); it measures how our assembly
behaves on memory with ScriptMem's shape: one-line turns, a whole work that would fit into the answer window.

    curl -sL -o /tmp/pg14095.txt https://www.gutenberg.org/cache/epub/14095/pg14095.txt
    python evalset/script_proxy/parse_play.py /tmp/pg14095.txt /tmp/play.json
    git clone --depth 1 https://github.com/AML-memory/agent-memory-leaderboard /tmp/aml-official
    AML_OFFICIAL=/tmp/aml-official FIT_CHARS=0      uv run --env-file .env python scripts/eval_script_proxy.py --play /tmp/play.json --label classic
    AML_OFFICIAL=/tmp/aml-official FIT_CHARS=400000 uv run --env-file .env python scripts/eval_script_proxy.py --play /tmp/play.json --label fit
    (--no-answer prints the retrieval metrics only and needs no chat key)
"""
from __future__ import annotations
import argparse, importlib.util, json, os, re, statistics, sys, tempfile, time, urllib.error, urllib.request
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("aml_scriptmem", str(Path(os.environ["AML_OFFICIAL"]) / "data/scriptmem/pipeline.py"))
official = importlib.util.module_from_spec(spec); spec.loader.exec_module(official)

KEY = os.environ.get("OPENAI_API_KEY") or sys.exit("OPENAI_API_KEY is not set")
BASE = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
MODEL = os.environ.get("ANSWER_MODEL", "openai/gpt-4o-mini")
WS = re.compile(r"\s+")
def norm(s: str) -> str: return WS.sub(" ", s.strip().lower())


def chat(prompt: str, max_tokens: int = 256) -> tuple[str, dict]:
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": max_tokens}
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}, method="POST")
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=180) as r: out = json.load(r)
            return (out["choices"][0]["message"].get("content") or "").strip(), out.get("usage", {})
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code in (400, 401, 402, 404): raise
            if attempt == 5: raise
            time.sleep(2 ** attempt)
    raise RuntimeError


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--play", required=True, help="play.json written by evalset/script_proxy/parse_play.py (kept outside the repository)")
    ap.add_argument("--questions", default=str(REPO / "evalset/script_proxy/questions.json"))
    ap.add_argument("--label", default="proxy"); ap.add_argument("--out-dir", default=str(REPO / "evalset/results"))
    ap.add_argument("--limit", type=int); ap.add_argument("--no-answer", action="store_true", help="retrieval metrics only")
    ap.add_argument("--types", help="comma-separated qa_type filter, e.g. ordering")
    ap.add_argument("--chronological", action="store_true", help="experiment: show the returned blocks in conversation order instead of rank order")
    ap.add_argument("--evidence-only", action="store_true", help="experiment (oracle): keep only the returned blocks that hold a gold evidence line")
    args = ap.parse_args()
    play = json.load(open(args.play, encoding="utf-8")); questions = json.load(open(args.questions, encoding="utf-8"))
    if args.types: questions = [q for q in questions if q["qa_type"] in set(args.types.split(","))]
    if args.limit: questions = questions[: args.limit]

    from fastapi.testclient import TestClient
    from app.config import Settings
    from app.main import create_app
    from app.store import MemoryStore
    tmp = tempfile.TemporaryDirectory()
    settings = replace(Settings.from_env(), api_key="", data_dir=os.path.join(tmp.name, "d"), log_level="WARNING")
    print(f"config: fit_chars={settings.fit_chars} turn_marks={settings.turn_marks} window={settings.context_window} embed={settings.embed_model!r} head={settings.dense_head} "
          f"planner={settings.planner_model!r} rerank={settings.rerank_model!r}/{settings.rerank_doc} options={settings.options_mode} answer={MODEL}", flush=True)
    client = TestClient(create_app(settings=settings, store=MemoryStore(settings.data_dir))); client.__enter__()
    run = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    user, session = f"eval:{run}:scriptmem:conv-0", f"eval:{run}:sample:0"
    lines: dict[str, str] = {}; msgs = []
    for s in play["sessions"]:
        for e in s["entries"]:
            content = f"{e['speaker']}: {e['text']}" if e["type"] == "dialogue" and e["speaker"] else e["text"]
            lines[e["dia_id"]] = norm(content); msgs.append({"role": "user", "content": content})
    t0 = time.time()
    for n in range(0, len(msgs), 20):
        r = client.post("/v1/memories/add", json={"request_id": f"eval:{run}:scriptmem:conv-0:chunk-{n // 20}", "user_id": user, "session_id": session, "messages": msgs[n:n + 20]})
        assert r.status_code == 200, r.text
    print(f"added {len(msgs)} turns ({sum(len(m['content']) for m in msgs)} chars) in {time.time() - t0:.0f}s", flush=True)

    usage = {"prompt_tokens": 0, "completion_tokens": 0}; rows = []
    for q in questions:
        t0 = time.time()
        data = client.post("/v1/memories/search", json={"query": q["question"], "options": q["option"], "user_id": user, "top_k": 100}).json()["data"]
        latency = time.time() - t0
        if args.evidence_only:
            data = [d for d in data if any(lines[e] in norm(d["content"]) for e in q["evidence"])]
        if args.chronological:      # the same blocks, re-ordered by where their first line stands in the play
            order = list(lines.values())
            def place(d):
                b = norm(d["content"])
                return next((i for i, ln in enumerate(order) if ln in b), len(order))
            data = sorted(data, key=place)
        blocks = [norm(d["content"]) for d in data]
        joined = "\n".join(blocks)
        ev_rank = {e: next((i + 1 for i, b in enumerate(blocks) if lines[e] in b), None) for e in q["evidence"]}
        found = [r for r in ev_rank.values() if r]
        covered = sum(1 for v in lines.values() if v in joined) / len(lines)
        mem = "\n".join(f"- [{d['created_at']}] {d['content']}" if d.get("created_at") else f"- {d['content']}" for d in data)
        row = {"qa_id": q["qa_id"], "qa_type": q["qa_type"], "gold": q["answer_letters"], "results": len(data), "chars": len(mem),
               "evidence_recall": len(found) / len(ev_rank), "evidence_all": len(found) == len(ev_rank), "first_evidence_rank": min(found) if found else None,
               "evidence_all_by_rank": max(found) if len(found) == len(ev_rank) else None, "corpus_coverage": round(covered, 3), "latency_s": round(latency, 2)}
        if not args.no_answer:
            prompt = (official.CHOICE_ANSWER_TEMPLATE.replace("{{speaker_1_name}}", "the people in the conversation").replace("{{speaker_1_memories}}", mem)
                      .replace("{{speaker_2_name}}", "(no second speaker)").replace("{{speaker_2_memories}}", "").replace("{{question}}", q["question"]))
            pred, u = chat(prompt)
            usage["prompt_tokens"] += u.get("prompt_tokens", 0); usage["completion_tokens"] += u.get("completion_tokens", 0)
            letters, malformed = official.predicted_letters(pred, q["qa_type"])
            row.update({"prediction": pred, "predicted": letters, "score": official.score_item(q["qa_type"], q["answer_letters"], letters, malformed)})
        rows.append(row)
        print(f"{q['qa_id'][-5:]} {q['qa_type']:<13} ev {row['evidence_recall']:.2f} first@{row['first_evidence_rank']!s:<4} n={len(data):3d} cover {covered:.2f} "
              + (f"gold {''.join(q['answer_letters']):<5} pred {''.join(row['predicted']):<5} {'OK' if row['score'] else '--'}" if not args.no_answer else ""), flush=True)
    out = Path(args.out_dir) / f"proxy-script-{args.label}-{run}.json"
    out.write_text(json.dumps({"label": args.label, "settings": {"fit_chars": settings.fit_chars, "context_window": settings.context_window, "rerank": settings.rerank_model, "embed": settings.embed_model, "planner": settings.planner_model}, "rows": rows, "usage": usage}, indent=1, ensure_ascii=False))
    g = defaultdict(list)
    for r in rows: g[r["qa_type"]].append(r)
    g["ALL"] = rows
    print(f"\n### proxy script · label={args.label} · questions={len(rows)}")
    print("| type | n | accuracy | evidence recall | all evidence in list | first evidence rank (median) | corpus coverage | results | chars | p50 latency |"); print("|---" * 10 + "|")
    for t in ["single_choice", "multi_select", "ordering", "ALL"]:
        x = g.get(t, [])
        if not x: continue
        acc = f"{sum(r['score'] for r in x) / len(x):.3f}" if not args.no_answer else "-"
        fr = [r["first_evidence_rank"] for r in x if r["first_evidence_rank"]]
        print(f"| {t} | {len(x)} | {acc} | {statistics.mean(r['evidence_recall'] for r in x):.3f} | {sum(r['evidence_all'] for r in x) / len(x):.3f} | {statistics.median(fr) if fr else '-'} | "
              f"{statistics.mean(r['corpus_coverage'] for r in x):.2f} | {statistics.mean(r['results'] for r in x):.0f} | {statistics.mean(r['chars'] for r in x):.0f} | {statistics.median(r['latency_s'] for r in x):.2f}s |")
    cost = usage["prompt_tokens"] * 0.15 / 1e6 + usage["completion_tokens"] * 0.60 / 1e6
    print(f"\nanswer tokens: {usage['prompt_tokens']} in / {usage['completion_tokens']} out · ≈ ${cost:.2f}"); print(f"wrote {out}")
    client.__exit__(None, None, None); tmp.cleanup()


if __name__ == "__main__":
    main()
