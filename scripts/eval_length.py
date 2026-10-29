"""How much should one Search return? Response-length arms on public data, scored with the organisers' own prompts.

One Search per question (the configured service, top_k 100, the full MAX_RESPONSE_CHARS budget); every arm then hands
the answer model a PREFIX of that same list, cut the way the service cuts (whole blocks, stop once the running total
reaches the cap). Arms are therefore paired: they differ only in how much of the ranked list the reader sees.

Answer and judge prompts are read (parsed with `ast`, never imported or executed) from a local clone of the organisers'
pipelines, github.com/AML-memory/agent-memory-leaderboard: data/longmemeval-s/pipeline.py holds OPEN_ENDED_ANSWER_TEMPLATE
and ACCURACY_PROMPT, which data/locomo-refined/pipeline.py repeats word for word. The prompt texts are not copied into
this repository.

    git clone --depth 1 https://github.com/AML-memory/agent-memory-leaderboard /tmp/aml-official
    AML_OFFICIAL=/tmp/aml-official env -u OPENAI_API_KEY uv run --env-file .env python scripts/eval_length.py \\
        --dataset locomo --data /path/to/LoCoMo_refined/data/public --arms 20000,40000,60000,0 --label lexical-planner
    ... --dataset lme --data /path/to/longmemeval_s_cleaned.json --per-type 10

Datasets (public, kept OUTSIDE the repository; FAQ Q9: pre-checks only, never fitted to questions):
  locomo  LoCoMo-Refined public split (mem-eval-suite/LoCoMo_refined, CC BY-NC 4.0); Adds as in scripts/eval_public.py.
  lme     LongMemEval-S cleaned (huggingface.co/datasets/xiaowu0162/longmemeval-cleaned); one user per question, each
          haystack session Added in chunks of at most 20 messages or 2,000 words, turn time = session time + 1 min/turn;
          the reader's question carries its date ("Today is …"), as LongMemEval's own reader prompt gives the date.
Sampling is deterministic: per category, evenly spaced questions in file order (no look at answers or difficulty).
Arm 0 = everything the Search returned. Answer model and judge default to gpt-4o-mini and Qwen3-14B (the organisers'
judge) through the OPENAI_BASE_URL gateway; --max-usd stops the run before the estimate passes the limit.
"""
from __future__ import annotations

import argparse
import ast
import http.client
import json
import os
import re
import statistics
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

WS = re.compile(r"\s+")
PRICES = {  # USD per million tokens (input, output); a cost guard, not an invoice
    "openai/gpt-4o-mini": (0.15, 0.60), "gpt-4o-mini": (0.15, 0.60),
    "qwen/qwen3-14b": (0.06, 0.24),
}


def norm(s: str) -> str:
    return WS.sub(" ", s.strip().lower())


# ---------- the organisers' prompts ----------

def official_prompts(root: Path) -> tuple[str, str]:
    tree = ast.parse((root / "data/longmemeval-s/pipeline.py").read_text(encoding="utf-8"))
    found: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("OPEN_ENDED_ANSWER_TEMPLATE", "ACCURACY_PROMPT"):
                    found[target.id] = node.value.value
    if len(found) != 2:
        sys.exit(f"could not find both prompts in {root}/data/longmemeval-s/pipeline.py")
    return found["OPEN_ENDED_ANSWER_TEMPLATE"], found["ACCURACY_PROMPT"]


def render_answer(template: str, memories: str, question: str) -> str:
    values = {"speaker_1_name": "speaker 1", "speaker_1_memories": memories, "speaker_2_name": "speaker 2",
              "speaker_2_memories": "", "question": question}
    return re.sub(r"\{\{(speaker_1_name|speaker_1_memories|speaker_2_name|speaker_2_memories|question)\}\}",
                  lambda m: values[m.group(1)], template)


def render_judge(template: str, question: str, gold: str, generated: str) -> str:
    values = {"question": question, "gold_answer": gold, "generated_answer": generated}
    return re.sub(r"\{(question|gold_answer|generated_answer)\}", lambda m: values[m.group(1)], template)


def judge_label(text: str) -> str | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    for m in re.finditer(r"\{[^{}]*\}", text, re.DOTALL):
        try:
            label = str(json.loads(m.group(0)).get("label", "")).upper()
        except (json.JSONDecodeError, AttributeError):
            continue
        if label in ("CORRECT", "WRONG"):
            return label
    tail = text.strip().upper()[-40:]
    if "CORRECT" in tail and "WRONG" not in tail:
        return "CORRECT"
    if "WRONG" in tail and "CORRECT" not in tail:
        return "WRONG"
    return None


# ---------- chat gateway ----------

class Gateway:
    def __init__(self, key: str, base: str, max_usd: float):
        self.key, self.base, self.max_usd = key, base.rstrip("/"), max_usd
        self.lock = threading.Lock()
        self.usd = 0.0
        self.tokens = defaultdict(lambda: [0, 0])

    def chat(self, model: str, prompt: str, max_tokens: int, extra: dict | None = None) -> str:
        with self.lock:
            if self.usd >= self.max_usd:
                raise RuntimeError(f"cost guard: estimated ${self.usd:.2f} reached --max-usd {self.max_usd}")
        body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
                "max_tokens": max_tokens, **(extra or {})}
        req = urllib.request.Request(self.base + "/chat/completions", data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    out = json.load(resp)
                break
            except urllib.error.HTTPError as err:
                if err.code in (408, 429, 500, 502, 503, 504, 524) and attempt < 5:
                    time.sleep(2 ** attempt)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, http.client.HTTPException):
                if attempt < 5:
                    time.sleep(2 ** attempt)
                    continue
                raise
        usage = out.get("usage") or {}
        p_in, p_out = PRICES.get(model, (0.5, 1.5))
        with self.lock:
            t = self.tokens[model]
            t[0] += usage.get("prompt_tokens", 0)
            t[1] += usage.get("completion_tokens", 0)
            cost = usage.get("cost")   # OpenRouter reports the charged amount; otherwise estimate from list prices
            self.usd += float(cost) if isinstance(cost, (int, float)) else (
                usage.get("prompt_tokens", 0) * p_in / 1e6 + usage.get("completion_tokens", 0) * p_out / 1e6)
        return str(out["choices"][0]["message"].get("content") or "").strip()


# ---------- datasets ----------

def spaced(items: list, n: int) -> list:
    if n <= 0 or len(items) <= n:
        return list(items)
    return [items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)] if n > 1 else [items[0]]


def locomo_units(data: Path, per_conv: dict[str, int], run: str):
    """One unit = one conversation: (user_id, Add payloads, questions)."""
    from scripts.eval_public import CATEGORY, add_payloads, load
    convs, by_sample = load(data)
    for conv in convs:
        user = f"public:{run}:locomo_refined:{conv['sample_id']}"
        qs_by_cat: dict[str, list[dict]] = defaultdict(list)
        for q in by_sample.get(conv["sample_id"], []):
            qs_by_cat[CATEGORY.get(str(q["category"]), str(q["category"]))].append(q)
        picked = []
        for cat, n in per_conv.items():
            for q in spaced(qs_by_cat.get(cat, []), n):
                picked.append({"id": q["qa_id"], "category": cat, "question": q["question"], "search": q["question"],
                               "gold": " | ".join(str(a) for a in q.get("answer", [])),
                               "evidence": [norm(e.get("text", "")) for e in q.get("evidence_messages", []) if e.get("text")]})
        yield user, list(add_payloads(conv, run)), picked


def iter_json_array(path: Path):
    raw = path.read_text(encoding="utf-8")
    dec = json.JSONDecoder()
    i = raw.index("[") + 1
    while True:
        while i < len(raw) and raw[i] in " \n\r\t,":
            i += 1
        if i >= len(raw) or raw[i] == "]":
            return
        obj, i = dec.raw_decode(raw, i)
        yield obj


def lme_when(s: str) -> int | None:
    try:
        return int(datetime.strptime(s.strip(), "%Y/%m/%d (%a) %H:%M").replace(tzinfo=timezone.utc).timestamp() * 1000)
    except ValueError:
        return None


def lme_units(data: Path, per_type: int, run: str):
    items = list(iter_json_array(data))
    by_type: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        by_type[it["question_type"]].append(it)
    for qtype in sorted(by_type):
        for it in spaced(by_type[qtype], per_type):
            user = f"public:{run}:lme:{it['question_id']}"
            payloads, evidence, n = [], [], 0
            for sid, when_s, turns in zip(it["haystack_session_ids"], it["haystack_dates"], it["haystack_sessions"]):
                when = lme_when(when_s)
                msgs = []
                for i, t in enumerate(turns):
                    content = str(t.get("content") or "").strip()
                    if not content:
                        continue
                    msgs.append({"role": t.get("role") or "user", "content": content,
                                 **({"timestamp": when + i * 60_000} if when is not None else {})})
                    if t.get("has_answer"):
                        evidence.append(norm(content)[:300])
                chunk: list[dict] = []
                words = 0
                for m in msgs:
                    w = len(m["content"].split())
                    if chunk and (len(chunk) >= 20 or words + w > 2000):
                        payloads.append({"request_id": f"{user}:chunk-{n}", "user_id": user,
                                         "session_id": f"{user}:session:{sid}", "messages": chunk})
                        n += 1
                        chunk, words = [], 0
                    chunk.append(m)
                    words += w
                if chunk:
                    payloads.append({"request_id": f"{user}:chunk-{n}", "user_id": user,
                                     "session_id": f"{user}:session:{sid}", "messages": chunk})
                    n += 1
            category = qtype + ("+abs" if it["question_id"].endswith("_abs") else "")
            question = f"{it['question']} (Today is {it['question_date']}.)"
            yield user, payloads, [{"id": it["question_id"], "category": category, "question": question,
                                    "search": it["question"], "gold": str(it["answer"]), "evidence": evidence}]


# ---------- arms ----------

def prefix(contents: list[str], cap: int) -> list[str]:
    """The service's own cut: keep adding whole blocks while the running total is below the cap (0 = no cap)."""
    if cap <= 0:
        return contents
    out, used = [], 0
    for c in contents:
        if used >= cap:
            break
        out.append(c)
        used += len(c)
    return out


def first_evidence_rank(contents: list[str], evidence: list[str]) -> int | None:
    if not evidence:
        return None
    normed = [norm(c) for c in contents]
    ranks = [next((i for i, c in enumerate(normed, 1) if e and e in c), None) for e in evidence]
    found = [r for r in ranks if r is not None]
    return min(found) if found else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=("locomo", "lme"), required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--arms", default="20000,40000,60000,0", help="comma-separated character caps; 0 = whole list")
    ap.add_argument("--label", default="dev")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--per-type", type=int, default=10, help="lme: questions per question type")
    ap.add_argument("--locomo-mix", default="single_hop:5,multi_hop:2,temporal:2,open_domain:1",
                    help="locomo: questions per conversation per category")
    ap.add_argument("--answer-model", default=os.environ.get("ANSWER_MODEL", "openai/gpt-4o-mini"))
    ap.add_argument("--judge-model", default=os.environ.get("OFFICIAL_JUDGE_MODEL", "qwen/qwen3-14b"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-usd", type=float, default=3.0)
    ap.add_argument("--retrieval-only", action="store_true", help="no answer/judge calls (list statistics only)")
    ap.add_argument("--units", type=int, help="stop after this many conversations (locomo) / questions (lme)")
    args = ap.parse_args()

    answer_tpl, judge_tpl = official_prompts(Path(os.environ.get("AML_OFFICIAL") or sys.exit("set AML_OFFICIAL")))
    arms = [int(x) for x in args.arms.split(",")]
    gw = None
    if not args.retrieval_only:
        key = os.environ.get("OPENAI_API_KEY") or sys.exit("OPENAI_API_KEY is not set")
        gw = Gateway(key, os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"), args.max_usd)

    from fastapi.testclient import TestClient

    from app.config import Settings
    from app.main import create_app
    from app.store import MemoryStore

    tmp = tempfile.TemporaryDirectory()
    settings = replace(Settings.from_env(), api_key="", data_dir=os.path.join(tmp.name, "d"), log_level="WARNING")
    print(f"service: planner={settings.planner_model or 'off'} embed={getattr(settings, 'embed_model', '') or 'off'} "
          f"rerank={settings.rerank_model or 'off'} max_response_chars={settings.max_response_chars} arms={arms}", flush=True)
    app = create_app(settings=settings, store=MemoryStore(settings.data_dir))
    client = TestClient(app)
    client.__enter__()
    run = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

    if args.dataset == "locomo":
        mix = {k: int(v) for k, v in (p.split(":") for p in args.locomo_mix.split(","))}
        units = locomo_units(Path(args.data), mix, run)
    else:
        units = lme_units(Path(args.data), args.per_type, run)

    def answer_and_judge(q: dict, contents: list[str], cap: int) -> dict:
        ctx = prefix(contents, cap)
        memories = "\n".join(ctx)
        cell = {"blocks": len(ctx), "chars": len(memories), "evidence_inside": None}
        if q["evidence"]:
            cell["evidence_inside"] = first_evidence_rank(ctx, q["evidence"]) is not None
        if gw is None:
            return cell
        try:
            generated = gw.chat(args.answer_model, render_answer(answer_tpl, memories, q["question"]), 256)
            cell["answer"] = generated
            label = None
            for _ in range(3):
                label = judge_label(gw.chat(args.judge_model, render_judge(judge_tpl, q["question"], q["gold"], generated), 1500))
                if label:
                    break
            cell["correct"] = None if label is None else label == "CORRECT"
        except Exception as exc:  # noqa: BLE001 - one failed call must not end a paid run
            cell["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
            cell["correct"] = None
        return cell

    rows: list[dict] = []
    t_start = time.time()
    pool = ThreadPoolExecutor(max_workers=args.workers)
    try:
        for n_unit, (user, payloads, questions) in enumerate(units):
            if args.units is not None and n_unit >= args.units:
                break
            if not questions:
                continue
            t0 = time.time()
            for p in payloads:
                r = client.post("/v1/memories/add", json=p)
                assert r.status_code == 200, r.text
            add_s = time.time() - t0
            for q in questions:
                t1 = time.time()
                data = client.post("/v1/memories/search",
                                   json={"query": q["search"], "user_id": user, "top_k": args.top_k}).json()["data"]
                contents = [str(d.get("content", "")) for d in data]
                row = {"id": q["id"], "category": q["category"], "results": len(contents),
                       "chars": sum(len(c) for c in contents), "search_s": round(time.time() - t1, 2),
                       "first_rank": first_evidence_rank(contents, q["evidence"]), "arms": {}}
                futures = {cap: pool.submit(answer_and_judge, q, contents, cap) for cap in arms}
                row["_futures"] = futures
                rows.append(row)
            spent = f", ≈ ${gw.usd:.2f}" if gw else ""
            print(f"{user.rsplit(':', 1)[-1]}: add {add_s:.1f}s ({len(payloads)} chunks), {len(questions)} q, "
                  f"elapsed {time.time() - t_start:.0f}s{spent}", flush=True)
    finally:
        for row in rows:
            for cap, fut in row.pop("_futures", {}).items():
                try:
                    row["arms"][str(cap)] = fut.result()
                except Exception as exc:  # noqa: BLE001
                    row["arms"][str(cap)] = {"error": str(exc)[:160], "correct": None}
        pool.shutdown()
        client.__exit__(None, None, None)
        tmp.cleanup()

    # ---------- report ----------
    ref = str(arms[-1])
    cats = sorted({r["category"] for r in rows})
    print(f"\n### eval_length · {args.dataset} · label={args.label} · questions={len(rows)} · reference arm={ref}")
    print(f"list size: results median {statistics.median(r['results'] for r in rows):.0f}, "
          f"chars median {statistics.median(r['chars'] for r in rows):.0f}, max {max(r['chars'] for r in rows)}")
    # paired: only questions every arm got a verdict for (a gateway refusal, e.g. an HTTP 403 moderation block on a
    # long context, removes the question from all arms instead of biasing one)
    common = [r for r in rows if gw is None or all(r["arms"][str(a)].get("correct") is not None for a in arms)]
    if gw is not None:
        print(f"questions with a verdict in every arm: {len(common)} of {len(rows)}")
    print("| category | n | " + " | ".join(f"cap {a or 'all'}" for a in arms) + " |")
    print("|---" * (2 + len(arms)) + "|")
    for cat in cats + ["ALL"]:
        g = [r for r in common if cat == "ALL" or r["category"] == cat]
        if not g:
            continue
        cells = []
        for a in arms:
            vals = [r["arms"][str(a)].get("correct") for r in g]
            vals = [v for v in vals if v is not None]
            ev = [r["arms"][str(a)].get("evidence_inside") for r in g if r["arms"][str(a)].get("evidence_inside") is not None]
            acc = f"{sum(vals) / len(vals):.3f}" if vals else "-"
            cells.append(f"{acc} (ev {sum(ev) / len(ev):.2f})" if ev else acc)
        print(f"| {cat} | {len(g)} | " + " | ".join(cells) + " |")
    if gw is not None:
        print(f"\npaired vs cap {ref or 'all'} (wins/losses over questions judged in both arms):")
        for a in arms[:-1]:
            w = l = 0
            for r in rows:
                x, y = r["arms"][str(a)].get("correct"), r["arms"][ref].get("correct")
                if x is None or y is None:
                    continue
                w += x and not y
                l += y and not x
            print(f"  cap {a}: +{w} / -{l}")
        mean_chars = {a: statistics.mean(r["arms"][str(a)].get("chars", 0) for r in rows) for a in arms}
        print("mean context chars: " + ", ".join(f"{a or 'all'}={mean_chars[a]:.0f}" for a in arms))
        print("tokens: " + "; ".join(f"{m} {t[0]} in / {t[1]} out" for m, t in gw.tokens.items()) + f" · ≈ ${gw.usd:.2f}")
    out = Path("evalset/results") / f"length-{args.dataset}-{args.label}-{run}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"label": args.label, "dataset": args.dataset, "arms": arms, "answer_model": args.answer_model,
                               "judge_model": args.judge_model, "rows": rows}, indent=1, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
