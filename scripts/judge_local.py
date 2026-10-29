"""End-to-end proxy of the platform's scoring: our Search output -> gpt-4o-mini answers -> gpt-4o-mini judges.

    OPENAI_API_KEY=... uv run python scripts/judge_local.py --label v0.2            # in-process, all samples
    OPENAI_API_KEY=... uv run python scripts/judge_local.py --samples mia --limit 10

The platform's answer model, prompts and scoring are locked and unpublished; this is only a proxy to
compare our own configurations on answer accuracy rather than hit@k. Uses gpt-4o-mini for both roles
(the open-source division's mandated model). Prints per-capability accuracy and token usage/cost.
Never run it on leaderboard data; evalset/ is our own writing.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import tempfile
import time
import urllib.request
from collections import defaultdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalset.samples import by_name  # noqa: E402
from evalset.schema import CAPABILITIES  # noqa: E402

MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")   # e.g. JUDGE_MODEL=openai/gpt-4o-mini on OpenRouter
PRICE_IN, PRICE_OUT = 0.15 / 1e6, 0.60 / 1e6   # USD per token, gpt-4o-mini list price

ANSWER_SYSTEM = (
    "You are answering a question about a user using memories retrieved from past conversations. "
    "Use only the memories. Each memory line starts with a UTC timestamp and the speaker role; later "
    "statements override earlier ones when they conflict. Be concise: give the answer in one sentence "
    "or a single letter for multiple choice. If the memories do not contain the information, reply exactly: "
    "I don't know."
)
JUDGE_SYSTEM = (
    "You grade an answer against a reference answer. Reply with exactly one word: CORRECT if the answer "
    "conveys the same fact(s) as the reference (wording may differ; partial credit is INCORRECT), "
    "ABSTAIN if the answer says it does not know, otherwise INCORRECT."
)


def chat(messages: list[dict], max_tokens: int, key: str) -> tuple[str, dict]:
    body = {"model": MODEL, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"}, method="POST")
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                out = json.load(resp)
            return out["choices"][0]["message"]["content"].strip(), out.get("usage", {})
        except urllib.error.HTTPError as err:
            if err.code in (408, 429, 500, 502, 503, 504, 524) and attempt < 5:
                time.sleep(2 ** attempt)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, http.client.HTTPException):
            # dropped connections / timeouts on the gateway side: retry with backoff (seen 2026-10-09: RemoteDisconnected)
            if attempt < 5:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError("unreachable")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples")
    ap.add_argument("--limit", type=int, help="first N questions per sample")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--label", default="dev")
    ap.add_argument("--max-context-chars", type=int, default=400_000)
    ap.add_argument("--qids", help="comma-separated question ids or id prefixes to run (default: all)")
    args = ap.parse_args()
    qid_filter = tuple(x.strip() for x in args.qids.split(",") if x.strip()) if args.qids else ()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("OPENAI_API_KEY is not set")

    from fastapi.testclient import TestClient

    from app.config import Settings
    from app.main import create_app
    from app.store import MemoryStore

    tmp = tempfile.TemporaryDirectory()
    settings = replace(Settings.from_env(), api_key="", data_dir=os.path.join(tmp.name, "d"), log_level="WARNING")
    app = create_app(settings=settings, store=MemoryStore(settings.data_dir))
    client = TestClient(app)
    client.__enter__()

    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    rows: list[dict] = []
    for sample in by_name(args.samples.split(",") if args.samples else None):
        sample.validate()
        for payload in sample.add_payloads(run="judge"):
            assert client.post("/v1/memories/add", json=payload).status_code == 200
        questions = sample.questions[: args.limit] if args.limit else sample.questions
        if qid_filter:
            questions = [q for q in questions if any(q.qid == f or q.qid.startswith(f) for f in qid_filter)]
        for q in questions:
            body = {"query": q.query, "user_id": sample.user_id, "top_k": args.top_k}
            if q.options:
                body["options"] = list(q.options)
            data = client.post("/v1/memories/search", json=body).json()["data"]
            context = "\n\n".join(f"[memory {i}]\n{d['content']}" for i, d in enumerate(data, 1))[: args.max_context_chars]
            question = q.query + ("\nOptions:\n" + "\n".join(q.options) if q.options else "")
            answer, u1 = chat([{"role": "system", "content": ANSWER_SYSTEM},
                               {"role": "user", "content": f"Memories:\n{context}\n\nQuestion: {question}"}], 120, key)
            reference = q.answer or "(no information in memory; the correct behaviour is to say you don't know)"
            verdict, u2 = chat([{"role": "system", "content": JUDGE_SYSTEM},
                                {"role": "user", "content": f"Question: {q.query}\nReference: {reference}\nAnswer: {answer}"}], 3, key)
            for u in (u1, u2):
                usage["prompt_tokens"] += u.get("prompt_tokens", 0)
                usage["completion_tokens"] += u.get("completion_tokens", 0)
            verdict = verdict.upper().split()[0] if verdict else "INCORRECT"
            expect_abstain = not q.evidence
            correct = (verdict == "ABSTAIN") if expect_abstain else (verdict == "CORRECT")
            rows.append({"sample": sample.name, "qid": q.qid, "capability": q.top_capability, "answer": answer,
                         "reference": reference, "verdict": verdict, "correct": correct, "context_chars": len(context)})
            mark = "ok " if correct else "XX "
            print(f"{mark}{q.qid:<10} {verdict:<9} | {answer[:70]}")

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["capability"]].append(r)
    groups["ALL"] = rows
    print(f"\n### judge_local · label={args.label} · model={MODEL} · questions={len(rows)}")
    print("| capability | n | accuracy | avg context chars |")
    print("|---|---|---|---|")
    for cap in [c for c in CAPABILITIES if c in groups] + ["ALL"]:
        g = groups[cap]
        print(f"| {cap} | {len(g)} | {sum(r['correct'] for r in g) / len(g):.3f} | {int(sum(r['context_chars'] for r in g) / len(g))} |")
    cost = usage["prompt_tokens"] * PRICE_IN + usage["completion_tokens"] * PRICE_OUT
    print(f"\ntokens: {usage['prompt_tokens']} in / {usage['completion_tokens']} out · ≈ ${cost:.3f}")
    out = Path("evalset/results") / f"judge-{args.label}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"label": args.label, "model": MODEL, "usage": usage, "rows": rows}, indent=1, ensure_ascii=False))
    print(f"wrote {out}")
    client.__exit__(None, None, None)
    tmp.cleanup()


if __name__ == "__main__":
    main()
