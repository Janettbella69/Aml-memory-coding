"""Run the local mini evaluation set against the memory service and print a per-capability table.

In-process (default, fresh temp DB, honours CONTEXT_WINDOW / RULE_SLOTS / FALLBACK_RESULTS env):
    uv run python scripts/eval_local.py --label v0.1
Against a running server (local or the public deployment):
    uv run python scripts/eval_local.py --base https://memory.example.com --api-key KEY

Metrics (retrieval only; the answer model is the platform's, so we measure whether the gold evidence is
in what we return, how early, and how much text it costs):
- hit@k   : all (mode=all) / any (mode=any) evidence messages appear within the first k results
- MRR     : 1 / rank of the result that completes the evidence (0 if never completed within top_k)
- order_ok: for questions with traps, the evidence is completed BEFORE the first trap appears
- tokens  : total returned content for the question, chars/4 (budget ~110k; platform truncates after 117,760)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tempfile
import time
import urllib.request
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalset.samples import by_name  # noqa: E402

QID_FILTER: tuple[str, ...] = ()   # --qids
from evalset.schema import CAPABILITIES, Sample  # noqa: E402

KS = (1, 5, 10, 20)
WS_RE = re.compile(r"\s+")


def norm(text: str) -> str:
    return WS_RE.sub(" ", text).strip().lower()


# ------------------------------------------------------------------ clients
def http_client(base: str, api_key: str | None) -> Callable[[str, dict], dict]:
    def call(path: str, payload: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["X-Api-Key"] = api_key
        req = urllib.request.Request(base.rstrip("/") + path, data=json.dumps(payload).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.load(resp)
    return call


def inprocess_client(tmp: str) -> Callable[[str, dict], dict]:
    from fastapi.testclient import TestClient

    from app.config import Settings
    from app.main import create_app
    from app.store import MemoryStore

    settings = replace(Settings.from_env(), api_key="", data_dir=os.path.join(tmp, "d"), log_level="WARNING")
    app = create_app(settings=settings, store=MemoryStore(settings.data_dir))
    for name in ("", "httpx", "aml_memory"):
        logging.getLogger(name).setLevel(logging.WARNING)
    client = TestClient(app)
    client.__enter__()

    def call(path: str, payload: dict) -> dict:
        r = client.post(path, json=payload)
        if r.status_code != 200:
            raise RuntimeError(f"{path} -> {r.status_code}: {r.text[:300]}")
        return r.json()
    return call


# ------------------------------------------------------------------ evaluation
def evaluate_sample(call: Callable[[str, dict], dict], sample: Sample, top_k: int, run: str) -> list[dict]:
    sample.validate()
    for payload in sample.add_payloads(run=run):
        out = call("/v1/memories/add", payload)
        assert out.get("request_id") == payload["request_id"], out
    texts = {k: norm(v) for k, v in sample.texts_by_key().items()}
    all_texts = [norm(t) for t in sample.all_texts()]
    rows: list[dict] = []
    for q in sample.questions:
        if QID_FILTER and not any(q.qid == f or q.qid.startswith(f) for f in QID_FILTER):
            continue
        body = {"query": q.query, "user_id": sample.user_id, "top_k": top_k}
        if q.options:
            body["options"] = list(q.options)
        t0 = time.perf_counter()
        data = call("/v1/memories/search", body)["data"]
        latency_ms = (time.perf_counter() - t0) * 1000
        contents = [norm(str(r.get("content", ""))) for r in data]
        assert len(data) <= top_k, f"{q.qid}: returned {len(data)} > top_k {top_k}"

        def first_rank(key: str) -> int | None:
            needle = texts[key]
            for i, c in enumerate(contents, 1):
                if needle in c:
                    return i
            return None

        ev_ranks = {k: first_rank(k) for k in q.evidence}
        found = [r for r in ev_ranks.values() if r is not None]
        if not q.evidence:
            rank = None
        elif q.mode == "all":
            rank = max(found) if len(found) == len(q.evidence) else None
        else:
            rank = min(found) if found else None
        trap_ranks = {k: first_rank(k) for k in q.trap}
        trap_first = min((r for r in trap_ranks.values() if r is not None), default=None)
        order_ok = None
        if q.trap:
            order_ok = rank is not None and (trap_first is None or rank < trap_first)
        chars = sum(len(c) for c in contents)
        covered = sum(1 for t in all_texts if any(t in c for c in contents))
        rows.append({
            "sample": sample.name, "qid": q.qid, "capability": q.top_capability, "sub": q.capability.split("/", 1)[-1],
            "abstain": not q.evidence, "mode": q.mode, "rank": rank, "evidence_ranks": ev_ranks,
            "trap_ranks": trap_ranks, "order_ok": order_ok, "results": len(data), "chars": chars,
            "tokens_est": chars // 4, "covered_msgs": covered, "total_msgs": len(all_texts), "latency_ms": round(latency_ms, 1),
            **{f"hit@{k}": (rank is not None and rank <= k) for k in KS},
            f"hit@{top_k}": rank is not None,
        })
    return rows


def summarise(rows: list[dict], top_k: int) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["capability"]].append(r)
    groups["ALL"] = rows

    def agg(items: list[dict]) -> dict:
        scored = [r for r in items if not r["abstain"]]
        trapped = [r for r in items if r["order_ok"] is not None]
        out = {"n": len(items), "n_scored": len(scored)}
        for k in (*KS, top_k):
            out[f"hit@{k}"] = round(sum(1 for r in scored if r[f"hit@{k}"]) / len(scored), 3) if scored else None
        out["mrr"] = round(sum(1 / r["rank"] for r in scored if r["rank"]) / len(scored), 3) if scored else None
        out["order_ok"] = round(sum(1 for r in trapped if r["order_ok"]) / len(trapped), 3) if trapped else None
        out["n_trap"] = len(trapped)
        out["tokens_avg"] = int(sum(r["tokens_est"] for r in items) / len(items)) if items else 0
        out["tokens_max"] = max((r["tokens_est"] for r in items), default=0)
        out["latency_p50_ms"] = sorted(r["latency_ms"] for r in items)[len(items) // 2] if items else 0
        return out

    ordered = [c for c in CAPABILITIES if c in groups] + ["ALL"]
    return {c: agg(groups[c]) for c in ordered}


def print_table(summary: dict, top_k: int) -> None:
    cols = ["n", "hit@1", "hit@5", "hit@10", "hit@20", f"hit@{top_k}", "mrr", "order_ok", "n_trap", "tokens_avg", "tokens_max"]
    head = "| capability | " + " | ".join(cols) + " |"
    print(head)
    print("|" + "---|" * (len(cols) + 1))
    for cap, s in summary.items():
        cells = []
        for c in cols:
            v = s[c]
            cells.append("-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v)))
        print(f"| {cap} | " + " | ".join(cells) + " |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", help="server base URL; default runs the app in-process with a temp DB")
    ap.add_argument("--api-key", default=os.environ.get("MEMORY_API_KEY") or None)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--samples", help="comma-separated sample names (default: all)")
    ap.add_argument("--label", default="dev", help="name of this configuration, stored in the result file")
    ap.add_argument("--json", help="write per-question rows + summary to this path (default evalset/results/<label>-<ts>.json)")
    ap.add_argument("--no-json", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true", help="print every question that missed hit@10 or failed order_ok")
    ap.add_argument("--qids", help="comma-separated question ids or id prefixes to run (default: all)")
    args = ap.parse_args()
    global QID_FILTER
    QID_FILTER = tuple(x.strip() for x in args.qids.split(",") if x.strip()) if args.qids else ()

    samples = by_name(args.samples.split(",") if args.samples else None)
    run = "local" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")   # fresh request_ids/user_ids per run
    for s in samples:
        s.validate()

    tmp = None
    if args.base:
        call = http_client(args.base, args.api_key)
    else:
        tmp = tempfile.TemporaryDirectory()
        call = inprocess_client(tmp.name)

    rows: list[dict] = []
    for s in samples:
        # user_id must be unique per run on a shared server, otherwise old chunks pollute the search
        s_run = Sample(name=f"{s.name}-{run}" if args.base else s.name, style=s.style, sessions=s.sessions,
                       questions=s.questions, speakers=s.speakers, description=s.description)
        rows.extend(evaluate_sample(call, s_run, args.top_k, run))
    summary = summarise(rows, args.top_k)

    print(f"\n### eval_local · label={args.label} · top_k={args.top_k} · samples={[s.name for s in samples]} · "
          f"questions={len(rows)} · {'base=' + args.base if args.base else 'in-process'}")
    print_table(summary, args.top_k)

    if args.verbose:
        print("\nMisses (no hit@10 or order_ok=False):")
        for r in rows:
            if r["abstain"]:
                continue
            if not r["hit@10"] or r["order_ok"] is False:
                print(f"- {r['qid']} [{r['capability']}/{r['sub']}] rank={r['rank']} evidence={r['evidence_ranks']} "
                      f"traps={r['trap_ranks']} order_ok={r['order_ok']}")

    if not args.no_json:
        out = Path(args.json) if args.json else Path("evalset/results") / f"{args.label}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"label": args.label, "top_k": args.top_k, "base": args.base, "summary": summary, "rows": rows},
                                  indent=1, ensure_ascii=False))
        print(f"\nwrote {out}")
    if tmp:
        tmp.cleanup()


if __name__ == "__main__":
    main()
