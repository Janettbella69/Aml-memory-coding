"""Parameter sweep over the local eval set (in-process). Each configuration is one subprocess so the
Settings.from_env() values apply. Prints one line per configuration.

    uv run python scripts/sweep_local.py RECENCY_BAND=0,0.1,0.2,0.3 LEAD_HITS=3,5 RULE_SLOTS=4,8 CONTEXT_WINDOW=1,2
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    grid: dict[str, list[str]] = {}
    for arg in sys.argv[1:]:
        name, _, values = arg.partition("=")
        grid[name] = values.split(",")
    names = list(grid)
    print("| " + " | ".join(names) + " | hit@1 | hit@5 | hit@10 | hit@100 | mrr | order_ok | tokens_avg | tokens_max |")
    print("|" + "---|" * (len(names) + 8))
    for combo in itertools.product(*(grid[n] for n in names)):
        env = dict(os.environ, **dict(zip(names, combo)), LOG_LEVEL="WARNING")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            out = tmp.name
        subprocess.run([sys.executable, str(ROOT / "scripts" / "eval_local.py"), "--json", out, "--label", "sweep"],
                       env=env, cwd=ROOT, check=True, capture_output=True)
        s = json.loads(Path(out).read_text())["summary"]["ALL"]
        os.unlink(out)
        top = [k for k in s if k.startswith("hit@") and k not in ("hit@1", "hit@5", "hit@10", "hit@20")][0]
        print("| " + " | ".join(combo) + f" | {s['hit@1']:.3f} | {s['hit@5']:.3f} | {s['hit@10']:.3f} | {s[top]:.3f} | {s['mrr']:.3f} | "
              f"{s['order_ok']:.3f} | {s['tokens_avg']} | {s['tokens_max']} |")


if __name__ == "__main__":
    main()
