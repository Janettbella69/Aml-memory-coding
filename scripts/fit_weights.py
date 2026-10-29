"""Fit the leaderboard's Overall score as a weighted mean of the seven capability scores.

    uv run python scripts/fit_weights.py /path/to/leaderboard_data.json

The platform does not publish the aggregation. The official Cycle-1 data file
(huggingface.co/spaces/agent-memory-leaderboard/leaderboard, data/generated/leaderboard_data.json; 50 open-source
and 15 commercial entries) lets us recover it: non-negative weights summing to one, fitted by projected gradient.
On the 50 open-source rows the fit is exact to 0.16 points (leave-one-out 0.14), i.e. Overall is effectively a
question-count-weighted mean. Result (2026-10-09): fact recall 0.46, execution 0.33, personalization 0.09,
composition 0.05, governance 0.03, safety 0.03, temporal 0.02. See docs/CYCLE1_NOTES.md § Score weights.
numpy only; the data file stays outside the repository.
"""
from __future__ import annotations

import json
import sys

import numpy as np

CAPS = ["factRecall", "composition", "temporal", "governance", "personalization", "execution", "safety"]


def simplex_fit(X: np.ndarray, y: np.ndarray, fixed: tuple[int, float] | None = None, iters: int = 100_000) -> np.ndarray:
    """Least squares with w >= 0 and sum(w) = 1 (optionally one weight pinned), by projected gradient."""
    n = X.shape[1]
    w = np.ones(n) / n
    for _ in range(iters):
        w = w - 0.5 * (X.T @ (X @ w - y) / len(y))
        if fixed is None:
            w = _project(w, 1.0)
        else:
            i, val = fixed
            rest = [j for j in range(n) if j != i]
            w[i] = val
            w[rest] = _project(w[rest], 1.0 - val)
    return w


def _project(v: np.ndarray, total: float) -> np.ndarray:
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    k = np.nonzero(u * np.arange(1, len(v) + 1) > (css - total))[0][-1]
    theta = (css[k] - total) / (k + 1)
    return np.maximum(v - theta, 0.0)


def rmse(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    return float(np.sqrt(((X @ w - y) ** 2).mean()))


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    data = json.load(open(sys.argv[1], encoding="utf-8"))
    boards = {"academic": data["overall"]["academic"], "commercial": data["overall"]["commercial"]}
    boards["all"] = boards["academic"] + boards["commercial"]
    for name, rows in boards.items():
        X = np.array([[r[c] for c in CAPS] for r in rows])
        y = np.array([r["overall"] for r in rows])
        w = simplex_fit(X, y)
        loo = []
        for i in range(len(y)):
            m = np.ones(len(y), bool)
            m[i] = False
            loo.append(abs(X[i] @ simplex_fit(X[m], y[m], iters=20_000) - y[i]))
        print(f"\n## {name} (n={len(y)})  rmse {rmse(X, y, w) * 100:.2f} pts · leave-one-out mean {np.mean(loo) * 100:.2f} / max {np.max(loo) * 100:.2f} pts")
        for c, wi in sorted(zip(CAPS, w), key=lambda t: -t[1]):
            print(f"  {c:16s} {wi:.3f}")
        if name == "academic":
            for i, c in ((2, "temporal"), (0, "factRecall")):
                wf = simplex_fit(X, y, fixed=(i, 1 / 7))
                print(f"  (forcing {c} to 1/7 → rmse {rmse(X, y, wf) * 100:.2f} pts)")
    # counterfactual: MemMachine's capability profile with execution kept at the field's 0.30
    acad = {r["system"]: r for r in boards["academic"]}
    if "MemMachine" in acad:
        X = np.array([[r[c] for c in CAPS] for r in boards["academic"]])
        y = np.array([r["overall"] for r in boards["academic"]])
        w = simplex_fit(X, y)
        mm = np.array([acad["MemMachine"][c] for c in CAPS])
        alt = mm.copy()
        alt[CAPS.index("execution")] = 0.30
        print(f"\nMemMachine actual {acad['MemMachine']['overall'] * 100:.1f} · predicted {mm @ w * 100:.1f} · "
              f"with execution held at 30.0 → {alt @ w * 100:.1f} (cycle-1 #1 = {max(r['overall'] for r in boards['academic']) * 100:.1f})")


if __name__ == "__main__":
    main()
