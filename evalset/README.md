# Local mini evaluation set

Our own writing, fictional people and companies. It is **not** leaderboard data and must never be
replaced by leaderboard data (rule: evaluation data may not be analysed or trained on).

Set v2 (2026-10-08): 84 questions; v1 (2026-10-07) had 76. Added: relative-time questions (EN `mia-te-05`, ZH `zh-te-02/03`) and multiple-choice variants whose distractors appear elsewhere in the same conversation (`mia-fr-05/06`, `mia-go-05`, `lt-fr-05`, `dv-fr-03`).

| sample | style | sessions | turns | questions | what it stresses |
|---|---|---|---|---|---|
| `mia` | user ↔ assistant, timestamps | 10 (Mar–Sep 2025) | 96 | 29 | rules, contract values that change, pet costs across sessions, stance change, deletion |
| `lintao` | two friends, `Name:`-prefixed turns mapped to user/assistant | 11 (Feb–Oct 2025) | 100 | 20 | speaker attribution, a launch date that moves three times, multi-hop, career timeline |
| `devops` | developer ↔ coding assistant | 8 (May–Sep 2025) | 60 | 16 | standing rules, workflow edits, a rule replaced by another, empirical numbers |
| `nostamp` | user ↔ assistant, **no timestamps** | 3 | 20 | 6 | ordering by Add ordinal only, dates inside the text |
| `xiaoyu` | Chinese, user ↔ assistant | 3 | 18 | 8 | the CJK trigram index and Chinese rule tagging |

Capabilities follow the leaderboard's seven groups: fact_recall, composition, temporal, governance,
personalization, execution, safety. Each question carries `evidence` (message keys that must be
retrieved), `mode` (`any` or `all` for multi-hop), optional `trap` keys (stale values that must rank
*below* the evidence) and optional multiple-choice `options`. Safety questions have no evidence; only
their token cost is reported.

## Running

```bash
uv run python scripts/eval_local.py --label v0.2 -v          # in-process, temp DB, honours CONTEXT_WINDOW etc.
uv run python scripts/eval_local.py --base https://memory.example.com --api-key KEY
uv run python scripts/sweep_local.py RECENCY_BAND=0,0.2 CONTEXT_WINDOW=1,2   # grid over env settings
```

Metrics are retrieval-only (the answer model belongs to the platform): hit@k, MRR of the result that
completes the evidence, `order_ok` for trap questions, and the estimated token cost of the response
(the platform keeps a 117,760-token prefix in rank order). Results go to `evalset/results/` (git-ignored).

## Adding a question

1. Give the evidence turn(s) a key: `("user", "text", "my-key")` in the sample's session list.
2. Add `Q("<sample>-<cap>-NN", "<capability>/<sub>", "question", ["my-key"], ...)`.
3. `uv run pytest tests/test_evalset.py` checks keys resolve and chunks respect the adapter limits.
