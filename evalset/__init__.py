"""Local mini evaluation set for aml-memory.

Everything in here is our own writing (fictional people, fictional companies). It is NOT
leaderboard data and must never be replaced by leaderboard data (competition rule: do not
analyse or train on evaluation data).

Layout:
- schema.py   dataclasses + builder that turns a sample into Add payloads the way the
              official adapter does (chunks of 20 messages / 2,000 words, per session).
- samples/    one module per sample; each exports SAMPLE.
- scripts/eval_local.py (repo root) runs the set against the service and prints a
  per-capability table.
"""
