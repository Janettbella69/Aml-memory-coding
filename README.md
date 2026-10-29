# aml-memory

A small retrieval-only memory service for the [Agent Memory Leaderboard](https://agentmemoryleaderboard.ai/) Add/Search contract (Cycle 2, textual track, open-source division).

Raw conversation turns go into one SQLite file per user, indexed with FTS5 (porter-stemmed words, plus character trigrams for CJK text) and embedded with `text-embedding-v4` (256 dims). Search ranks turns with BM25, fuses the dense neighbours behind the first five lexical hits, and folds in up to three extra queries and a date range proposed by one `gpt-4o-mini` call; every hit comes back with the two turns around it, rendered with UTC timestamps and roles. Near-equal hits are ordered newest-first so an overwritten value never outranks its replacement, and the user's standing rules and preferences ride along. Nothing is generated: every result is stored conversation text. Without the two API keys (`OPENAI_API_KEY`, `DASHSCOPE_API_KEY`) the same code runs fully deterministic, BM25 only.

Submitted version: 1.0.0 (git tag `v1.0.0`) with the settings listed under 1.0 in `docs/METHOD_CARD.md`, which also records the measurement behind each choice.

See `AGENTS.md` for the contract and rules, `PLAN.md` for the roadmap and status, `docs/METHOD_CARD.md` for attribution and the change log, `evalset/README.md` for the local evaluation set.

```bash
uv sync && uv run pytest -q
uv run python scripts/eval_local.py --label dev -v      # local mini-set, per-capability table
uv run uvicorn --factory app.main:create_app --port 8000
uv run python scripts/smoke_test.py http://127.0.0.1:8000
```

Docker: `cp .env.example .env`, set `MEMORY_API_KEY` (plus `OPENAI_API_KEY` and `DASHSCOPE_API_KEY` for the submitted configuration), then `docker compose up -d --build` (binds 127.0.0.1:8000; put a TLS proxy in front). Production stack with Caddy TLS: `deploy/README.md`.

License: MIT. Prior work is credited in `NOTICE` and `docs/METHOD_CARD.md`.
