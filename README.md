# aml-memory-coding

The coding-track entry of [aml-memory](https://github.com/Janettbella69/Aml-memory) for the [Agent Memory Leaderboard](https://agentmemoryleaderboard.ai/) Add/Search contract (Cycle 2, Coding Memory track, open-source division). It starts from a byte-identical copy of the textual entry's frozen version (`Janettbella69/Aml-memory@9f5c5bf`, v1.0.0); that entry is registered, deployed and evaluated separately and is not changed by anything in this repository.

A coding memory stores the trajectories of earlier tasks in the same repository — agent transcripts: a task statement, tool calls, tool output, the agent's reasoning and its edits. Raw turns go into one SQLite file per user, indexed with FTS5 (porter-stemmed words; camelCase identifiers also by their inner words; character trigrams for CJK text). Search ranks turns with BM25, groups the hits by past session, ranks the sessions with tool output weighted down, and returns one card per session: the task as it was given, the agent's closing message, then the matching turns, rendered with UTC timestamps and roles. Nothing is generated and no model is called: every result is stored transcript text, and the whole pipeline is deterministic.

Version: coding-0.1.0 (git tag `coding-0.1.0`), configuration in `.env.example`. `docs/METHOD_CARD.md` records attribution and the measurement behind each choice: the organisers' reference catalog for the track, SWEContextBench, run through `scripts/eval_swecb.py` as a retrieval check (the scored suite, CAMBench Coding, is private).

See `AGENTS.md` for the contract and rules, `PLAN.md` for the status board, `docs/SUBMISSION.md` for the form fields, `deploy/tencent/README.md` for the deployment.

```bash
uv sync && uv run pytest -q                              # 116 tests
uv run uvicorn --factory app.main:create_app --port 8000
uv run python scripts/smoke_test.py http://127.0.0.1:8000
uv run python scripts/eval_swecb.py --cases <transcripts dir> --related <tasks dir> --relationship <pairs.json> --label dev
```

Docker: `cp .env.example .env`, set `MEMORY_API_KEY`, then `docker compose up -d --build` (binds 127.0.0.1:8000; put a TLS proxy in front). Behind an existing nginx on a shared box: `deploy/tencent/`. Stand-alone with Caddy TLS: `deploy/README.md`.

License: MIT. Prior work and evaluation data are credited in `NOTICE` and `docs/METHOD_CARD.md`.
