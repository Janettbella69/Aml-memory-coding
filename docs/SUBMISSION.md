# Submission sheet — AML Cycle 2 (Textual, Open-source Methods)

Fill-in sheet mirroring the official evaluation-access form and the Create Eval Job form (field names checked against agentmemoryleaderboard.ai on 2026-10-07). `TBD` = not decided yet. Keep issued keys and personal contact details out of this file; enter them only in the form.

## Evaluation access request

| Field | Value |
|---|---|
| System name | `aml-memory` |
| Version | TBD — `v1.0` at freeze (code is `0.3.0` today) |
| Contact details | TBD (enter in the form only; do not commit) |
| Organization or team | TBD |
| Intended evaluation type | Textual Memory track · Open-source Methods division |
| Method or product description | Deterministic retrieval service for the AML Add/Search contract. Raw conversation turns are stored in SQLite and indexed with FTS5 (porter-stemmed words, plus character trigrams for CJK text); Search ranks them with BM25 over the question and its options, orders near-equal hits newest-first so overwritten values do not outrank their replacements, places the user's standing rules after the leading hits, expands each hit with the two turns around it, de-duplicates, and renders every turn with its UTC timestamp and role. A dense channel (`text-embedding-v4`, 256 d) is fused by RRF behind the first five lexical hits; one `gpt-4o-mini` call per Search proposes up to three extra queries and a date range (fused the same way); the full top-100 of verbatim blocks is returned. Nothing is generated; every result is conversation text with its timestamp and role. |
| Information approved for public display | System name, version, the method description above, the public repository URL and the fixed commit. |
| Complete submission notes | Deployed API version `1.0.0`; endpoints below; auth `X-Api-Key`; capacity below; operational constraints: single instance, SQLite, no server-side request timeout, Add is synchronous; prior-work attribution and method changes in `docs/METHOD_CARD.md`. |
| Deployed Add endpoint URL | `https://memory.azorashopping.site/v1/memories/add` (Alibaba Cloud box, Caddy TLS; live since 2026-10-07) |
| Deployed Search endpoint URL | `https://memory.azorashopping.site/v1/memories/search` |
| Health URL | `https://memory.azorashopping.site/health` (unauthenticated GET, returns 200; re-checked 2026-10-08) |
| Authentication | `X-Api-Key: <Memory System Key>` (declared scheme). The service also accepts `Authorization: Bearer <key>` and `Authorization: Token <key>`. The scheme chosen in the form must be the one used; a mismatch is a 401. `none` is allowed only for public smoke. |
| Memory System Key | the value of `MEMORY_API_KEY` in the server's `.env` (≥ 16 characters); never committed |
| Capacity declaration | Max add concurrency 16 · Search concurrency 16 · `top_k` 1–1000 (official runs use 100) · Add synchronous, persisted and searchable before the 200 · no server-side timeout (platform limit 30 min/request) · single instance |
| Typical latency (FAQ Q5) | Measured 2026-10-08 through TLS at 16-way concurrency: Add p50 234 ms / p99 454 ms; Search p50 2.2 s / p99 5.0 s with the gpt-4o-mini planner (≈ 40 ms without it) |
| Request body & storage capacity (FAQ Q5) | Add body up to 20 messages / 2,000 words per platform chunk (no explicit body limit enforced below FastAPI defaults); storage ≈ 5× raw text at rest, 34 GB free on the box (≥ 20 GB needed for the Textual track) |
| Models & external dependencies (FAQ Q5/Q7) | `gpt-4o-mini` (via OpenRouter, model id `openai/gpt-4o-mini`) as the Search-time query planner only; no generative model in Add (N/A); `text-embedding-v4` (Alibaba Model Studio, China endpoint, 256 dims) for the dense channel at Add and Search; no reranker (the optional reranker module is off in the submitted configuration); SQLite FTS5; numpy; Caddy TLS |
| Multimodal processing declaration (FAQ Q5) | N/A — Textual track only |
| Service maintenance (FAQ Q5) | Single operator; health monitored via `/health`; Docker `restart: unless-stopped`; no planned changes during a Full run (FAQ Q29); idempotency records and data volume kept until the run is closed (FAQ Q26/Q27), deleted within 30 days after (FAQ Q31) |
| Public repository URL | `https://github.com/Janettbella69/Aml-memory` (the old lowercase URL redirects; private until the release — switch with `gh repo edit Janettbella69/Aml-memory --visibility public --accept-visibility-change-consequences`) |
| Fixed commit hash | the commit tagged `v1.0.0` (full hash: `git rev-parse v1.0.0^{commit}`) |
| Attribution of prior work | `docs/METHOD_CARD.md` § "Prior work and attribution" (FTS5 baseline pattern, idea only — the 0.1 draft's adapted lines were rewritten in 0.2; ReFind: timestamp rendering, context window, ordinal ordering, porter, RRF, service shape — MIT, notice in `NOTICE`) |
| Method changes | `docs/METHOD_CARD.md` § "Method changes by version" |

## Create Eval Job

| Field | Value |
|---|---|
| Evaluation track | Textual |
| System name | `aml-memory` |
| Version name | `v1.0.0` |
| Run label | `aml-memory-v1.0-smoke-01`, `-smoke-02`, … then `aml-memory-v1.0-full-01` (labels must be unique per key) |
| Mode | `smoke` first; `full` once smoke passes and the checklist is confirmed. The form also offers `light` and `scriptmem`; `full` is the only formal participant mode. |
| Max add concurrency | 16 |
| Search concurrency | 16 |

## Quotas and dates (official, 2026-10-07)

- Smoke: 1 per hour; at most 30 per track this edition. Private, never public-eligible.
- Full: at most 2 per AML Key per track; the second unlocks 30 days after the first completes — with evaluation closing on 2026-11-04 this means one Full run in practice. A failed or cancelled Full resumes from its last checkpoint and does not consume a new Full.
- Evaluation keys are issued by 19:00 Beijing time on the day after the request.
- Materials deadline: 2026-10-31 23:59 (UTC+8). Evaluation closes: 2026-11-04 23:59 (UTC+8). Official leaderboard: mid-November 2026.
- Once a Full evaluation is accepted the version cannot be replaced or withdrawn because of an unfavourable result; freeze the version before applying.
- Endpoints need not stay online after the evaluation closes.
- Before every Full run the 8-item gate in `docs/CHECKLIST.md` must be confirmed.

## 提交说明与运行指南（表单文本，冻结时把 <COMMIT> 换成固定 commit）

```
【系统】aml-memory v1.0.0，固定 commit <COMMIT>，公开仓库 https://github.com/Janettbella69/Aml-memory（MIT）。参赛组别：开源方法榜；赛道：文本记忆。

【部署】阿里云国际版单实例（2 vCPU / 1.6 GB / 34 GB 可用），Docker + Caddy 自动 TLS。
Add：https://memory.azorashopping.site/v1/memories/add  Search：https://memory.azorashopping.site/v1/memories/search  Health：https://memory.azorashopping.site/health（无鉴权 GET）。
鉴权：X-Api-Key（同时接受 Authorization: Bearer / Token）。容量与限制：Add 并发 16、Search 并发 16、top_k 1–1000；Add 同步写入，返回 200 即已持久化且可检索；request_id 幂等（同 id 不同内容返回 409），幂等记录持久化在 SQLite，重启不丢；无服务端超时（遵守平台 30 分钟上限）；限流时返回 429 + Retry-After。实测（16 路并发、公网 TLS）：Add p50 234 ms / p99 454 ms，Search p50 2.2 s / p99 5.0 s（含模型调用）。存储约为原文 5 倍。评测数据不记录文本日志、不用于训练，任务结束 30 天内删除。

【方法】每个 user_id 一个独立 SQLite 文件，原始对话逐轮存储，不做摘要。检索为三路召回 + 倒数排名融合（RRF）：(1) FTS5 BM25（Porter 词干）；(2) CJK 字符三元组 BM25；(3) text-embedding-v4（256 维）余弦相似。排序上近分的新值优先（覆盖旧值）；用户的规则/偏好语句单独占位；每条命中附前后各 2 轮上下文并跨 Add 块去重；渲染为「[UTC 时间戳] 角色: 原文」，并对「昨天 / 三周前」等相对时间附上按该轮时间戳换算的绝对日期注释。Search 阶段调用一次 gpt-4o-mini 做查询规划（最多 3 个补充查询 + 可选日期范围），结果同样以 RRF 融合；调用失败则确定性回退。只返回原文证据，不生成、不伪装答案；不跨 user_id 读取。

【模型】Add 阶段不调用生成模型（N/A）。Search 阶段唯一的生成模型是 gpt-4o-mini：我们经 OpenRouter 路由（模型标识 openai/gpt-4o-mini，即 OpenAI 同一模型）；复现时可直接设置 OPENAI_BASE_URL=https://api.openai.com/v1、PLANNER_MODEL=gpt-4o-mini 使用 OpenAI 官方接口，代码无需改动。向量模型为 text-embedding-v4（阿里云百炼，参赛问答第 05 条指定），DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL 可配置。不使用 reranker（代码里保留了可选的重排模块，提交配置下关闭：用主办方的答题与判分提示词实测，它不提高答对率）。无自训练或开放权重模型。除规划器这一次模型调用外，整条链路是确定性的，便于复现。

【来源与披露】思路借鉴并已在 docs/METHOD_CARD.md、NOTICE 中注明：ReFind（imlrz/ReFind，MIT，arXiv:2608.12888）的时间戳渲染、上下文窗口、Porter + RRF、模型驱动的查询规划；InvMem（wenxiaof345-ctrl/vanilla-rag-memory）的稠密 + BM25 RRF 融合与相对时间标注；Hybrid Episodic Memory（tlysanhuo/agent-memory-challenge，MIT）的加权 RRF；FantaSmallhamster/ldbd-sqlite-fts-baseline 的「FTS5 存原文 + 最新回退」模式（v0.1 草稿曾改编其约 30 行，v0.2 起全部重写，现无其代码）。全部方法改动按版本（0.1–1.0）逐条记录在 docs/METHOD_CARD.md「Method changes by version」。

【运行指南】本地：uv sync && uv run pytest -q && uv run uvicorn --factory app.main:create_app --port 8000。生产：cp .env.example .env，填写 MEMORY_API_KEY、MEMORY_DOMAIN、OPENAI_API_KEY（OpenRouter）、DASHSCOPE_API_KEY，然后 docker compose -f deploy/docker-compose.prod.yml --env-file .env up -d --build；详见 README.md、deploy/README.md，配置项见 .env.example。
```
