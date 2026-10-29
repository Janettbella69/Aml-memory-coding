# Submission sheet — AML Cycle 2, Coding Memory track (Open-source Methods)

Fill-in sheet for the coding entry's evaluation-access request and Create Eval Job form (field names as on agentmemoryleaderboard.ai, checked 2026-10-07; the organisers' written requirements for a separately registered coding implementation, 2026-10-28, are the five items marked ★). `TBD` = not decided yet. Keep issued keys and personal contact details out of this file; enter them only in the form or the organisers' secure channel.

The textual entry (`aml-memory` v1.0.0, `Janettbella69/Aml-memory@9f5c5bf`, `memory.azorashopping.site`) keeps its own binding and its Full result; nothing below replaces it.

## Evaluation access request (new binding for the coding implementation — FAQ Q6 "更换端点流程": name the original system and application, and the target track)

| Field | Value |
|---|---|
| System name | `aml-memory-coding` |
| ★ Version name | `coding-0.2.0` |
| Contact details | TBD (enter in the form only; do not commit) |
| Organization or team | same as the textual entry |
| Intended evaluation type | Coding Memory track · Open-source Methods division |
| Method or product description | Deterministic retrieval service for the AML Add/Search contract, specialised for agent transcripts. Raw turns of earlier tasks in the same repository are stored in SQLite (one file per `user_id`) and indexed with FTS5 (porter-stemmed words, camelCase identifiers also by their inner words, character trigrams for CJK text). Search ranks turns with BM25 over the new task's description, groups the hits by past session, ranks the sessions by the reciprocal-rank sum of their hits with tool output weighted down, and returns one card per session: the session's opening turn (the task as it was given), a labelled Add-time summary of the session (one `gpt-4o-mini` call per Add request, stored next to the verbatim turns, never instead of them), the agent's closing message (what was changed and how it was verified), then its matching turns, rendered with UTC timestamps and roles; one `gpt-4o-mini` call per Search then reads the briefs of the leading cards and moves the useful sessions first. Search never generates an answer: every result is stored transcript text or that labelled summary, and either call failing leaves the deterministic order. |
| Information approved for public display | System name, version name, the method description above, the public repository URL and the fixed commit. |
| Complete submission notes | Deployed API version `coding-0.2.0`; endpoints below; auth `X-Api-Key`; capacity below; operational constraints: single instance behind a shared nginx, SQLite, no server-side request timeout, Add synchronous; prior-work attribution, evaluation-data attribution and method changes in `docs/METHOD_CARD.md` and `NOTICE`. |
| ★ Deployed Add endpoint URL | `https://codememory.azorashopping.site/v1/memories/add` (Tencent Cloud box, nginx TLS — a different server from the textual entry) |
| ★ Deployed Search endpoint URL | `https://codememory.azorashopping.site/v1/memories/search` |
| ★ Health URL | `https://codememory.azorashopping.site/health` (unauthenticated GET, 200, returns `{"status":"ok","version":"coding-0.2.0"}`) |
| ★ Authentication | `X-Api-Key: <Memory System Key>` (declared scheme; `Authorization: Bearer` / `Token` also accepted). A NEW key, generated for this deployment only — never the textual entry's key — sent through the organisers' secure channel, not in email text or this repository. |
| Memory System Key | the value of `MEMORY_API_KEY` in the Tencent box's `.env` (≥ 16 characters); never committed |
| ★ Capacity declaration | Max add concurrency 16 · Search concurrency 16 · `top_k` 1–1000 (official runs use 100) · Add synchronous, persisted and searchable before the 200 · no server-side timeout (platform limit 30 min/request) · single instance · `client_max_body_size 200m` at the proxy. Measured 2026-10-29 through TLS at 16-way (`scripts/soak_local.py`): 160 Adds, 0 errors, 5.6 adds/s; 64 Searches, 0 errors |
| Typical latency (FAQ Q5) | Measured 2026-10-29 through TLS at 16-way concurrency: Add p50 2.4 s / p90 3.6 s / p99 4.7 s (one gpt-4o-mini summary call per Add); Search p50 0.3 s / p99 1.2 s without the card selector firing, 2–14 s with it (one gpt-4o-mini call); a whole trajectory per Add takes 5–15 s — all well inside the platform's 30-minute limit |
| Request body & storage capacity (FAQ Q5) | Add bodies up to 200 MB at the proxy (a whole trajectory per request is fine); storage ≈ 5× raw text at rest; disk headroom recorded with the soak |
| Models & external dependencies (FAQ Q5/Q7) | `gpt-4o-mini` only (via OpenRouter, model id `openai/gpt-4o-mini` = the same OpenAI model, routed): one call per Add request that writes a labelled summary turn stored next to the verbatim turns, and one call per Search that re-orders the leading session cards; both fall back to the deterministic behaviour on any failure. No embedding model, no reranker; SQLite FTS5, numpy; nginx TLS. |
| Multimodal processing declaration (FAQ Q5) | N/A — Coding track only |
| Service maintenance (FAQ Q5) | Single operator; health monitored via `/health`; Docker `restart: unless-stopped`; no changes during a Full run (FAQ Q29); idempotency records and data volume kept until the run is closed (FAQ Q26/Q27), deleted within 30 days after (FAQ Q31) |
| Public repository URL | `https://github.com/Janettbella69/Aml-memory-coding` (public) |
| ★ Fixed commit hash | `873b4d726548c3c70d2fd46282ab2318f5ba41be` = the commit tagged `coding-0.2.0` (pushed 2026-10-29; `git rev-parse coding-0.2.0^{commit}`) — the deployed code is that commit (`git clone --branch coding-0.2.0`) |
| Attribution of prior work | `docs/METHOD_CARD.md` § "Prior work and attribution" (ReFind — MIT, incl. the session-level RRF idea; the rewritten FTS5-baseline history) and `NOTICE` (also SWEContextBench, the evaluation data used for the local retrieval check) |
| Method changes | `docs/METHOD_CARD.md` § "Method changes by version", entry `coding-0.2.0` (base = textual 1.0.0 imported byte-for-byte) |

## Create Eval Job

| Field | Value |
|---|---|
| Evaluation track | Coding |
| System name | `aml-memory-coding` |
| Version name | `coding-0.2.0` |
| Run label | `aml-memory-coding-0.1-smoke-01`, `-smoke-02`, … then `aml-memory-coding-0.1-full-01` (labels unique per key) |
| Mode | `smoke` first (the platform requires a passed smoke per track and binding); `full` once it passes and `docs/CHECKLIST.md` is confirmed |
| Max add concurrency | 16 (or the value measured in the soak) |
| Search concurrency | 16 |

## Quotas and dates (official)

- Smoke: 1 per hour; at most 30 per track this edition. Private, never public-eligible. A smoke passed by the textual binding does not count for this one.
- Full: at most 2 per AML Key per track; the second unlocks 30 days after the first — one Full in practice. A failed or cancelled Full resumes from its last checkpoint and does not consume a new Full. Official duration guidance for a Full: 0.5–2 days.
- Materials deadline: 2026-10-31 23:59 (UTC+8). Evaluation closes: 2026-11-04 23:59 (UTC+8). Official leaderboard: mid-November 2026.
- Once a Full evaluation is accepted the version cannot be replaced or withdrawn; freeze the version (tag) before applying.
- Before every Full run the 8-item gate in `docs/CHECKLIST.md` must be confirmed.

## 提交说明与运行指南（表单文本，2026-10-29 已填入固定 commit 与压测结果）

```
【系统】aml-memory-coding，版本名 coding-0.2.0，固定 commit 873b4d726548c3c70d2fd46282ab2318f5ba41be（git tag coding-0.2.0），公开仓库 https://github.com/Janettbella69/Aml-memory-coding（MIT）。参赛组别：开源方法榜；赛道：代码记忆。按组委会 2026-10-28 回复单独备案：本实现与文本赛道的 aml-memory v1.0.0（Janettbella69/Aml-memory@9f5c5bf，memory.azorashopping.site）是不同代码、不同地址、不同凭据，不覆盖、不混入既有文本评测。

【部署】腾讯云单实例（2 vCPU / 3 GB，与另一站点共用一台机，nginx TLS），Docker。
Add：https://codememory.azorashopping.site/v1/memories/add  Search：https://codememory.azorashopping.site/v1/memories/search  Health：https://codememory.azorashopping.site/health（无鉴权 GET）。
鉴权：X-Api-Key（同时接受 Authorization: Bearer / Token），本部署专用的新密钥，经安全渠道提交。容量与限制：Add 并发 16、Search 并发 16、top_k 1–1000；单次请求体上限 200 MB（整段会话轨迹可一次写入）；Add 同步写入，返回 200 即已持久化且可检索（写入时的一次 gpt-4o-mini 摘要调用也在返回前完成）；request_id 幂等（同 id 不同内容返回 409），幂等记录持久化在 SQLite，重启不丢；无服务端超时（遵守平台 30 分钟上限）。实测（2026-10-29，16 路并发、公网 TLS）：Add 160 次 0 错误、p50 2.4 s / p99 4.7 s；Search 64 次 0 错误、p50 0.3 s / p99 1.2 s，触发卡片选择器时单次 2–14 s。评测数据不记录文本日志、不用于训练，任务结束 30 天内删除。

【方法】面向同仓库历史工程记录（Agent 会话轨迹：任务描述、工具调用、工具输出、推理与修改）。每个 user_id 一个独立 SQLite 文件，原始回合逐条存储，不做摘要。检索：FTS5 BM25（Porter 词干；驼峰标识符同时按内部单词建索引与查询；CJK 字符三元组）。装配：命中按历史会话分组，会话按其命中的倒数排名求和排序（非开头的 user 回合即工具输出权重 0.3），每个会话返回一张「卡片」——该会话的开头回合（当时的任务原文）、写入时生成的摘要、Agent 的收尾消息（改了什么、怎么验证的）加最多 8 条命中回合，渲染为「[UTC 时间戳] 角色: 原文」；最后由一次 gpt-4o-mini 调用按对新任务的有用程度把前 12 张卡片重新排序（只动顺序）。不使用规则槽、时间优先和上下文窗口（面向对话的设计，对会话轨迹无益，已实测）。只返回原文证据，不生成、不伪装答案；不跨 user_id 读取。

【模型】唯一使用的模型是 gpt-4o-mini（经 OpenRouter 路由，模型标识 openai/gpt-4o-mini，即 OpenAI 同一模型；复现时可设 OPENAI_BASE_URL=https://api.openai.com/v1、模型名 gpt-4o-mini 直连官方接口，代码无需改动），两处调用：(1) Add 阶段每个写入请求一次，把该段会话轨迹写成带标签的摘要（任务 / 根因 / 改动文件 / 修法 / 验证 / 坑），作为一条标明「写入时生成」的额外回合存在原文旁边，原文一条不少；(2) Search 阶段每次一次，读取前 12 张卡片的简介（任务原文、摘要、收尾消息）判断哪些历史会话对新任务有用，把这些卡片整体前移，不生成、不改写任何返回文本。两处调用失败时均退回确定性行为。不使用向量模型、不使用 reranker（代码中保留的 text-embedding-v4 向量通道、查询规划器、reranker 在本版本配置下关闭）。无自训练或开放权重模型。

【来源与披露】基础代码即本队文本赛道的 aml-memory v1.0.0 原样导入（docs/METHOD_CARD.md 全文记录）。思路借鉴并已注明：ReFind（imlrz/ReFind，MIT，arXiv:2608.12888）的时间戳渲染、Porter + RRF、会话级 RRF；InvMem、Hybrid Episodic Memory（见方法卡片）；FantaSmallhamster/ldbd-sqlite-fts-baseline 的「FTS5 存原文 + 最新回退」模式（v0.1 草稿曾改编其约 30 行，v0.2 起全部重写）。本地检索检验数据：SWEContextBench（jiayuanz3，Hugging Face MIT，arXiv:2602.08316；官网列为代码赛道参考目录），仅用于检验、未用于训练，数据不入库。全部方法改动按版本记录在 docs/METHOD_CARD.md「Method changes by version」，代码赛道改动在 coding-0.2.0 条目。

【运行指南】本地：uv sync && uv run pytest -q && uv run uvicorn --factory app.main:create_app --port 8000。生产：cp .env.example .env，填写 MEMORY_API_KEY 和 OPENAI_API_KEY（OpenRouter），然后 docker compose -p codememory -f deploy/tencent/docker-compose.yml --env-file .env up -d --build（挂在已有 nginx 后面，见 deploy/tencent/README.md）；独立机器可用 deploy/README.md 的 Caddy 方案。配置项见 .env.example。
```
