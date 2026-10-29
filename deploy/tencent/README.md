# Deploy runbook — coding entry on the Tencent Cloud box

Target: `https://codememory.azorashopping.site` → the memory container of this repository, on a box whose ports 80/443
already belong to the nginx container of another stack (`~/AIAzora-v3`, compose project `aiazora-v3`). That nginx keeps
owning the ports; this stack only publishes the memory service on the docker bridge gateway (`BIND_IP:8200`, not reachable
from the internet) and the nginx gets one hostname block that proxies to it. The textual entry on the Alibaba Cloud box
(`memory.azorashopping.site`) is a different deployment and is not touched by anything here.

Box: Ubuntu, 2 vCPU / 3 GB, docker without sudo, login `ssh -i ~/.ssh/id_ed25519_43_162_99_246 ubuntu@43.162.107.53`.
Repo on the box: `~/aml-memory-coding`. Compose project: `codememory` (volume `codememory-data`).

## One-time
1. DNS: `A` record `codememory.azorashopping.site` → the box (done 2026-10-28; `dig +short codememory.azorashopping.site`).
2. Code onto the box — either `git clone https://github.com/Janettbella69/Aml-memory-coding.git ~/aml-memory-coding`
   (once the repository is pushed), or from the laptop without GitHub:
   `git archive --format=tar --prefix=aml-memory-coding/ HEAD | ssh -i ~/.ssh/id_ed25519_43_162_99_246 ubuntu@43.162.107.53 'tar -x -C ~'`.
3. `.env`: `cd ~/aml-memory-coding && cp .env.example .env`, then set `MEMORY_API_KEY` to a random secret ≥ 16 chars
   (`openssl rand -hex 24`); it is declared as `X-Api-Key` and sent to the organisers through their secure channel only.
   coding-0.2.0 also needs `OPENAI_API_KEY` (the OpenRouter key; `OPENAI_BASE_URL=https://openrouter.ai/api/v1` and the
   model ids `openai/gpt-4o-mini` are already in the example) for the Add-time summaries and the card selector; keep
   `SUMMARY_CACHE=/data/summary_cache.sqlite` (inside the volume, so a history written again costs nothing). The start-up
   log must show `add-time summaries enabled` and `card selector enabled`; without the key both silently stay off.
   Check the bridge gateway: `docker network inspect bridge --format '{{(index .IPAM.Config 0).Gateway}}'` — if it is not
   `172.17.0.1`, set `BIND_IP=` in `.env` to that address.
4. Start the service: `docker compose -p codememory -f deploy/tencent/docker-compose.yml --env-file .env up -d --build`,
   then `curl -s http://$(grep ^BIND_IP .env | cut -d= -f2):8200/health` → `{"status":"ok","version":"coding-0.2.0"}`.
5. nginx, step 1 (HTTP): insert `deploy/tencent/nginx-80.conf` inside the `http { … }` block of `~/AIAzora-v3/nginx.conf`
   (before its final `}`), then `docker exec aiazora-v3-nginx-1 nginx -t && docker exec aiazora-v3-nginx-1 nginx -s reload`.
   The file is bind-mounted read-only into the container, so editing it on the host is enough.
6. Certificate (the certbot container of the Azora stack, webroot shared with nginx):
   `cd ~/AIAzora-v3 && docker compose -f docker-compose.prod.yml run --rm certbot certonly --webroot -w /var/www/certbot -d codememory.azorashopping.site --agree-tos --no-eff-email`
   → `Successfully received certificate` under `certbot/conf/live/codememory.azorashopping.site/`.
7. nginx, step 2 (TLS): insert `deploy/tencent/nginx-443.conf` the same way, `nginx -t`, reload.
8. `curl -s https://codememory.azorashopping.site/health` → `{"status":"ok","version":"coding-0.2.0"}`;
   from the laptop: `uv run python scripts/smoke_test.py https://codememory.azorashopping.site <MEMORY_API_KEY>`.

Renewal: the certbot container renews every certificate it holds every 12 h; nginx needs a reload afterwards
(`docker exec aiazora-v3-nginx-1 nginx -s reload`), the same gap the main site has. The certificate lasts 90 days,
longer than the evaluation.

## Capacity check (numbers go into docs/METHOD_CARD.md and the submission sheet)
From the laptop, at a quiet hour for the other site on the box (2 vCPU shared):
`uv run python scripts/soak_local.py --base https://codememory.azorashopping.site --api-key <MEMORY_API_KEY> --users 100 --chunks 10`
(16-way concurrent Add + Search; record p50/p99 and errors). Disk: `df -h /` on the box; the coding track's Add volume is
not published (12 repositories, 1,290 historical tasks per the organisers' README) — budget ≥ 10 GB free.

## Every update
`cd ~/aml-memory-coding && git pull` (or a new `git archive … | tar -x`), then
`docker compose -p codememory -f deploy/tencent/docker-compose.yml --env-file .env up -d --build` and the health check.
Never during an accepted Full run (platform rule: no deploys, model or configuration changes; stateless restarts only).

## Before the Full run
Start from an empty store:
`docker compose -p codememory -f deploy/tencent/docker-compose.yml --env-file .env down -v && docker compose -p codememory -f deploy/tencent/docker-compose.yml --env-file .env up -d --build`

## After the Full run (within 30 days, rule)
`docker compose -p codememory -f deploy/tencent/docker-compose.yml --env-file .env down -v` (drops `codememory-data`).
The nginx blocks and the certificate can stay; endpoints need not stay up after 2026-11-04.
