#!/usr/bin/env bash
# Run ON THE VPS from the repo root:  bash deploy/deploy.sh
# Builds/starts the stack, waits for /health through Caddy, runs the contract smoke with the real key.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] || { echo "missing .env (cp .env.example .env; set MEMORY_API_KEY and MEMORY_DOMAIN)"; exit 1; }
set -a; . ./.env; set +a
: "${MEMORY_DOMAIN:?MEMORY_DOMAIN must be set in .env}"
: "${MEMORY_API_KEY:?MEMORY_API_KEY must be set in .env}"
df -h / | tail -1 | awk '{print "disk free on /: " $4 " (need >= 20G for a Full run)"}'
docker compose -f deploy/docker-compose.prod.yml --env-file .env up -d --build
echo "waiting for https://${MEMORY_DOMAIN}/health ..."
for i in $(seq 1 60); do
  if curl -fsS "https://${MEMORY_DOMAIN}/health" >/dev/null 2>&1; then echo "health ok"; break; fi
  sleep 3
  [ "$i" = 60 ] && { echo "health never came up; docker compose -f deploy/docker-compose.prod.yml logs caddy memory"; exit 1; }
done
python3 scripts/smoke_test.py "https://${MEMORY_DOMAIN}" "${MEMORY_API_KEY}"
echo "deployed: https://${MEMORY_DOMAIN}/v1/memories/add  https://${MEMORY_DOMAIN}/v1/memories/search"
