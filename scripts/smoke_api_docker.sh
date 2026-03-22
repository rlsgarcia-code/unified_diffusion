#!/usr/bin/env bash
set -euo pipefail

BASE_URL="http://127.0.0.1:8000"

docker compose -f docker-compose.api.yml up -d --build

cleanup() {
  docker compose -f docker-compose.api.yml down >/dev/null 2>&1 || true
}
trap cleanup EXIT

for _ in $(seq 1 60); do
  if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

curl -fsS "${BASE_URL}/health"
curl -fsS "${BASE_URL}/models"
