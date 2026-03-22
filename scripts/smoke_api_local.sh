#!/usr/bin/env bash
set -euo pipefail

HOST="${UDIFF_API_HOST:-127.0.0.1}"
PORT="${UDIFF_API_PORT:-8000}"
BASE_URL="http://${HOST}:${PORT}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/tmp/uv-cache}"

mkdir -p "${UV_CACHE_DIR}"

UV_CACHE_DIR="${UV_CACHE_DIR}" uv run uvicorn service.fastapi_app.main:app --host "${HOST}" --port "${PORT}" >/tmp/udiff-api-local.log 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "${SERVER_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for _ in $(seq 1 30); do
  if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
  cat /tmp/udiff-api-local.log
  exit 1
fi

curl -fsS "${BASE_URL}/health"
curl -fsS "${BASE_URL}/models"
