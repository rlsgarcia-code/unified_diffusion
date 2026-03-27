#!/usr/bin/env bash
set -euo pipefail

HOST="${UDIFF_API_HOST:-127.0.0.1}"
PORT="${UDIFF_API_PORT:-8000}"
BASE_URL="http://${HOST}:${PORT}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/tmp/uv-cache}"
TMP_DIR="${TMPDIR:-/tmp}/udiff-smoke-local"
VERIFY_FILE="${TMP_DIR}/demo.safetensors"
REGISTRY_PATH="${TMP_DIR}/custom-models.json"
MODELS_DIR="${TMP_DIR}/models"

mkdir -p "${UV_CACHE_DIR}"
mkdir -p "${TMP_DIR}"
printf 'abc123' >"${VERIFY_FILE}"

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
curl -fsS -o /dev/null -D - "${BASE_URL}/"
curl -fsS "${BASE_URL}/openapi.json"
curl -fsS "${BASE_URL}/practices"
SHA256="$(shasum -a 256 "${VERIFY_FILE}" | awk '{print $1}')"
curl -fsS "${BASE_URL}/verify-file" \
  -H 'Content-Type: application/json' \
  -d "{\"path\":\"${VERIFY_FILE}\",\"sha256\":\"${SHA256}\"}"
curl -fsS "${BASE_URL}/register-local" \
  -H 'Content-Type: application/json' \
  -d "{\"path\":\"${VERIFY_FILE}\",\"registry_path\":\"${REGISTRY_PATH}\",\"models_dir\":\"${MODELS_DIR}\",\"model_slug\":\"smoke-model\",\"canonical_id\":\"local.civitai.smoke-model\",\"sha256\":\"${SHA256}\"}"
