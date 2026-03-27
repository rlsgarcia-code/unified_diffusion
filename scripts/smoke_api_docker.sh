#!/usr/bin/env bash
set -euo pipefail

BASE_URL="http://127.0.0.1:8000"
TMP_DIR="${TMPDIR:-/tmp}/udiff-smoke-docker"
VERIFY_FILE="${TMP_DIR}/demo.safetensors"
REGISTRY_PATH="${TMP_DIR}/custom-models.json"
MODELS_DIR="${TMP_DIR}/models"

mkdir -p "${TMP_DIR}"
printf 'abc123' >"${VERIFY_FILE}"

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
