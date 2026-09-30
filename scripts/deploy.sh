#!/usr/bin/env bash
# Deploy AI Web Apps: build React UI, verify models, and start server
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== 1. Building React Web Interface ==="
if [ -d "web" ]; then
  npm --prefix web install --no-audit --no-fund
  npm --prefix web run build
  echo "[OK] web/dist built successfully."
fi

echo "=== 2. Running Automated Tests (pytest) ==="
ENABLED_MODELS= pytest -q
echo "[OK] Pytests passed 100%."

echo "=== 3. Starting AI Web Apps Server ==="
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
echo "Server running at: http://$HOST:$PORT"
echo "Swagger API Docs: http://localhost:$PORT/docs"
echo "React Web UI:    http://localhost:$PORT/"
exec uvicorn api.main:app --host "$HOST" --port "$PORT"
