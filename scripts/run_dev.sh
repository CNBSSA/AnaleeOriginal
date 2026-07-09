#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${HOME}/.local/bin:${PATH}"
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
export FLASK_ENV="${FLASK_ENV:-development}"
export FLASK_SECRET_KEY="${FLASK_SECRET_KEY:-dev-secret-key-for-local}"
export DATABASE_URL="${DATABASE_URL:-sqlite:////workspace/dev.db}"
export PORT="${PORT:-5000}"
exec python3 main.py
