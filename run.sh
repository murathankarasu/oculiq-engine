#!/usr/bin/env bash
# Oculiq local server — first run creates a venv, installs deps and downloads the model.
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "[oculiq] creating venv…"
  python3 -m venv .venv
fi
source .venv/bin/activate

# .env varsa yükle (API anahtarları vb.)
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

if ! python -c "import ultralytics, fastapi, uvicorn" 2>/dev/null; then
  echo "[oculiq] installing dependencies (one-time)…"
  pip install -q --upgrade pip
  pip install -q -r server/requirements.txt
fi

exec python -m uvicorn server.main:app --host 127.0.0.1 --port "${PORT:-8123}"
