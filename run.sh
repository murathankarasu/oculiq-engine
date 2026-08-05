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

# Varsayilan: yalnizca bu makine (127.0.0.1). Saha kurulumunda aga acmak icin
# .env'e HOST=0.0.0.0 ve OCULIQ_TOKEN=<uzun-rastgele-deger> yaz — token olmadan
# ag arayuzunden gelen istekler reddedilir (bkz. server/main.py).
HOST="${HOST:-127.0.0.1}"
if [ "$HOST" != "127.0.0.1" ] && [ -z "$OCULIQ_TOKEN" ]; then
  echo "[oculiq] HOST=$HOST ama OCULIQ_TOKEN tanimli degil — ag erisimi reddedilecek."
  echo "[oculiq] .env dosyasina ekleyin:  OCULIQ_TOKEN=$(python3 -c 'import secrets;print(secrets.token_urlsafe(24))')"
fi
exec python -m uvicorn server.main:app --host "$HOST" --port "${PORT:-8123}"
