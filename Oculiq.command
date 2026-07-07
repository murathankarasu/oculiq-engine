#!/usr/bin/env bash
# Oculiq başlatıcı — Finder'da ÇİFT TIKLA yeter.
# Sunucuyu başlatır, hazır olunca tarayıcıyı kendisi açar.
# (İlk açışta macOS uyarırsa: sağ tık → Open.)
cd "$(dirname "$0")"
PORT="${PORT:-8123}"

# Zaten çalışıyorsa sadece tarayıcıyı aç
if curl -s -o /dev/null "http://127.0.0.1:$PORT/api/config" 2>/dev/null; then
  echo "Oculiq zaten çalışıyor — tarayıcı açılıyor…"
  open "http://localhost:$PORT"
  exit 0
fi

echo "Oculiq başlatılıyor (port $PORT)…"
echo "Kapatmak için bu pencerede Ctrl+C."

# Sunucu hazır olunca tarayıcıyı aç (arka planda bekler)
(
  for _ in $(seq 1 90); do
    sleep 1
    if curl -s -o /dev/null "http://127.0.0.1:$PORT/api/config" 2>/dev/null; then
      open "http://localhost:$PORT"
      exit 0
    fi
  done
) &

exec ./run.sh
