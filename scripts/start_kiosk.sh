#!/usr/bin/env bash
# Touchscreen kiosk: start the MIRAI server, then show the UI full-screen on the Jetson's display.
#   bash scripts/start_kiosk.sh                 # 8 GB Orin Nano defaults (low-memory profile)
#   bash scripts/start_kiosk.sh --no-low-memory # regular PyTorch embeddings
#   bash scripts/start_kiosk.sh --server-only   # no browser (e.g. testing over SSH)
# Extra arguments go to server.py (e.g. --idle-timeout 180 --max-tokens 192).
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
if [[ ! -x .venv/bin/python ]]; then
    echo 'Run bash scripts/install_jetson.sh first.' >&2
    exit 1
fi

profile=(--low-memory)
browser=true
args=()
for arg in "$@"; do
    case "$arg" in
        --no-low-memory) profile=() ;;
        --server-only) browser=false ;;
        *) args+=("$arg") ;;
    esac
done
port=8000
for ((i = 0; i < ${#args[@]}; i++)); do
    [[ "${args[$i]}" == --port ]] && port="${args[$((i + 1))]}"
done
url="http://127.0.0.1:${port}"

.venv/bin/python server.py "${profile[@]}" "${args[@]}" &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT INT TERM

echo "Waiting for MIRAI at $url ..."
for _ in {1..120}; do
    if curl --fail --silent "$url/api/status" >/dev/null; then
        break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo 'MIRAI server stopped during startup; see the error above.' >&2
        exit 1
    fi
    sleep 1
done

if [[ "$browser" != true ]]; then
    echo "MIRAI server running at $url (Ctrl+C to stop)."
    wait "$server_pid"
    exit
fi

export DISPLAY="${DISPLAY:-:0}"
chrome="$(command -v chromium-browser || command -v chromium || command -v google-chrome || true)"
if [[ -n "$chrome" ]]; then
    # localhost counts as a secure origin, so the USB mic works without HTTPS. The flags
    # auto-allow the mic, allow spoken answers without an extra tap and block swipe-back.
    # CPU and GPU share 8 GB on the Jetson, so the last flags keep the browser small enough
    # that Ollama is not the process the OOM killer picks.
    "$chrome" --kiosk --app="$url" --user-data-dir="$HOME/.mirai-kiosk" \
        --use-fake-ui-for-media-stream --autoplay-policy=no-user-gesture-required \
        --noerrdialogs --disable-infobars --disable-translate --disable-features=Translate \
        --overscroll-history-navigation=0 --disable-pinch --disable-extensions \
        --no-first-run --password-store=basic --check-for-update-interval=31536000 \
        --renderer-process-limit=1 --disable-background-networking --disable-component-update \
        --disable-sync --disable-default-apps --js-flags=--max-old-space-size=128 &
elif command -v firefox >/dev/null 2>&1; then
    echo 'Chromium not found; Firefox kiosk will ask once for microphone permission.' >&2
    firefox --kiosk "$url" &
else
    echo "No browser found. Install one (sudo apt-get install chromium-browser) or open $url manually." >&2
fi
wait "$server_pid"
