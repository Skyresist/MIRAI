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

# Autostart may already have a server loading: a second copy would fail with
# "[Errno 98] Address already in use", so wait for that one instead.
server_pattern='python[^ ]* server\.py'
server_pid=''
if pgrep -f "$server_pattern" >/dev/null; then
    echo 'A MIRAI server is already running (probably started at login); using it.'
else
    .venv/bin/python server.py "${profile[@]}" "${args[@]}" &
    server_pid=$!
    trap 'kill "$server_pid" 2>/dev/null || true' EXIT INT TERM
fi
server_alive() {
    if [[ -n "$server_pid" ]]; then kill -0 "$server_pid" 2>/dev/null; else pgrep -f "$server_pattern" >/dev/null; fi
}

# No time limit: a cold boot can take minutes, and opening the browser early shows a dead page.
echo "Waiting for MIRAI at $url ..."
until curl --fail --silent "$url/api/status" >/dev/null; do
    if ! server_alive; then
        echo 'MIRAI server stopped during startup; see the error above.' >&2
        exit 1
    fi
    sleep 1
done
echo 'MIRAI server is ready.'

if [[ "$browser" != true ]]; then
    echo "MIRAI server running at $url (Ctrl+C to stop)."
    if [[ -n "$server_pid" ]]; then wait "$server_pid"; fi
    exit
fi

export DISPLAY="${DISPLAY:-:0}"
# After every reboot Ubuntu picks the Jetson's built-in input (nothing plugged in: silence) as the
# default mic, and the browser uses the default. Wait (up to 60 s) for the USB mic, then make it the default.
if command -v pactl >/dev/null 2>&1; then
    for _ in $(seq 60); do
        # || true: at boot the sound server may not be up yet, and set -e must not end the kiosk.
        usb_mic="$(pactl list short sources 2>/dev/null | awk '$2 ~ /^alsa_input\..*usb/ {print $2; exit}' || true)"
        [[ -n "$usb_mic" ]] && pactl set-default-source "$usb_mic" && break
        sleep 1
    done
    echo "Microphone: $(pactl get-default-source 2>/dev/null)"
fi
chrome="$(command -v chromium-browser || command -v chromium || command -v google-chrome || true)"
if [[ -n "$chrome" ]]; then
    # A cached copy of the old screen would hide UI updates (e.g. the cancel button) after an update.
    rm -rf "$HOME/.mirai-kiosk/Default/Cache" "$HOME/.mirai-kiosk/Default/Code Cache"
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
    # A kiosk-only profile that always allows the mic (Firefox's permission prompt is hidden in kiosk
    # mode and its answer was lost after reboots) and lets answers play aloud. Snap Firefox may only
    # use folders under ~/snap/firefox, so the profile goes there when Firefox is a snap.
    profile="$HOME/.mirai-kiosk-firefox"
    [[ -d "$HOME/snap/firefox" ]] && profile="$HOME/snap/firefox/common/mirai-kiosk"
    mkdir -p "$profile"
    cat > "$profile/user.js" <<'EOF'
user_pref("media.navigator.permission.disabled", true);
user_pref("permissions.default.microphone", 1);
user_pref("media.autoplay.default", 0);
user_pref("browser.sessionstore.resume_from_crash", false);
EOF
    # Same reason as the Chromium cache above.
    rm -rf "$profile/cache2" "$HOME"/.cache/mozilla/firefox/*/cache2 "$HOME"/snap/firefox/common/.cache/mozilla/firefox/*/cache2
    firefox --kiosk --no-remote --profile "$profile" "$url" &
else
    echo "No browser found. Install one (sudo apt-get install chromium-browser) or open $url manually." >&2
fi
# Our own server, or else the browser.
wait
