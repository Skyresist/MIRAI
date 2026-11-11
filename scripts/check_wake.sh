#!/usr/bin/env bash
# Checks every step of the "Halo MIRAI" wake phrase and prints what it finds:
#   bash scripts/check_wake.sh
# Run it while the kiosk is on the sleeping screen. It asks you to say "Halo MIRAI" once.
# The recording is deleted afterwards.
set -uo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

echo '== 1. Code version (should be the newest commit)'
git log --oneline -1

echo; echo '== 2. Folder the kiosk starts from at login (should be this folder)'
echo "this folder: $project_dir"
grep -h '^Exec' "$HOME/.config/autostart/mirai-kiosk.desktop" 2>/dev/null || echo 'no autostart entry'

echo; echo '== 3. MIRAI server running'
pgrep -af 'server\.py' || echo 'NOT running'

echo; echo '== 4. Mics plugged in'
pactl list short sources | awk '$2 !~ /monitor$/ {print $1, $2}'
respeaker="$(pactl list short sources | awk '$2 ~ /^alsa_input\..*ReSpeaker/ {print $2; exit}')"
[[ -n "$respeaker" ]] && echo "ReSpeaker found: $respeaker" || echo 'ReSpeaker NOT found'

echo; echo '== 5. Mic the kiosk browser is recording from right now'
outputs="$(pactl list short source-outputs)"
if [[ -z "$outputs" ]]; then
    echo 'NOTHING is recording: the screen is not listening.'
else
    pactl list short sources | awk 'NR == FNR {name[$1] = $2; next} {print "recording from:", name[$2]}' - <(echo "$outputs")
fi

echo; echo '== 6. Live test: say "Halo MIRAI" now (recording 4 seconds) ...'
clip="$(mktemp --suffix=.wav)"
trap 'rm -f "$clip"' EXIT
if [[ -z "$respeaker" ]]; then
    echo 'skipped: no ReSpeaker'
    exit
fi
timeout -s INT 4 parecord --device="$respeaker" --channels=1 --rate=16000 --file-format=wav "$clip"
echo 'done recording, asking Whisper ...'
.venv/bin/python - "$clip" <<'EOF'
import re, sys, wave, array
from config import Settings
from glossary import Glossary
from speech import Speech

with wave.open(sys.argv[1]) as clip:
    samples = array.array('h', clip.readframes(clip.getnframes()))
peak = max(map(abs, samples), default=0)
print(f'loudest sound: {peak} of 32767', '(too quiet: speak closer or raise the mic volume)' if peak < 2000 else '(ok)')
settings = Settings()
try:
    text, _ = Speech(settings, Glossary(settings.glossary_path)).transcribe(sys.argv[1], 'id', strict=False)
except ValueError:
    text = ''
print(f'Whisper heard: {text!r}')
# Same check as WAKE in static/app.js.
woke = re.search(r'mir[ae][iey]|h[ae]l+o+mira', re.sub('[^a-z]', '', text.lower()))
print('WAKE PHRASE MATCHED' if woke else 'wake phrase NOT matched')
EOF
