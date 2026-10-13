#!/usr/bin/env bash
# Read-only output/mute diagnostics; --play also plays the latest MIRAI answer.
set -u
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
report() {
    printf '\n>>> '
    printf '%s ' "$@"
    printf '\n'
    "$@" || true
}
if command -v pactl >/dev/null 2>&1; then
    report pactl info
    report pactl list short sinks
    report pactl get-sink-mute @DEFAULT_SINK@
    report pactl get-sink-volume @DEFAULT_SINK@
else
    echo 'pactl not available (PulseAudio/PipeWire output information unavailable).'
fi
if command -v wpctl >/dev/null 2>&1; then
    report wpctl status
fi
if command -v aplay >/dev/null 2>&1; then
    report aplay -l
fi
if command -v amixer >/dev/null 2>&1; then
    report amixer sget Master
fi
if [[ -x .venv/bin/python ]]; then
    .venv/bin/python - "${1:-}" <<'PY'
from pathlib import Path
import struct
import sys
import wave

from config import Settings
from speech import Speech

path = Settings().state_dir / 'last-answer.wav'
print(f'\n>>> MIRAI audio file: {path}')
if not path.is_file():
    print('No saved answer found. Ask MIRAI a question with playback enabled first.')
else:
    with wave.open(str(path), 'rb') as wav:
        print(f'{wav.getnchannels()} channel(s), {wav.getframerate()} Hz, '
              f'{wav.getnframes() / wav.getframerate():.1f} seconds')
        if wav.getsampwidth() == 2:
            samples = struct.iter_unpack('<h', wav.readframes(wav.getnframes()))
            print('Peak sample level:', max((abs(sample[0]) for sample in samples), default=0))
    if sys.argv[1] == '--play':
        print('Playing this answer through the current default sound output...', flush=True)
        try:
            Speech.play(path)
            print('Player completed successfully. If silent, check the output device and mute/volume.')
        except Exception as error:
            print('Playback error:', error)
PY
fi
