#!/usr/bin/env bash
# Guided microphone test: read each question aloud, then compare Whisper base vs small on it.
#   bash scripts/record_stt.sh                    # records into recordings/, then runs eval_stt.py
#   MIRAI_MIC=plughw:1,0 bash scripts/record_stt.sh   # pick a mic from `arecord -l`
# Speak at kiosk distance, the way a visitor would. Recordings stay on the Jetson (git ignores them).
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
folder="${1:-recordings}"
seconds=6
device=()
[[ -n "${MIRAI_MIC:-}" ]] && device=(-D "$MIRAI_MIC")

# Brand names and mixed Indonesian/English are where speech recognition fails, so they lead.
questions=(
    'id	Apa itu BINUS ASO?'
    'en	What is BINUS ASO?'
    'id	Syarat Widia Scholarship apa?'
    'en	What does the AJGS scholarship cover?'
    'id	Ada UKM apa saja?'
    "en	What is D'BASE?"
    'en	Is there a kendo club?'
    "id	Apa itu D'BASE?"
    "id	Apa bedanya D'BASE dan BASE CORE?"
    'id	Ceritakan tentang AeroBASE.'
    "id	Saya suka desain, apa itu D'Sign?"
    'id	HMB itu apa?'
    'en	What if I like robotics more?'
    'id	TPKS itu apa?'
    'id	Ada jurusan apa aja?'
    'en	What do students learn in Automotive and Robotics Engineering?'
    'id	Bisa kerja di Jepang setelah lulus?'
    'en	Do I need to speak Japanese before enrolling?'
    'id	Berapa biaya kuliahnya?'
    'en	Can I pay in installments?'
)

mkdir -p "$folder"
: > "$folder/transcripts.tsv"
echo "Read each question aloud after pressing Enter (${seconds} s per clip). Type s + Enter to skip one."
for i in "${!questions[@]}"; do
    IFS=$'\t' read -r language text <<< "${questions[$i]}"
    file="$(printf 'q%02d.wav' $((i + 1)))"
    printf '\n[%d/%d] (%s)  "%s"\n' $((i + 1)) "${#questions[@]}" "$language" "$text"
    read -r -p 'Enter = record, s = skip: ' answer
    [[ "$answer" == s ]] && continue
    arecord -q "${device[@]}" -f S16_LE -r 16000 -c 1 -d "$seconds" "$folder/$file"
    # A muted or wrong mic records silence and every model then "fails"; catch it per clip.
    peak="$(.venv/bin/python -c 'import array, sys, wave; w = wave.open(sys.argv[1]); s = array.array("h", w.readframes(w.getnframes())); print(max(map(abs, s), default=0))' "$folder/$file")"
    if (( peak < 1000 )); then
        echo "  Very quiet (peak $peak of 32767): check the mic, its volume (alsamixer) or MIRAI_MIC."
    else
        echo "  Recorded (peak $peak of 32767)."
    fi
    printf '%s\t%s\t%s\n' "$file" "$language" "$text" >> "$folder/transcripts.tsv"
done

echo
echo "Comparing Whisper models (first run downloads them)..."
.venv/bin/python scripts/eval_stt.py "$folder" --models base small
