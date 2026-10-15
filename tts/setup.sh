#!/usr/bin/env bash
# Builds the Kokoro + Applio RVC environment next to this script.
set -euo pipefail
cd "$(dirname "$0")"

APPLIO_COMMIT=c6fc349fee0e4886811425ff3d9e61cd1b32a3d5

if [ ! -d Applio ]; then
    git clone https://github.com/IAHispano/Applio.git
    git -C Applio checkout "$APPLIO_COMMIT"
fi

python3 -m venv rvc-venv
rvc-venv/bin/pip install --no-cache-dir --upgrade pip
rvc-venv/bin/pip install --no-cache-dir torch==2.11.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu128
# Applio engine deps minus torch and Windows-only packages
grep -vE '^(torch|torchaudio)==|sys_platform == "win32"|^#' Applio/requirements-engine.txt \
    > /tmp/applio-req.txt
rvc-venv/bin/pip install --no-cache-dir -r /tmp/applio-req.txt -r requirements.txt
rm /tmp/applio-req.txt

echo "Done. Put your RVC .pth/.index in this folder and run ./speak.py -m model.pth \"text\""
