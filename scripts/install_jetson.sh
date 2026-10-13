#!/usr/bin/env bash
# CPU-compatible first install. Ollama detects supported GPUs independently.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
if [[ "${1:-}" != "--no-system-packages" ]]; then
    sudo apt-get update
    sudo apt-get install -y python3-venv python3-pip curl zstd libgomp1 espeak-ng alsa-utils unzip
fi
bash scripts/prepare_python.sh
# Keep any already installed NVIDIA PyTorch. For a new Jetson, start with CPU PyTorch.
if ! .venv/bin/python -c 'import torch' >/dev/null 2>&1; then
    .venv/bin/python -m pip install --disable-pip-version-check torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
fi
.venv/bin/python -m pip install --disable-pip-version-check -r requirements-voice.txt -r requirements-server.txt -r requirements-dev.txt
.venv/bin/python -m pip check
if ! command -v ollama >/dev/null 2>&1; then
    installer="$(mktemp)"
    trap 'rm -f "$installer"' EXIT
    curl --fail --silent --show-error --location https://ollama.com/install.sh -o "$installer"
    OLLAMA_VERSION=0.35.1 sh "$installer"
    rm -f "$installer"
    trap - EXIT
fi
sudo systemctl enable --now ollama
ready=false
for attempt in {1..30}; do
    if curl --fail --silent http://127.0.0.1:11434/api/version; then
        ready=true
        break
    fi
    sleep 1
done
if [[ "$ready" != true ]]; then
    echo 'Ollama did not start. Check: sudo journalctl -u ollama --no-pager -n 40' >&2
    exit 1
fi
.venv/bin/python scripts/download_models.py
.venv/bin/python -m pytest -q
.venv/bin/python scripts/smoke_test.py --live --voice
mkdir -p data/pdfs
echo 'MIRAI installed. Add university PDFs to data/pdfs, then run: bash scripts/start_kiosk.sh'
