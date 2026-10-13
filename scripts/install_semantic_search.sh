#!/usr/bin/env bash
# Update the existing Python environment; keep Ollama, PDFs, voice models and memories.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
if [[ ! -x .venv/bin/python ]]; then
    echo 'Run bash scripts/install_jetson.sh first.' >&2
    exit 1
fi
.venv/bin/python -m pip install --disable-pip-version-check -r requirements-semantic.txt
.venv/bin/python -m pip check
.venv/bin/python scripts/download_models.py --onnx-only
.venv/bin/python scripts/smoke_test.py --embedding-backend onnx --num-ctx 2048 --live
echo 'Semantic search ready. Run: bash scripts/start_low_memory.sh'
