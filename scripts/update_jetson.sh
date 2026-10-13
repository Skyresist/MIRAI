#!/usr/bin/env bash
# Run ON THE JETSON after new code was pushed from the laptop:
#   bash scripts/update_jetson.sh
# Pulls the code, installs any new Python packages, fetches new models, runs the tests.
# Keeps PDFs (data/pdfs), models/, .mirai/ caches and .venv - git never touches them.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
if [[ ! -x .venv/bin/python ]]; then
    echo 'No .venv yet: run bash scripts/install_jetson.sh first.' >&2
    exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo 'This Jetson copy has local edits to tracked files; commit or stash them first:' >&2
    git status --short --untracked-files=no >&2
    exit 1
fi

before="$(git rev-parse HEAD)"
git pull --ff-only
after="$(git rev-parse HEAD)"
if [[ "$before" == "$after" ]]; then
    echo 'Already up to date.'
else
    echo 'Changes pulled:'
    git log --oneline "$before..$after"
fi

.venv/bin/python -m pip install --disable-pip-version-check -q -r requirements-voice.txt -r requirements-server.txt -r requirements-dev.txt
.venv/bin/python -m pip check
# Only downloads what is missing (e.g. the ~135 MB e5-small ONNX model after the first update).
# The kiosk uses quantised ONNX embeddings; Whisper and Piper voices came with install_jetson.sh.
.venv/bin/python scripts/download_models.py --onnx-only
model="$(.venv/bin/python -c 'from config import BASE_MODEL; print(BASE_MODEL)')"
if ! ollama list | grep -q "$model"; then
    .venv/bin/python scripts/pull_model.py --model "$model"
fi
.venv/bin/python -m pytest -q
echo
echo 'Update complete. Restart the kiosk to use it:  bash scripts/start_kiosk.sh'
