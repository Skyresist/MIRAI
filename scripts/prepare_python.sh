#!/usr/bin/env bash
# Bootstrap a private Python on JetPack 6/Ubuntu 22.04 without changing system Python.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
if [[ -x .venv/bin/python ]] && .venv/bin/python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
    echo 'Keeping the existing compatible MIRAI Python environment.'
    exit 0
fi
bootstrap_python="${MIRAI_BOOTSTRAP_PYTHON:-python3}"
if [[ ! -x .setup-tools/bin/python ]]; then
    "$bootstrap_python" -m venv .setup-tools
fi
.setup-tools/bin/python -m pip install --disable-pip-version-check uv==0.12.19
export UV_CACHE_DIR="$project_dir/.setup-tools/cache"
export UV_PYTHON_INSTALL_DIR="$project_dir/.python"
.setup-tools/bin/uv python install 3.12 --no-bin
# Preserve an older or incomplete environment; never delete its packages.
if [[ -e .venv || -L .venv ]]; then
    backup_dir="$(mktemp -d "$project_dir/.venv-backup.XXXXXX")"
    mv .venv "$backup_dir/venv"
    echo "Preserved the previous environment in: $backup_dir/venv"
fi
.setup-tools/bin/uv venv --python 3.12 --managed-python --seed .venv
.venv/bin/python -c 'import sys; print("MIRAI Python:", sys.version.split()[0])'
