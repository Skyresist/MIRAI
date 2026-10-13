#!/usr/bin/env bash
# Keep Qwen 4B, Piper and semantic retrieval; use the quantized ONNX embedding model.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$project_dir/scripts/start.sh" --low-memory "$@"
