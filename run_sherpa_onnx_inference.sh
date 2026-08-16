#!/usr/bin/env bash
set -euo pipefail

# Prevent Git Bash/MSYS from rewriting Linux container paths such as /work.
export MSYS_NO_PATHCONV="${MSYS_NO_PATHCONV:-1}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

USER_FLAG=()
if command -v id >/dev/null 2>&1; then
  USER_FLAG=(--user "$(id -u):$(id -g)")
fi

exec docker compose run --rm \
  "${USER_FLAG[@]}" \
  --env HOME=/tmp \
  --entrypoint "" \
  sherpa-onnx-asr \
  python3 -m inference.sherpa_onnx.infer "$@"
