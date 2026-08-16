#!/usr/bin/env bash
set -euo pipefail

# STRUCTURE ONLY ON THIS HOST: the tracked Qwen BF16 text-only profile requires
# >=40 GiB VRAM. The adapter will reject the present 16 GiB GPU before loading.
export MSYS_NO_PATHCONV="${MSYS_NO_PATHCONV:-1}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

USER_FLAG=()
if command -v id >/dev/null 2>&1; then
  USER_FLAG=(--user "$(id -u):$(id -g)")
fi

exec docker compose --profile large-gpu run --rm \
  "${USER_FLAG[@]}" \
  --env HOME=/tmp \
  --entrypoint "" \
  qwen-omni-asr \
  python3 -m inference.multimodal.infer "$@"
