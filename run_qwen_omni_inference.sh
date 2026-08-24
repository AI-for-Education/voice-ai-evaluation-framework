#!/usr/bin/env bash
set -euo pipefail

# STRUCTURE ONLY ON THIS HOST: the tracked Qwen BF16 text-only profile requires
# >=40 GiB VRAM. The adapter will reject the present 16 GiB GPU before loading.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source "$SCRIPT_DIR/inference/runtime_identity.sh"
pipeline_runtime_docker_args \
  "qwen-omni-asr" \
  "voice-ai-evaluation-framework-qwen-omni:latest" \
  "run_qwen_omni_inference.sh"

USER_FLAG=()
# Git Bash reports a Windows SID-derived UID that Docker Desktop cannot use to
# write bind-mounted files. Docker Desktop already maps those writes safely.
if [[ "${OS:-}" != "Windows_NT" ]] && command -v id >/dev/null 2>&1; then
  USER_FLAG=(--user "$(id -u):$(id -g)")
fi

exec docker compose --profile large-gpu run --rm \
  "${USER_FLAG[@]}" \
  "${PIPELINE_RUNTIME_DOCKER_ARGS[@]}" \
  --env HOME=/tmp \
  --entrypoint "" \
  qwen-omni-asr \
  python3 -m inference.multimodal.infer "$@"
