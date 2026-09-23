#!/usr/bin/env bash
set -euo pipefail

# Prevent Git Bash/MSYS from rewriting Linux container paths such as /work.
export MSYS_NO_PATHCONV="${MSYS_NO_PATHCONV:-1}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source "$SCRIPT_DIR/inference/execution_environment_identity.sh"
pipeline_execution_environment_docker_args \
  "torch-asr" \
  "voice-ai-evaluation-framework-asr:latest" \
  "run_torch_inference.sh"

USER_FLAG=()
if [[ "${OS:-}" != "Windows_NT" ]] && command -v id >/dev/null 2>&1; then
  USER_FLAG=(--user "$(id -u):$(id -g)")
fi

exec docker compose run --rm \
  "${USER_FLAG[@]}" \
  "${PIPELINE_EXECUTION_ENVIRONMENT_DOCKER_ARGS[@]}" \
  --env HOME=/tmp \
  --entrypoint "" \
  torch-asr \
  python3 -m inference.torch.infer "$@"
