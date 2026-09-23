#!/usr/bin/env bash
set -euo pipefail

# Prevent Git Bash/MSYS from rewriting Linux container paths such as /work.
export MSYS_NO_PATHCONV="${MSYS_NO_PATHCONV:-1}"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY for workspace 'QA facility ASR' is not available in this process." >&2
  echo "Start a new terminal or Codex process after setting the user variable." >&2
  exit 2
fi

echo "[INFO] Required OpenRouter workspace: QA facility ASR (selected by API-key ownership)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source "$SCRIPT_DIR/inference/execution_environment_identity.sh"
pipeline_execution_environment_docker_args \
  "openrouter-asr" \
  "voice-ai-evaluation-framework-openrouter:latest" \
  "run_openrouter_inference.sh"

USER_FLAG=()
if [[ "${OS:-}" != "Windows_NT" ]] && command -v id >/dev/null 2>&1; then
  USER_FLAG=(--user "$(id -u):$(id -g)")
fi

exec docker compose run --rm \
  "${USER_FLAG[@]}" \
  "${PIPELINE_EXECUTION_ENVIRONMENT_DOCKER_ARGS[@]}" \
  --env OPENROUTER_API_KEY \
  --entrypoint "" \
  openrouter-asr \
  python3 -m inference.openrouter.infer "$@"
