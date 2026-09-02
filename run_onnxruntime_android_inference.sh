#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source "$SCRIPT_DIR/inference/execution_environment_identity.sh"
pipeline_execution_environment_docker_args \
  "onnxruntime-android-asr" \
  "voice-ai-evaluation-framework-onnxruntime-android:latest" \
  "run_onnxruntime_android_inference.sh"

USER_FLAG=()
if [[ "${OS:-}" != "Windows_NT" ]] && command -v id >/dev/null 2>&1; then
  USER_FLAG=(--user "$(id -u):$(id -g)")
fi

exec docker compose run --rm \
  "${USER_FLAG[@]}" \
  "${PIPELINE_EXECUTION_ENVIRONMENT_DOCKER_ARGS[@]}" \
  --env HOME=/tmp \
  --entrypoint "" \
  onnxruntime-android-asr \
  python3 -m inference.onnxruntime.android_infer "$@"
