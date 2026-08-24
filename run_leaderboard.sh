#!/usr/bin/env bash
set -euo pipefail

EVALUATIONS_ROOT="${1:-input_output_data/output/evaluations}"
OUTPUT_DIR="${2:-input_output_data/output/leaderboards}"

USER_FLAG=()
if [[ "${OS:-}" != "Windows_NT" ]] && command -v id >/dev/null 2>&1; then
  USER_FLAG=(--user "$(id -u):$(id -g)")
fi

docker compose run --rm \
  "${USER_FLAG[@]}" \
  --env HOME=/tmp \
  --entrypoint python3 \
  egra-eval \
  -m egra_eval2.leaderboard \
    --evaluations-root "$EVALUATIONS_ROOT" \
    --output-dir "$OUTPUT_DIR"
