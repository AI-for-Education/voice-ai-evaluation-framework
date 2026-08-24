#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  cat <<'EOF' >&2
Usage:
  ./run_build_reference_views.sh \
    --dataset_root input_output_data/input/<dataset> \
    --manifest_in input_output_data/output/<experiment>/manifests/ref_manifest.raw_segments.jsonl
EOF
  exit 1
fi

USER_FLAG=(--user "$(id -u):$(id -g)")
ENV_VARS=(
  --env HOME=/tmp
  --env XDG_CACHE_HOME=/tmp/.cache
)

docker compose run --rm "${USER_FLAG[@]}" "${ENV_VARS[@]}" --entrypoint "" \
  --workdir /work \
  egra-eval \
  python3 -m tools.build_reference_views "$@"
