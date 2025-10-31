#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  ./run_eval.sh --dataset_root PATH --output_root PATH [extra options]

Example:
  ./run_eval.sh \
    --dataset_root /io/input/1_Batch2_Data-v2 \
    --output_root /io/output/experiments/exp_batch2 \
    --dataset_annotator Flora \
    --nemo_manifest /io/input/1_Batch2_Data-v2/nemo_asr_output/transcriptions.jsonl
EOF
  exit 1
}

DATASET_ROOT=""
OUTPUT_ROOT=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset_root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --output_root|--output_dir)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --help|-h)
      usage
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$DATASET_ROOT" || -z "$OUTPUT_ROOT" ]]; then
  usage
fi

USER_FLAG=(--user "$(id -u):$(id -g)")
ENV_VARS=(
  --env HOME=/tmp
  --env MPLCONFIGDIR=/tmp/matplotlib
  --env NUMBA_CACHE_DIR=/tmp/numba_cache
  --env XDG_CACHE_HOME=/tmp/.cache
  --env LHOTSE_TOOLS_DIR=/tmp/lhotse_tools
  --env LHOTSE_DATA_HOME=/tmp/lhotse_data
)

docker compose run --rm "${USER_FLAG[@]}" "${ENV_VARS[@]}" --entrypoint "" \
  egra-eval \
  python3 /work/evaluation.py \
    --dataset_root "$DATASET_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    "${EXTRA_ARGS[@]}"
