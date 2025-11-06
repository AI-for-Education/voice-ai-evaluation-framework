#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  ./run_inference.sh --dataset_root PATH --output_dir PATH --model PATH [extra options]

Example:
  ./run_inference.sh \
    --dataset_root input_output_data/input/1_Batch2_Data_16spk_subset/ \
    --output_dir input_output_data/output/1_Batch2_Data_16spk_subset/nemo_asr_output/ \
    --model nemo_inference/models/Swahili_exp1_100epochs.nemo \
    --dataset_annotator Flora
EOF
  exit 1
}

DATASET_ROOT=""
OUTPUT_DIR=""
MODEL_PATH=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset_root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --output_dir|--output_root)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --model)
      MODEL_PATH="$2"
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

if [[ -z "$DATASET_ROOT" || -z "$OUTPUT_DIR" || -z "$MODEL_PATH" ]]; then
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
  nemo-asr \
  python3 /work/infer.py \
    --model "$MODEL_PATH" \
    --dataset_root "$DATASET_ROOT" \
    --output_root "$OUTPUT_DIR" \
    --debug \
    "${EXTRA_ARGS[@]}"
