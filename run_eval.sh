#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  ./run_eval.sh --dataset_root PATH --output_root PATH --passages_csv PATH [extra options]

Example:
  ./run_eval.sh \
    --dataset_root input_output_data/input/1_Batch2_Data_16spk_subset \
    --output_root input_output_data/output/experiments/1_Batch2_Data_16spk_subset \
    --nemo_manifest input_output_data/output/1_Batch2_Data_16spk_subset/nemo_asr_output/transcriptions.jsonl \
    --passages_csv input_output_data/input/oral_passages.csv
EOF
  exit 1
}

DATASET_ROOT=""
OUTPUT_ROOT=""
PASSAGES_CSV=""
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
    --passages_csv)
      PASSAGES_CSV="$2"
      EXTRA_ARGS+=("$1" "$2")
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

if [[ -z "$DATASET_ROOT" || -z "$OUTPUT_ROOT" || -z "$PASSAGES_CSV" ]]; then
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
