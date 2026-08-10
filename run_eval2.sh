#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  ./run_eval2.sh --dataset_root PATH --manifest_in PATH [--output_root PATH] [extra options]

Example:
  ./run_eval2.sh \
    --dataset_root input_output_data/input/1_Batch2_Data_16spk_subset \
    --output_root input_output_data/output/experiments/1_Batch2_Data_16spk_subset \
    --manifest_in input_output_data/output/experiments/1_Batch2_Data_16spk_subset/manifests/ref_manifest.segment.clean.jsonl
EOF
  exit 1
}

DATASET_ROOT=""
OUTPUT_ROOT=""
MANIFEST_IN=""
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
    --manifest_in)
      MANIFEST_IN="$2"
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

# If output root not provided, default to input_output_data/output/experiments/exp_{datetime}
if [[ -z "$OUTPUT_ROOT" ]]; then
  DEFAULT_DIR="input_output_data/output/experiments/exp_$(date '+%Y_%m_%d_%H_%M_%S')"
  echo "No --output_root supplied. Using default: $DEFAULT_DIR"
  OUTPUT_ROOT="$DEFAULT_DIR"
fi

if [[ -z "$DATASET_ROOT" || -z "$MANIFEST_IN" ]]; then
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
  python3 /work/eval_pipeline2.py \
    --dataset_root "$DATASET_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    "${EXTRA_ARGS[@]}"
