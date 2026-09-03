#!/usr/bin/env bash
set -euo pipefail

# Prevent Git Bash/MSYS from rewriting Linux container paths such as /work.
export MSYS_NO_PATHCONV="${MSYS_NO_PATHCONV:-1}"

usage() {
  cat <<'EOF' >&2
Usage:
  ./run_make_ref_manifest.sh --dataset_root PATH --passages_csv PATH --output_jsonl PATH [extra options]

Example:
  ./run_make_ref_manifest.sh \
    --dataset_root input_output_data/input/1_Batch2_Data_16spk_subset \
    --passages_csv input_output_data/input/oral_passages.csv \
    --output_jsonl input_output_data/output/experiments/1_Batch2_Data_16spk_subset/manifests/ref_manifest.raw.jsonl
EOF
  exit 1
}

DATASET_ROOT=""
PASSAGES_CSV=""
OUTPUT_JSONL=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset_root)
      DATASET_ROOT="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --passages_csv)
      PASSAGES_CSV="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --output_jsonl)
      OUTPUT_JSONL="$2"
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

if [[ -z "$DATASET_ROOT" || -z "$PASSAGES_CSV" || -z "$OUTPUT_JSONL" ]]; then
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
  --workdir /work \
  egra-eval \
  python3 -m tools.make_ref_manifest \
    "${EXTRA_ARGS[@]}"
