#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  ./run_eval2.sh --dataset_root PATH --manifest_in PATH [--output_root PATH] [extra options]

Example:
  ./run_eval2.sh \
    --dataset_root input_output_data/input/1_Batch2_Data_16spk_subset \
    --manifest_in input_output_data/output/evaluations/<model>_<timestamp>/manifests/ref_manifest.clean.jsonl

The default model-native evaluation is written under one of:
  input_output_data/output/evaluations/<model>_<timestamp>/orthographic/
  input_output_data/output/evaluations/<model>_<timestamp>/ipa/

Add --scoring_representation ipa for the parallel IPA/PER view under:
  input_output_data/output/evaluations/<model>_<timestamp>/ipa/

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

if [[ -z "$DATASET_ROOT" || -z "$MANIFEST_IN" ]]; then
  usage
fi

USER_FLAG=()
# Git Bash reports a Windows SID-derived UID that Docker Desktop cannot use to
# write bind-mounted files. Docker Desktop already maps those writes safely.
if [[ "${OS:-}" != "Windows_NT" ]]; then
  USER_FLAG=(--user "$(id -u):$(id -g)")
fi
OUTPUT_ARGS=()
if [[ -n "$OUTPUT_ROOT" ]]; then
  OUTPUT_ARGS=(--output_root "$OUTPUT_ROOT")
fi
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
    "${OUTPUT_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
