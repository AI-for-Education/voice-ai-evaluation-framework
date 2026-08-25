#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  ./run_manifest.sh --dataset_root PATH --audio_manifest PATH [extra options]

Example:
  ./run_manifest.sh \
    --dataset_root input_output_data/input/1_Batch2_Data_16spk_subset \
    --audio_manifest input_output_data/output/experiments/1_Batch2_Data_16spk_subset/manifests/ref_manifest.raw_segments.jsonl \
    --prediction_manifest input_output_data/output/transcripts/<model>_<timestamp>/transcriptions.jsonl

With the standard transcript path, output is inferred as:
  input_output_data/output/evaluations/<model>_<timestamp>/manifests/

Any model run with declared output units triggers generation/reuse of the dataset IPA reference view.
EOF
  exit 1
}

DATASET_ROOT=""
OUTPUT_ROOT=""
AUDIO_MANIFEST=""
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
    --audio_manifest|--manifest_base_in)
      AUDIO_MANIFEST="$2"
      EXTRA_ARGS+=(--audio_manifest "$2")
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

if [[ -z "$DATASET_ROOT" || -z "$AUDIO_MANIFEST" ]]; then
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
  python3 /work/manifest_pipeline.py \
    --dataset_root "$DATASET_ROOT" \
    "${OUTPUT_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
