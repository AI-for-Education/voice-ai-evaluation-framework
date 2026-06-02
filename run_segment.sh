#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  ./run_segment.sh --textgrid_root PATH --manifest_in PATH --manifest_out PATH --segments_out_root PATH [extra options]

Example:
  ./run_segment.sh \
    --textgrid_root input_output_data/input/1_Batch2_Data_16spk_subset/0_IAR/2_TextGrid \
    --manifest_in input_output_data/output/1_Batch2_Data_16spk_subset/nemo_asr_output/transcriptions.jsonl \
    --manifest_out input_output_data/output/1_Batch2_Data_16spk_subset/nemo_asr_output/transcriptions_segments.jsonl \
    --segments_out_root input_output_data/output/1_Batch2_Data_16spk_subset/audio_segments
EOF
  exit 1
}

TEXTGRID_ROOT=""
SEGMENTS_OUT_ROOT=""
MANIFEST_IN=""
MANIFEST_OUT=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --textgrid_root|--textgrid-root)
      TEXTGRID_ROOT="$2"
      EXTRA_ARGS+=(--textgrid-root "$2")
      shift 2
      ;;
    --manifest_in|--manifest-in)
      MANIFEST_IN="$2"
      EXTRA_ARGS+=(--manifest-in "$2")
      shift 2
      ;;
    --manifest_out|--manifest-out)
      MANIFEST_OUT="$2"
      EXTRA_ARGS+=(--manifest-out "$2")
      shift 2
      ;;
    --segments_out_root|--segments-out-root)
      SEGMENTS_OUT_ROOT="$2"
      EXTRA_ARGS+=(--segments-out-root "$2")
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

if [[ -z "$TEXTGRID_ROOT" || -z "$SEGMENTS_OUT_ROOT" || -z "$MANIFEST_IN" || -z "$MANIFEST_OUT" ]]; then
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
  python3 /work/segment_manifests.py \
    "${EXTRA_ARGS[@]}"
