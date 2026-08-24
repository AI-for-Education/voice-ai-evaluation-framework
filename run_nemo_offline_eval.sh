#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

print_usage() {
  cat <<EOF
Usage:
  ./run_nemo_offline_eval.sh [OPTIONS]

Key options (forwarded to make_nemo_manifests.py):
  --dataset_root PATH        Dataset root containing 0_Audio/ and 2_TextGrid/.
  --dataset_annotator NAME   Annotator folder inside 2_TextGrid/ to use.
  --output_dir PATH          Directory where the normalized manifests will be written.
  --egra_csv PATH            Canonical EGRA CSV (overrides dataset discovery).
  --meta_csv PATH            Metadata CSV (optional for manifest creation).
  --textgrids_dir PATH       Explicit TextGrid directory.
  --audio_root PATH          Root containing audio folders.
  --nemo_hyp_manifest PATH   ASR manifest with predictions.
  --out_ref PATH             Output path for the REF vs HYP manifest.
  --out_can PATH             Output path for the CAN vs HYP manifest.

Any additional options are passed straight to make_nemo_manifests.py.

Note:
  Provide either --output_dir or --dataset_root (the latter defaults to <dataset_root>/nemo_asr_output).
EOF
}

# Parse CLI arguments to capture values we need later.
DATASET_ROOT=""
OUTPUT_DIR=""
OUT_REF=""
OUT_CAN=""
NEMO_MANIFEST=""

ARGS=("$@")

USER_FLAG=(--user "$(id -u):$(id -g)")
ENV_VARS=(
  --env HOME=/tmp
  --env MPLCONFIGDIR=/tmp/matplotlib
  --env NUMBA_CACHE_DIR=/tmp/numba_cache
  --env XDG_CACHE_HOME=/tmp/.cache
  --env LHOTSE_TOOLS_DIR=/tmp/lhotse_tools
  --env LHOTSE_DATA_HOME=/tmp/lhotse_data
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset_root)
      DATASET_ROOT="$2"; shift 2 ;;
    --output_dir)
      OUTPUT_DIR="$2"; shift 2 ;;
    --out_ref)
      OUT_REF="$2"; shift 2 ;;
    --out_can)
      OUT_CAN="$2"; shift 2 ;;
    --nemo_hyp_manifest)
      NEMO_MANIFEST="$2"; shift 2 ;;
    --help|-h)
      print_usage
      exit 0 ;;
    *)
      shift ;;
  esac
done

# Reset positional parameters so we can pass everything (including the original args)
# to make_nemo_manifests.py after we determine default paths.
set -- "${ARGS[@]}"

# Determine where manifests should be written.
if [[ -n "$OUTPUT_DIR" ]]; then
  TARGET_DIR="$OUTPUT_DIR"
elif [[ -n "$DATASET_ROOT" ]]; then
  TARGET_DIR="${DATASET_ROOT%/}/nemo_asr_output"
else
  echo "Error: please supply --dataset_root (and optionally --output_dir) so the target directory can be determined." >&2
  exit 1
fi

#mkdir -p "$TARGET_DIR"
if ! mkdir -p "$TARGET_DIR" 2>/dev/null; then
  if [[ "$TARGET_DIR" == /io/* ]]; then
    echo "Warning: could not create $TARGET_DIR on the host; assuming it will be created inside the container." >&2
  else
    echo "Error: failed to create output directory $TARGET_DIR" >&2
    exit 1
  fi
fi

REF_PATH="${OUT_REF:-$TARGET_DIR/ref_manifest_norm.jsonl}"
CAN_PATH="${OUT_CAN:-$TARGET_DIR/can_manifest_norm.jsonl}"
MANIFEST_PATH="${NEMO_MANIFEST:-$TARGET_DIR/transcriptions.jsonl}"

# ------------------------------------------------------------
# Step 1: Generate normalized manifests (REF and CAN)
# ------------------------------------------------------------
docker compose run --rm "${USER_FLAG[@]}" "${ENV_VARS[@]}" --entrypoint "" nemo-asr \
  python3 /work/tools/make_nemo_manifests.py \
    --normalize_for_nemo \
    --out_ref "$REF_PATH" \
    --out_can "$CAN_PATH" \
    --nemo_hyp_manifest "$MANIFEST_PATH" \
    "$@"

# ------------------------------------------------------------
# Step 2: Download evaluation scripts if necessary
# ------------------------------------------------------------
docker compose run --rm "${USER_FLAG[@]}" "${ENV_VARS[@]}" --entrypoint "" nemo-asr bash -lc '
  set -e
  mkdir -p /work/tools/nemo_examples/asr
  echo "Downloading NeMo ASR evaluation scripts..."
  wget -q https://raw.githubusercontent.com/NVIDIA-NeMo/NeMo/refs/tags/v2.4.1/examples/asr/speech_to_text_eval.py \
      -O /work/tools/nemo_examples/asr/speech_to_text_eval.py
  wget -q https://raw.githubusercontent.com/NVIDIA-NeMo/NeMo/refs/tags/v2.4.1/examples/asr/transcribe_speech.py \
      -O /work/tools/nemo_examples/asr/transcribe_speech.py
  echo "NeMo scripts are ready in /work/tools/nemo_examples/asr/"
'

# ------------------------------------------------------------
# Step 3: Score REF vs HYP
# ------------------------------------------------------------
docker compose run --rm "${USER_FLAG[@]}" "${ENV_VARS[@]}" --entrypoint "" nemo-asr \
  python3 /work/tools/nemo_examples/asr/speech_to_text_eval.py \
    dataset_manifest="$REF_PATH" \
    only_score_manifest=true \
    scores_per_sample=true \
    use_cer=false

# ------------------------------------------------------------
# Step 4: Score CAN vs HYP
# ------------------------------------------------------------
docker compose run --rm "${USER_FLAG[@]}" "${ENV_VARS[@]}" --entrypoint "" nemo-asr \
  python3 /work/tools/nemo_examples/asr/speech_to_text_eval.py \
    dataset_manifest="$CAN_PATH" \
    only_score_manifest=true \
    scores_per_sample=true \
    use_cer=false

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
echo
echo "Done! NeMo per-sample scores saved to:"
echo "  REF vs HYP -> $REF_PATH"
echo "  CAN vs HYP -> $CAN_PATH"
echo
