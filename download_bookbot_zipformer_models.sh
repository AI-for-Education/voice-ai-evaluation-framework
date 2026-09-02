#!/usr/bin/env bash
set -euo pipefail

# Recover only the files used by the six pending BookBot matrix profiles.
# Downloads are pinned to immutable Hugging Face revisions and copied from a
# short staging path to avoid Windows MAX_PATH failures in Hub metadata paths.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DRY_RUN=0
if [[ ${1:-} == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi
if [[ $# -ne 0 ]]; then
  echo "Usage: bash ./download_bookbot_zipformer_models.sh [--dry-run]" >&2
  exit 2
fi

if command -v hf >/dev/null 2>&1; then
  HF_COMMAND=(hf)
elif [[ -x "$SCRIPT_DIR/.venv_test/Scripts/hf.exe" ]]; then
  HF_COMMAND=("$SCRIPT_DIR/.venv_test/Scripts/hf.exe")
elif [[ -x "$SCRIPT_DIR/.venv_test/bin/hf" ]]; then
  HF_COMMAND=("$SCRIPT_DIR/.venv_test/bin/hf")
else
  echo "[ERROR] Hugging Face CLI 'hf' is required." >&2
  echo "        Install huggingface-hub==1.27.0, then rerun this script." >&2
  exit 1
fi

stage=""
cleanup() {
  if [[ -n "$stage" && "$stage" == "$SCRIPT_DIR"/.bookbot-download.* && -d "$stage" ]]; then
    rm -rf -- "$stage"
  fi
}
trap cleanup EXIT

if [[ $DRY_RUN -eq 1 ]]; then
  stage="$SCRIPT_DIR/.bookbot-download.<temporary>"
else
  stage="$(mktemp -d "$SCRIPT_DIR/.bookbot-download.XXXXXX")"
fi

download_snapshot_files() {
  local label="$1"
  local repo="$2"
  local revision="$3"
  local destination="$4"
  shift 4
  local files=("$@")
  local staging_directory="${stage}/${label}"
  local command=(
    "${HF_COMMAND[@]}" download "$repo" "${files[@]}"
    --revision "$revision"
    --local-dir "$staging_directory"
  )

  echo "[BOOKBOT] ${repo}@${revision}"
  if [[ $DRY_RUN -eq 1 ]]; then
    printf '  %q' "${command[@]}"
    printf '\n'
    return
  fi

  "${command[@]}"
  for relative_path in "${files[@]}"; do
    mkdir -p -- "$(dirname -- "$destination/$relative_path")"
    cp -f -- "$staging_directory/$relative_path" "$destination/$relative_path"
  done
}

download_snapshot_files \
  native \
  bookbot/zipformer-streaming-robust-sw-v4 \
  f27bc1620ac08c6a4bc6a1cf6072d592e7b09a49 \
  inference/torch/models/zipformer-streaming-robust-sw-v4 \
  exp-causal/jit_script_chunk_32_left_128.pt \
  data/lang_phone/tokens.txt

download_snapshot_files \
  onnx \
  bookbot/sherpa-onnx-zipformer-streaming-robust-sw-v4 \
  0e52da6c03294fd983f3a8621b32ac6a71b4787d \
  inference/sherpa_onnx/models/sherpa-onnx-zipformer-streaming-robust-sw-v4 \
  encoder-epoch-40-avg-7-chunk-16-left-128.int8.onnx \
  decoder-epoch-40-avg-7-chunk-16-left-128.int8.onnx \
  joiner-epoch-40-avg-7-chunk-16-left-128.int8.onnx \
  tokens.txt

download_snapshot_files \
  ort \
  bookbot/sherpa-onnx-ort-zipformer-streaming-robust-sw-v4 \
  311c41c8770242c02478d4569fcf5e0cd00c1218 \
  inference/sherpa_onnx/models/sherpa-onnx-ort-zipformer-streaming-robust-sw-v4 \
  encoder-epoch-40-avg-7-chunk-16-left-128.int8.ort \
  decoder-epoch-40-avg-7-chunk-16-left-128.int8.ort \
  joiner-epoch-40-avg-7-chunk-16-left-128.int8.ort \
  tokens.txt \
  required_operators.config

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[BOOKBOT] Dry run only; no files were downloaded or replaced."
  exit 0
fi

if command -v sha256sum >/dev/null 2>&1; then
  sha256sum --check --strict inference/bookbot_zipformer_artifacts.sha256
elif command -v shasum >/dev/null 2>&1; then
  shasum --algorithm 256 --check inference/bookbot_zipformer_artifacts.sha256
else
  echo "[ERROR] SHA-256 verification requires sha256sum or shasum." >&2
  exit 1
fi
echo "[BOOKBOT] All pinned BookBot artifacts are downloaded and verified."
