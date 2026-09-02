#!/usr/bin/env bash
set -euo pipefail

# Runs the six unfinished BookBot configurations from workbook rows 30-35.
# BookBot is promptless transducer ASR: profiles select artifacts and decoders;
# audio is the only model input.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

FULL_MANIFEST="input_output_data/output/experiments/heldout_combined_fixed_20260525_exp41/manifests/ref_manifest.raw_segments.jsonl"
SMOKE_MANIFEST="input_output_data/output/Z_ignored/structure_smoke_test/one_file.jsonl"
MANIFEST="$FULL_MANIFEST"
OUTPUT_ROOT="input_output_data/output"
SHERPA_BATCH_SIZE=8
TORCH_BATCH_SIZE=1
NUM_THREADS=2
SMOKE_TEST=0
DRY_RUN=0
BUILD_IMAGE=1
CUSTOM_MANIFEST=0

usage() {
  cat <<'EOF'
Usage: bash ./run_bookbot_zipformer_matrix.sh [options]

Runs the six pending BookBot Zipformer profiles in workbook-row order:
native TorchScript greedy/beam4, INT8 ONNX greedy/beam4, then INT8 ORT
greedy/beam4.

Options:
  --audio-manifest PATH      Override the frozen full benchmark manifest with
                             a repo-relative path (exclusive with --smoke-test).
  --smoke-test               Use the tracked one-file smoke manifest and write
                             under output/smoke_tests.
  --output-root PATH         Repo-relative output root (default:
                             input_output_data/output).
  --sherpa-batch-size N      Batch size for ONNX/ORT runs (default: 8).
  --torch-batch-size N       Runner batch size for native runs (default: 1).
  --num-threads N            Runtime thread count (default: 2).
  --skip-build               Reuse the existing shared ASR image without
                             rebuilding it first.
  --dry-run                  Print the image-build and six inference commands
                             without building or starting models.
  -h, --help                 Show this help.

Examples:
  bash ./run_bookbot_zipformer_matrix.sh --smoke-test
  bash ./run_bookbot_zipformer_matrix.sh
  bash ./run_bookbot_zipformer_matrix.sh --dry-run
EOF
}

require_positive_integer() {
  local label="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] ${label} must be a positive integer; got: ${value}" >&2
    exit 2
  fi
}

require_repo_relative_path() {
  local label="$1"
  local value="$2"
  if [[ "$value" == /* || "$value" =~ ^[A-Za-z]: || "$value" == *\\* || "/$value/" == *"/../"* ]]; then
    echo "[ERROR] ${label} must be a forward-slash, repo-relative path; got: ${value}" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --audio-manifest)
      [[ $# -ge 2 ]] || { echo "[ERROR] --audio-manifest requires a path" >&2; exit 2; }
      MANIFEST="$2"
      CUSTOM_MANIFEST=1
      shift 2
      ;;
    --smoke-test)
      SMOKE_TEST=1
      shift
      ;;
    --output-root)
      [[ $# -ge 2 ]] || { echo "[ERROR] --output-root requires a path" >&2; exit 2; }
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --sherpa-batch-size)
      [[ $# -ge 2 ]] || { echo "[ERROR] --sherpa-batch-size requires a value" >&2; exit 2; }
      SHERPA_BATCH_SIZE="$2"
      shift 2
      ;;
    --torch-batch-size)
      [[ $# -ge 2 ]] || { echo "[ERROR] --torch-batch-size requires a value" >&2; exit 2; }
      TORCH_BATCH_SIZE="$2"
      shift 2
      ;;
    --num-threads)
      [[ $# -ge 2 ]] || { echo "[ERROR] --num-threads requires a value" >&2; exit 2; }
      NUM_THREADS="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --skip-build)
      BUILD_IMAGE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ $SMOKE_TEST -eq 1 && $CUSTOM_MANIFEST -eq 1 ]]; then
  echo "[ERROR] --smoke-test and --audio-manifest are mutually exclusive." >&2
  exit 2
fi
if [[ $SMOKE_TEST -eq 1 ]]; then
  MANIFEST="$SMOKE_MANIFEST"
fi

require_positive_integer "--sherpa-batch-size" "$SHERPA_BATCH_SIZE"
require_positive_integer "--torch-batch-size" "$TORCH_BATCH_SIZE"
require_positive_integer "--num-threads" "$NUM_THREADS"
require_repo_relative_path "--audio-manifest" "$MANIFEST"
require_repo_relative_path "--output-root" "$OUTPUT_ROOT"

required_files=(
  "$MANIFEST"
  "docker-compose.yml"
  "docker/Dockerfile"
  "inference/bookbot_zipformer_artifacts.sha256"
  "run_torch_inference.sh"
  "run_sherpa_onnx_inference.sh"
  "inference/torch/profiles/zipformer-streaming-robust-sw-v4-torchscript.yaml"
  "inference/torch/profiles/zipformer-streaming-robust-sw-v4-torchscript-modified-beam4.yaml"
  "inference/sherpa_onnx/profiles/zipformer-streaming-robust-sw-v4-onnx-int8.yaml"
  "inference/sherpa_onnx/profiles/zipformer-streaming-robust-sw-v4-onnx-int8-modified-beam4.yaml"
  "inference/sherpa_onnx/profiles/zipformer-streaming-robust-sw-v4-ort-int8.yaml"
  "inference/sherpa_onnx/profiles/zipformer-streaming-robust-sw-v4-ort-int8-modified-beam4.yaml"
  "inference/torch/models/zipformer-streaming-robust-sw-v4/exp-causal/jit_script_chunk_32_left_128.pt"
  "inference/torch/models/zipformer-streaming-robust-sw-v4/data/lang_phone/tokens.txt"
  "inference/sherpa_onnx/models/sherpa-onnx-zipformer-streaming-robust-sw-v4/encoder-epoch-40-avg-7-chunk-16-left-128.int8.onnx"
  "inference/sherpa_onnx/models/sherpa-onnx-zipformer-streaming-robust-sw-v4/decoder-epoch-40-avg-7-chunk-16-left-128.int8.onnx"
  "inference/sherpa_onnx/models/sherpa-onnx-zipformer-streaming-robust-sw-v4/joiner-epoch-40-avg-7-chunk-16-left-128.int8.onnx"
  "inference/sherpa_onnx/models/sherpa-onnx-zipformer-streaming-robust-sw-v4/tokens.txt"
  "inference/sherpa_onnx/models/sherpa-onnx-ort-zipformer-streaming-robust-sw-v4/encoder-epoch-40-avg-7-chunk-16-left-128.int8.ort"
  "inference/sherpa_onnx/models/sherpa-onnx-ort-zipformer-streaming-robust-sw-v4/decoder-epoch-40-avg-7-chunk-16-left-128.int8.ort"
  "inference/sherpa_onnx/models/sherpa-onnx-ort-zipformer-streaming-robust-sw-v4/joiner-epoch-40-avg-7-chunk-16-left-128.int8.ort"
  "inference/sherpa_onnx/models/sherpa-onnx-ort-zipformer-streaming-robust-sw-v4/tokens.txt"
  "inference/sherpa_onnx/models/sherpa-onnx-ort-zipformer-streaming-robust-sw-v4/required_operators.config"
)

missing=()
for path in "${required_files[@]}"; do
  [[ -s "$path" ]] || missing+=("$path")
done
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "[ERROR] BookBot preflight found missing or empty required files:" >&2
  printf '  - %s\n' "${missing[@]}" >&2
  echo "[ERROR] Recover pinned model files with:" >&2
  echo "        bash ./download_bookbot_zipformer_models.sh" >&2
  exit 1
fi

if command -v sha256sum >/dev/null 2>&1; then
  checksum_command=(
    sha256sum --check --strict inference/bookbot_zipformer_artifacts.sha256
  )
elif command -v shasum >/dev/null 2>&1; then
  checksum_command=(
    shasum --algorithm 256 --check inference/bookbot_zipformer_artifacts.sha256
  )
else
  echo "[ERROR] SHA-256 verification requires sha256sum or shasum." >&2
  exit 1
fi
if ! "${checksum_command[@]}"; then
  echo "[ERROR] Recover the exact pinned payloads with:" >&2
  echo "        bash ./download_bookbot_zipformer_models.sh" >&2
  exit 1
fi
echo "[BOOKBOT] Pinned artifact checksums passed."

profiles=(
  "torch|inference/torch/profiles/zipformer-streaming-robust-sw-v4-torchscript.yaml|${TORCH_BATCH_SIZE}|auto"
  "torch|inference/torch/profiles/zipformer-streaming-robust-sw-v4-torchscript-modified-beam4.yaml|${TORCH_BATCH_SIZE}|auto"
  "sherpa|inference/sherpa_onnx/profiles/zipformer-streaming-robust-sw-v4-onnx-int8.yaml|${SHERPA_BATCH_SIZE}|auto"
  "sherpa|inference/sherpa_onnx/profiles/zipformer-streaming-robust-sw-v4-onnx-int8-modified-beam4.yaml|${SHERPA_BATCH_SIZE}|auto"
  "sherpa|inference/sherpa_onnx/profiles/zipformer-streaming-robust-sw-v4-ort-int8.yaml|${SHERPA_BATCH_SIZE}|cpu"
  "sherpa|inference/sherpa_onnx/profiles/zipformer-streaming-robust-sw-v4-ort-int8-modified-beam4.yaml|${SHERPA_BATCH_SIZE}|cpu"
)

if [[ $DRY_RUN -eq 0 ]] && ! command -v docker >/dev/null 2>&1; then
  if command -v cygpath >/dev/null 2>&1 && [[ -n "${LOCALAPPDATA:-}" ]]; then
    docker_dir="$(cygpath -u "$LOCALAPPDATA")/Programs/DockerDesktop/resources/bin"
    if [[ -x "${docker_dir}/docker.exe" ]]; then
      export PATH="${docker_dir}:${PATH}"
    fi
  fi
fi
if [[ $DRY_RUN -eq 0 ]]; then
  command -v docker >/dev/null 2>&1 || {
    echo "[ERROR] docker is not on PATH; start Docker Desktop first." >&2
    exit 1
  }
  docker info >/dev/null 2>&1 || {
    echo "[ERROR] Docker Desktop is not ready; start it and rerun this script." >&2
    exit 1
  }
fi

echo "[BOOKBOT] Manifest: ${MANIFEST}"
echo "[BOOKBOT] Output root: ${OUTPUT_ROOT}"
echo "[BOOKBOT] Runs: ${#profiles[@]}"
[[ $DRY_RUN -eq 1 ]] && echo "[BOOKBOT] Dry run only; no model will start."

if [[ $BUILD_IMAGE -eq 1 ]]; then
  build_command=(docker compose build torch-asr)
  echo
  echo "[BOOKBOT] Preparing the shared ASR image used by both launchers."
  if [[ $DRY_RUN -eq 1 ]]; then
    printf '  %q' "${build_command[@]}"
    printf '\n'
  else
    "${build_command[@]}"
  fi
fi

for spec in "${profiles[@]}"; do
  IFS='|' read -r runtime profile batch_size provider <<<"$spec"
  if [[ "$runtime" == "torch" ]]; then
    launcher="./run_torch_inference.sh"
  else
    launcher="./run_sherpa_onnx_inference.sh"
  fi
  command=(
    bash "$launcher"
    --inference_profile "$profile"
    --audio_manifest "$MANIFEST"
    --output_root "$OUTPUT_ROOT"
    --batch_size "$batch_size"
    --num_threads "$NUM_THREADS"
    --fail_on_error
  )
  [[ "$provider" != "auto" ]] && command+=(--provider "$provider")
  [[ $SMOKE_TEST -eq 1 ]] && command+=(--smoke_test)

  echo
  echo "[BOOKBOT] ${profile}"
  if [[ $DRY_RUN -eq 1 ]]; then
    printf '  %q' "${command[@]}"
    printf '\n'
  else
    "${command[@]}"
  fi
done

echo
if [[ $DRY_RUN -eq 1 ]]; then
  echo "[BOOKBOT] Dry-run preflight passed for all six profiles."
else
  echo "[BOOKBOT] All six inference commands completed. Review each run_metadata.json for output.errors == 0."
fi
