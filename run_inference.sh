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

# Detect host NVIDIA GPU and enable docker GPU flag when available.
# This makes the script automatically pass GPU access to the container when
# the host has NVIDIA drivers and the container runtime supports it.
DOCKER_GPU_FLAG=()
if command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia-smi >/dev/null 2>&1; then
    # If the host has an NVIDIA GPU, ask Docker to expose GPUs to the container.
    # Note: this requires the host to have the NVIDIA container toolkit installed
    # (nvidia-container-toolkit) and a recent Docker / Compose that supports gpus.
    echo "[INFO] NVIDIA GPU detected on host - enabling Docker GPU access"
    DOCKER_GPU_FLAG=(--gpus all)
  fi
fi

# If GPU access was requested, ensure the image is built with a CUDA-enabled PyTorch
# backend. The Dockerfile supports a build-arg TORCH_CUDA (cpu|cu121). Building with
# cu121 will install CUDA-enabled PyTorch wheels in the image.
if [ "${#DOCKER_GPU_FLAG[@]}" -gt 0 ]; then
  if [ -n "${NO_DOCKER_BUILD-}" ]; then
    echo "[INFO] NO_DOCKER_BUILD is set - skipping docker image rebuild (would build with TORCH_CUDA=cu121)"
  else
    echo "[INFO] Building Docker image with CUDA-enabled PyTorch (TORCH_CUDA=cu121). This may take a while..."
    docker compose build --build-arg TORCH_CUDA=cu121 nemo-asr
  fi
fi

docker compose run --rm "${USER_FLAG[@]}" "${ENV_VARS[@]}" --entrypoint "" \
  nemo-asr \
  python3 /work/infer.py \
    --model "$MODEL_PATH" \
    --dataset_root "$DATASET_ROOT" \
    --output_root "$OUTPUT_DIR" \
    --debug \
    "${EXTRA_ARGS[@]}"
