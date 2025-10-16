#!/usr/bin/env bash
set -euo pipefail

mkdir -p input_output_data/input/nemo_asr_output
mkdir -p nemo_inference/tmp

docker compose run --rm --entrypoint "" \
  nemo-asr \
  python3 /work/infer.py \
    --model /models/Swahili_exp1_100epochs.nemo \
    --root_audio_dir /io/input/audio_and_texgrid \
    --output_manifest /io/input/nemo_asr_output/transcriptions.jsonl \
    --tmp_dir /tmp_segments \
    --tier_name child \
    --textgrid_keyword passage \
    --debug

# If you want GPU and have the runtime, just add --gpus all in the docker compose run command


