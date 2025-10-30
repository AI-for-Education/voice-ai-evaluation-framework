#!/usr/bin/env bash
set -euo pipefail

# Generate manifests for scoring offline using NeMo scripts (REF and CAN)

docker compose run --rm --entrypoint "" nemo-asr \
  python3 /work/tools/make_nemo_manifests.py \
    --egra_csv /io/input/egradata/egra_eval_2speakers.csv \
    --passages_csv /io/input/passages/oral_passages.csv \
    --textgrids_dir /io/input/audio_and_texgrid \
    --audio_root   /io/input/audio_and_texgrid \
    --nemo_hyp_manifest /io/input/nemo_asr_output/transcriptions.jsonl \
    --out_ref /io/input/nemo_asr_output/ref_manifest_norm.jsonl \
    --out_can /io/input/nemo_asr_output/can_manifest_norm.jsonl \
    --normalize_for_nemo

# Download NeMo ASR evaluation scripts if not already present
docker compose run --rm --entrypoint "" nemo-asr bash -lc '
  set -e
  mkdir -p /work/tools/nemo_examples/asr
  echo "Downloading NeMo ASR evaluation scripts..."
  wget -q https://raw.githubusercontent.com/NVIDIA-NeMo/NeMo/refs/tags/v2.4.1/examples/asr/speech_to_text_eval.py \
      -O /work/tools/nemo_examples/asr/speech_to_text_eval.py
  wget -q https://raw.githubusercontent.com/NVIDIA-NeMo/NeMo/refs/tags/v2.4.1/examples/asr/transcribe_speech.py \
      -O /work/tools/nemo_examples/asr/transcribe_speech.py
  echo "NeMo scripts are ready in /work/tools/nemo_examples/asr/"
'

# SCORE REF vs. HYP: Write scores per sample in manifest
docker compose run --rm --entrypoint "" nemo-asr \
  python3 /work/tools/nemo_examples/asr/speech_to_text_eval.py \
    dataset_manifest=/io/input/nemo_asr_output/ref_manifest_norm.jsonl \
    only_score_manifest=true \
    scores_per_sample=true \
    use_cer=false 

# SCORE CAN vs. HYP: Write scores per sample in manifest
docker compose run --rm --entrypoint "" nemo-asr \
  python3 /work/tools/nemo_examples/asr/speech_to_text_eval.py \
    dataset_manifest=/io/input/nemo_asr_output/can_manifest_norm.jsonl \
    only_score_manifest=true \
    scores_per_sample=true \
    use_cer=false 

echo ""
echo "Done! We have now per sample scores in the manifests using NeMo script:"
echo "   - /input_output_data/input/nemo_asr_output/ref_manifest_norm.jsonl"
echo "   - /input_output_data/input/nemo_asr_output/can_manifest_norm.jsonl"
echo ""

