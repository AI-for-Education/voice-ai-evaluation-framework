#!/usr/bin/env bash
set -euo pipefail

mkdir -p input_output_data/output

docker compose run --rm --entrypoint "" \
  egra-eval \
  python3 /work/evaluation.py \
    --egra_csv /io/input/egradata/egra_eval_2speakers.csv \
    --meta_csv /io/input/egradata/Student_Dummy_MetaData_EGRA_030925.csv \
    --passages_csv /io/input/passages/oral_passages.csv \
    --nemo_manifest /io/input/nemo_asr_output/transcriptions.jsonl \
    --out_csv /io/output/egra_eval_detailed.csv \
    --summary_csv /io/output/egra_eval_summary.csv

