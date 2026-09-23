# NeMo inference

This directory contains NVIDIA NeMo ASR inference while the shared input,
result, profile, and runner contracts live one level above it.

- `backend.py` restores a local `.nemo` artifact and preserves CTC, RNNT, and
  hybrid decoder selection.
- `infer.py` provides the profile-driven inference command and the legacy API
  used by root `infer.py`.
- `profiles/` contains portable, tracked model descriptions.
- `models/` contains ignored local model artifacts.
- `tmp/` contains ignored resampled or segmented audio.

## Profile-driven inference

Place the Exp41 model at:

```text
inference/nemo/models/model_exp41_avg.nemo
```

Then run:

```bash
./run_nemo_inference.sh \
  --inference_profile inference/nemo/profiles/swahili-exp41-ctc.yaml \
  --audio_manifest input_output_data/output/experiments/<experiment>/manifests/ref_manifest.raw_segments.jsonl
```

The default output is
`input_output_data/output/transcripts/<inference-setup-id>_<YYYY_MM_DD_HH_MM_SS_UTC>/`. Pass
`--smoke_test` to use
`input_output_data/output/smoke_tests/transcripts/<inference-setup-id>_<YYYY_MM_DD_HH_MM_SS_UTC>/`.
The shared runner prints file progress, device, batch size, resolved output and
final success/error counts.

Use `--root_audio_dir` instead to discover pre-segmented WAV files recursively.
The inference command writes both `transcriptions.jsonl` and
`run_metadata.json`. Input order and the existing `audio_filepath`, `duration`,
and `pred_text` handoff are preserved; failures add an optional `error` field.

Inference is offline. Profiles contain model-root-relative artifacts and never
download them. Set `ASR_MODEL_ROOT=/models` when a container mounts the NeMo
model directory at `/models`.

## Legacy compatibility

Root `infer.py` remains the only legacy interface. It accepts `--model`,
`--dataset_root`, optional TextGrid discovery/segmentation, `--manifest_in`, and
the existing decoder/execution flags. New inference commands use
`--inference_profile` instead of `--model`.

Audio is normalized to 16 kHz before transcription. The profile-driven adapter
reports unreadable inputs per file. Legacy TextGrid behaviour and temporary-file
cleanup remain unchanged.
