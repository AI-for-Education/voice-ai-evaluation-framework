# NeMo inference backend

This directory contains NVIDIA NeMo ASR inference. It supports dataset discovery,
optional TextGrid segmentation, and inference over already-segmented audio.

- `infer.py` loads a local `.nemo` model.
- `models/` contains ignored local model artifacts.
- `tmp/` contains temporary resampled or segmented audio.

Audio is normalized to 16 kHz before transcription. Model-specific CTC/RNNT
decoding behavior is otherwise unchanged by the backend separation. Audio
discovery, manifest loading, and resampling are shared with BookBot through
root `inference_common.py`.

Run the backend through its explicit launcher:

```bash
./run_nemo_inference.sh \
  --model nemo_inference/models/<model>.nemo \
  --audio_manifest input_output_data/output/experiments/<experiment>/manifests/ref_manifest.raw_segments.jsonl \
  --output_root input_output_data/output/<experiment>/nemo_asr_output_segments
```

Use `--root_audio_dir` instead to discover pre-segmented WAVs recursively.
`--dataset_root` remains available for legacy dataset discovery and optional
TextGrid segmentation. In older inference commands, `--manifest_in` is accepted
as an alias for `--audio_manifest`. Root `infer.py` and `run_inference.sh` remain
compatibility entrypoints.

Output is written to `<output_root>/transcriptions.jsonl` with
`audio_filepath`, `duration`, and `pred_text` fields.
