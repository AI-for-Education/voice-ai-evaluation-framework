# BookBot Transformers inference backend

This directory contains offline Hugging Face CTC inference for pre-segmented WAV
files. Models are loaded only from local directories.

## Models and decoding units

- `bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa` emits phoneme/IPA units.
- `bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-word-lm` has an
  orthographic `a-z` vocabulary.

The current implementation intentionally uses greedy CTC decoding only. It
builds the feature extractor and tokenizer directly, so neither `pyctcdecode`
nor KenLM is required. It reads the processor sampling rate and resamples input
audio when necessary. Audio discovery, manifest loading, and resampling are
shared with NeMo through root `inference_common.py`.

Phoneme output is useful for inference diagnostics, but it must not be scored
directly against an orthographic reference using WER or MER. Use the
orthographic checkpoint for the primary text comparison.

## Run

```bash
./run_transformers_inference.sh \
  --model transformers_inference/models/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-word-lm \
  --audio_manifest input_output_data/output/experiments/<experiment>/manifests/ref_manifest.raw_segments.jsonl \
  --output_root input_output_data/output/<experiment>/transformers_asr_output_segments \
  --batch_size 8
```

Use `--root_audio_dir` instead to discover pre-segmented WAVs recursively.
BookBot uses the preferred `--audio_manifest` name and does not expose NeMo's
legacy inference aliases.

The backend automatically uses CUDA when PyTorch can access it and otherwise
uses CPU. It writes `<output_root>/transcriptions.jsonl` with
`audio_filepath`, `duration`, and `pred_text`, matching the NeMo handoff format.

Private data from another worktree can be mounted through a local uncommitted
`docker-compose.override.yml` without changing the repository Compose file.
