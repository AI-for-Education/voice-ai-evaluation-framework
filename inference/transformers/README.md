# Transformers inference

This package runs local Hugging Face speech-recognition checkpoints
without network access. It preserves the shared `transcriptions.jsonl` handoff
used by NeMo and the evaluation pipeline.

This directory is for models loaded through the Hugging Face **Transformers
inference library**, using PyTorch as the inference engine. A model being
downloaded from the Hugging Face Hub does not by itself make it a Transformers
model. The BookBot Zipformer artifact uses Sherpa-ONNX and ONNX Runtime, so it
lives under `inference/sherpa_onnx/` with its matching launcher.

Two inference families are available:

- `ctc`: `AutoModelForCTC` with either greedy framewise decoding or a profile-
  selected packaged KenLM decoder. Greedy covers BookBot, MMS, Wav2Vec2-BERT,
  Wav2Vec2, and HuBERT ASR checkpoints; the LM mode is currently a separate
  BookBot 5-gram profile.
- `speech_seq2seq`: `AutoModelForSpeechSeq2Seq` with `model.generate()`. This
  covers Paza and standard Hugging Face Whisper checkpoints.

## Local model storage

Profiles in `profiles/` are tracked, but model artifacts are not. Materialize a
complete Hub snapshot below `inference/transformers/models/` using the artifact
directory named in its profile. For example:

```text
inference/transformers/models/
├── mms-1b-all/
├── paza-whisper-large-v3-turbo/
└── wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-word-lm/
```

Inference is deliberately offline (`local_files_only: true`). Downloading and
licence acceptance are separate preparation steps. In Docker, the launcher
mounts the model directory at `/models` and sets `ASR_MODEL_ROOT=/models`.

Download a full snapshot on a networked preparation machine with the current
Hugging Face `hf` CLI. The `--local-dir` basename must exactly match the
profile's `artifact` value:

```bash
hf download microsoft/paza-whisper-large-v3-turbo \
  --local-dir inference/transformers/models/paza-whisper-large-v3-turbo

hf download facebook/mms-1b-all \
  --local-dir inference/transformers/models/mms-1b-all
```

Do not add `--include` filters for MMS: the local snapshot must contain the
`swh` adapter weights and tokenizer vocabulary as well as the base model.
Authentication, when a repository requires it, belongs in `hf auth login` or
the preparation environment—not in a profile. Inference itself never invokes
the Hub client to download weights.

## Run

```bash
./run_transformers_inference.sh \
  --inference_profile inference/transformers/profiles/paza-whisper-large-v3-turbo-sw.yaml \
  --audio_manifest input_output_data/output/experiments/<experiment>/manifests/ref_manifest.raw_segments.jsonl \
  --batch_size 8
```

The Paza owner requires caller-managed segmentation above the checkpoint's
448-token input limit but does not publish an equivalent duration. Both Paza
profiles therefore freeze a project policy of deterministic 30-second chunks
with zero overlap. Segmentation happens only in memory, chunk transcripts are
joined in source order, and the original file still emits one result row. This
evaluation input set has 44 of 7,617 clips above 30 seconds; the duration is a
reproducible
local choice, not an owner recommendation.

BookBot's greedy and packaged-LM runs are separate profile instances. Both use
the same runner and timestamped output structure, so neither can overwrite the
other:

```bash
# Existing plain-processor greedy CTC
./run_transformers_inference.sh \
  --inference_profile inference/transformers/profiles/bookbot-orthographic-ctc.yaml \
  --audio_manifest input_output_data/output/experiments/heldout_combined_fixed_20260525_exp41/manifests/ref_manifest.raw_segments.jsonl \
  --batch_size 8

# Packaged 5-gram KenLM beam search
./run_transformers_inference.sh \
  --inference_profile inference/transformers/profiles/bookbot-orthographic-ctc-5gram.yaml \
  --audio_manifest input_output_data/output/experiments/heldout_combined_fixed_20260525_exp41/manifests/ref_manifest.raw_segments.jsonl \
  --batch_size 8
```

### Parameter evidence

Each tracked YAML profile contains structured `parameter_evidence` records that
link result-affecting settings to model-owner documentation, pinned artifact
metadata, or an explicit project decision in this tracked README. Evidence is
copied into run metadata. The parser and `tests/test_profile.py` validate field
coverage and source paths.

### Paired greedy and non-greedy profiles

Decoder comparisons use the BookBot pattern: one frozen inference setup ID per decoder,
with no change to the input manifest contract or the emitted
`transcriptions.jsonl` schema. Because the runner includes the inference setup ID in
each timestamped transcript directory, greedy and non-greedy outputs cannot
overwrite or mix with one another.

| Model | Greedy baseline | Non-greedy structure |
| --- | --- | --- |
| OpenAI Whisper Large | `whisper-large-sw.yaml` (`num_beams: 1`) | `whisper-large-sw-beam5.yaml` (`num_beams: 5`) |
| OpenAI Whisper Large v2 | `whisper-large-v2-sw.yaml` (`num_beams: 1`) | `whisper-large-v2-sw-beam5.yaml` (`num_beams: 5`) |
| Paza Whisper Large v3 Turbo | `paza-whisper-large-v3-turbo-sw.yaml` (`num_beams: 1`) | `paza-whisper-large-v3-turbo-sw-beam5.yaml` (`num_beams: 5`) |

The decoder choices have primary-source backing:

- OpenAI's Whisper CLI uses beam size 5 by default at temperature zero. The
  Transformers generation API defines deterministic beam search as
  `num_beams > 1` with `do_sample: false`. The two OpenAI variants therefore
  override only those choices and retain the checkpoint's packaged generation
  defaults for other beam behaviour.
- Microsoft's Paza model card publishes a Beam-5 example with
  `early_stopping: true` and `length_penalty: 0.8`; the Paza variant freezes
  those exact values.

Sources: [OpenAI Whisper CLI](https://github.com/openai/whisper/blob/main/whisper/transcribe.py),
[Transformers generation strategies](https://huggingface.co/docs/transformers/main_classes/text_generation),
and [Microsoft Paza model card](https://huggingface.co/microsoft/paza-whisper-large-v3-turbo).

All variants retain the same model-specific input/output contract and long-audio
routing as their baseline. Completed greedy and beam runs remain separately
identified by inference setup ID and timestamp in the output tree.

MMS and W2V-BERT remain greedy-only in the tracked structure. A fair CTC
non-greedy profile requires a compatible decoder vocabulary and language-model
artifact; none is currently packaged with those local checkpoints. Such
profiles will be added only when those artifacts and their provenance are
fixed, rather than presenting an incomplete configuration as runnable.

For a one-file check, add `--smoke_test` and use a one-row manifest. The runner
creates
`input_output_data/output/smoke_tests/transcripts/bookbot-orthographic-ctc-5gram_<timestamp>/`
automatically, so no custom run-directory naming or overwrite handling is
needed.

Do not rerun segmentation for this comparison. Reuse the fixed 7,617-segment
manifest above, then evaluate each new transcript directory independently.
Never pass reference text, reading passages,
assessment vocabulary, or evaluation data as decoder hotwords. The packaged
`alpha`, `beta`, unknown-word offset, and boundary setting are kept fixed.
Beam width 100 and one-best output match the installed decoder defaults and are
explicit in the profile to prevent version drift; they must not be tuned on
held-out data.

Use `--audio_manifest` with the fixed segment manifest so every comparison keeps
the same input set and order. Reserve `--root_audio_dir` for ad hoc discovery
when no ordered manifest exists. The default output is
`input_output_data/output/transcripts/<inference-setup-id>_<YYYY_MM_DD_HH_MM_SS_UTC>/`. Pass
`--smoke_test` to use
`input_output_data/output/smoke_tests/transcripts/<inference-setup-id>_<YYYY_MM_DD_HH_MM_SS_UTC>/`.
During inference the CLI shows a file progress bar and prints the resolved
device, batch size, output directory, final result count and error count.

## Model-specific decoding behaviour

These models stay in the same Transformers folder, launcher, and CUDA image; the
profiles select the differences that affect model input or decoding:

- Every Whisper profile returns the direct decoded model hypothesis in
  `pred_text`. No duration or repeated-phrase rule alters it after decoding.

- Both Paza profiles retain the packaged empty suppression-token list instead
  of introducing locally selected language-token suppression.
- OpenAI Whisper Large and Whisper Large v2 keep the existing short-form path
  for clips up to and including 30 seconds. Longer clips follow Hugging Face's
  native timestamp-based long-form path: feature extraction is untruncated,
  padding is to the longest clip in the long-form batch, an attention mask is
  returned, and `model.generate()` receives `return_timestamps=true`.
  `do_sample=false` is explicit in both profiles for reproducibility. This
  routing happens in memory and still emits one transcript row for the
  original audio path; source audio is never segmented or modified on disk.
- Paza does not enable this timestamp path because its model card directs
  callers to segment long inputs and does not document long-form timestamps.
  The two Paza profiles instead segment overlength inputs into deterministic
  30-second, zero-overlap chunks in memory, decode each chunk with timestamps
  disabled, and join text in source order.

Every active value is stored in the YAML profile and copied to
`run_metadata.json`, together with the effective generation configuration and
the defaults supplied by the model or inference library.

## Profiles

| Profile | Source checkpoint | Notes |
| --- | --- | --- |
| `bookbot-orthographic-ctc.yaml` | `bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-word-lm` | Orthographic; explicit plain processor preserves greedy no-KenLM behaviour. |
| `bookbot-orthographic-ctc-5gram.yaml` | `bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-word-lm` | Orthographic; packaged 5-gram KenLM beam search. Explicit beam width 100 and one-best output match the installed decoder defaults; one persistent worker is the deterministic project execution choice. Requires `pyctcdecode==0.5.0` and `kenlm==0.3.0` from the current Dockerfile. |
| `bookbot-phoneme-ctc.yaml` | `bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-alphabets-phonemes-bookbot` | Phoneme output; do not score directly as orthographic WER. |
| `mms-1b-all-swh.yaml` | `facebook/mms-1b-all` | Selects and loads the `swh` language adapter. |
| `w2v-bert-2.0-swahili-asr.yaml` | `badrex/w2v-bert-2.0-swahili-asr` | Automatic Wav2Vec2-BERT processor. |
| `paza-whisper-large-v3-turbo-sw.yaml` | `microsoft/paza-whisper-large-v3-turbo` | Greedy (`num_beams: 1`); project-scoped 30-second sequential chunking above the owner-documented token limit. |
| `paza-whisper-large-v3-turbo-sw-beam5.yaml` | `microsoft/paza-whisper-large-v3-turbo` | Beam-5 decoder with the same project-scoped sequential chunking policy. |
| `whisper-large-sw.yaml` | `openai/whisper-large` | Greedy (`num_beams: 1`); deterministic short form through 30 seconds and native timestamp-based long form above 30 seconds. |
| `whisper-large-sw-beam5.yaml` | `openai/whisper-large` | Beam-5 comparison structure; same input/output and long-form routing; not yet run. |
| `whisper-large-v2-sw.yaml` | `openai/whisper-large-v2` | Greedy (`num_beams: 1`); deterministic short form through 30 seconds and native timestamp-based long form above 30 seconds. |
| `whisper-large-v2-sw-beam5.yaml` | `openai/whisper-large-v2` | Beam-5 comparison structure; same input/output and long-form routing; not yet run. |
| `hubert-large-ls960-ft-en.yaml` | `facebook/hubert-large-ls960-ft` | English-only profile; the pinned artifact is not installed locally. |

CUDA is selected automatically when available. Explicit half-precision profile
settings fall back to float32 on CPU, and every effective device/dtype setting
is recorded in `run_metadata.json`. LM runs additionally record the decoding
strategy, decoder class, LM path and byte size, every beam/LM setting, worker
count, and the installed pyctcdecode and KenLM versions.

Every artifact directory currently present in `models/` has a corresponding
baseline profile above that passed a one-file GPU smoke transcription on 12
August 2026. The new Beam-5 comparison profiles have intentionally not been
run. The Wav2Vec2 and HuBERT profiles are also ready, but their artifact
directories must be downloaded before they can be loaded.
