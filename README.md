## Overview

The purpose of this project is to evaluate locally deployed ASR models on the task of early grade reading assessments (EGRA) for Kiswahili child speech. Inference is currently supported through NVIDIA NeMo, Hugging Face Transformers, ONNX Runtime, Sherpa-ONNX, and a dedicated multimodal generation flow.

The repository uses five primary terms: **model**, **inference setup**,
**execution stack**, **run**, and **evaluation**. A **benchmark** is a
standardized evaluation procedure, not another name for a model or setup. See
the authoritative [vocabulary bank](docs/vocabulary.md) before adding new
workflow terminology.

**Input (you need to provide)**

- A local ASR model artifact and its tracked inference profile. A first NeMo model is [provided here](https://drive.google.com/file/d/1NQTC8532QluX7KXQNGcebKj9FseUzrO-), and the broader candidate list is documented in the [ASR benchmark candidate-model spreadsheet](https://www.dropbox.com/scl/fi/t4f76nnj14n7gog2zm6du/ASR_benchmark_candidate_models.xlsx?rlkey=62h00z8ks4aan9k70vv03cp7u&st=u2zd2pnu&dl=0).

- Dataset of Kiswahili child speech comprising:
  - audio files, 
  - canonical texts, i.e. what the child should have uttered and
  - reference text, i.e. what the child actually uttered

**Output**

Given the above, evaluation will be performed, producing a final `egra_eval_summary.txt` report.

## Inference routes

Inference is organized under `inference/` by inference library or model-family
adapter while sharing the same audio, manifest, result, and evaluation
contracts:

- `inference/nemo/` contains NVIDIA NeMo inference and its profiles.
- `inference/transformers/` contains offline Hugging Face CTC and speech-seq2seq inference.
- `inference/sherpa_onnx/` contains offline Sherpa-ONNX streaming-transducer inference.
- `inference/torch/` contains native TorchScript streaming-transducer inference.
- `inference/onnxruntime/` contains PC CPU validation of FP32 and INT8 exports plus a separate, controlled Android-parity accuracy proxy for the packaged mobile artifact.
- `inference/multimodal/` contains prompt-driven audio-to-text generation for multimodal models, beginning with Gemma 4 E2B.
- `inference/common.py`, `inference/contracts.py`, and `inference/runner.py` provide the shared input, audio, result, and output behaviour.

Segment audio once with `run_segment.sh`, then pass the same segment manifest
to any inference route. All routes write unchanged `transcriptions.jsonl` rows
with `audio_filepath`, `duration`, and `pred_text`, plus `run_metadata.json`
describing the inference setup and execution stack. Use
`run_nemo_inference.sh`, `run_transformers_inference.sh`,
`run_onnxruntime_inference.sh`, `run_onnxruntime_android_inference.sh`,
`run_sherpa_onnx_inference.sh`, `run_torch_inference.sh`, or
`run_multimodal_inference.sh` for new runs;
root `infer.py` is the only legacy NeMo `--model` API. See the README in each
inference-library directory for model-specific details.

`run_metadata.json` records the normalized inference profile and canonical
hash, pinned model source plus available artifact checksums, completed adapter
state, exact decoder/generation call arguments, execution-stack versions,
command-line
invocation, Python warnings from initialization and transcription, and
Git/source-tree identity. Generative adapters separate
profile-requested values, packaged model defaults, actual call arguments, and
their effective merged configuration.
Profiles also reference the tracked pipeline definitions in
`inference/pipeline_contracts.json`. New metadata resolves them into four
readable evidence groups: model artifact, inference setup, execution stack,
and evaluation. Unavailable observations use an explicit `not_available`
value with a reason instead of guessing. Extended notes may be kept locally in
`docs/pipeline-provenance.md`; that file is local-development documentation and
is not versioned.
New inference runs score the direct adapter hypothesis in `pred_text`; no
duration or repetition rule truncates model output after decoding.

### Standard output layout

New profile-driven runs use one inference-setup/timestamp name throughout the pipeline:

```text
input_output_data/output/
├─ transcripts/
│  └─ <inference-setup-id>_<YYYY_MM_DD_HH_MM_SS_UTC>/
│     ├─ transcriptions.jsonl
│     └─ run_metadata.json
├─ evaluations/
│  └─ <inference-setup-id>_<YYYY_MM_DD_HH_MM_SS_UTC>/
│     ├─ manifests/
│     ├─ orthographic/             # Valid orthographic WER result
│     │  ├─ egra_eval_detailed.csv
│     │  ├─ egra_eval_summary.txt
│     │  └─ evaluation_metadata.json
│     ├─ ipa/                      # IPA results stay isolated by exact G2P identity
│     │  └─ <g2p_system_id>/
│     │     ├─ egra_eval_detailed.csv
│     │     ├─ egra_eval_summary.txt
│     │     └─ evaluation_metadata.json
└─ smoke_tests/
   ├─ transcripts/
   │  └─ <inference-setup-id>_<YYYY_MM_DD_HH_MM_SS_UTC>/
   └─ evaluations/
      └─ <inference-setup-id>_<YYYY_MM_DD_HH_MM_SS_UTC>/
```

Inference creates the inference-setup/UTC-timestamp run ID automatically. Runs
are written to a dot-prefixed staging directory and published only after the
transcript and metadata are complete. If the same inference setup starts more
than once in the same second, later runs receive `__2`, `__3`, and so on instead
of reusing an existing directory. `run_manifest.sh`
derives the matching evaluation directory from the standard
`--prediction_manifest` path. `run_eval2.sh` writes each scoring representation into
its own child directory, so WER and PER results cannot replace one another.
`auto` writes orthographic models under `orthographic/` and native phoneme
models under `ipa/<g2p_system_id>/`. Cross-representation
phoneme-vs-orthography WER is not a
valid evaluation route and is rejected.
Use `--smoke_test` on any current inference launcher for the smoke-test branch.
`--output_root` remains available when a different output base or an explicit
evaluation destination is required.

All inference routes print model-loading milestones and a shared file
progress bar. The bar advances after each completed batch and ends with result
and error counts; the exact output path is also printed.

| Stage | Argument | File role |
|---|---|---|
| Current inference routes | `--audio_manifest` | Exact audio segments to transcribe |
| Manifest merge audio input | `--audio_manifest` | Exact audio rows used for inference |
| Manifest merge predictions | `--prediction_manifest` | Adapter predictions in `transcriptions.jsonl` |
| Evaluation input | `--manifest_in` | Cleaned manifest containing references and predictions |

The root/original NeMo command is the sole compatibility exception and also
accepts its established `--manifest_in` spelling.

Run the same merge and evaluation steps once per inference run. Report each
run's metrics separately; phoneme-output WER/MER is not directly comparable
with orthographic-output WER/MER because the scored units are different.

## Inference setup readiness

The status below records local artifact checks and one-file inference smoke tests
performed on 12-13 August 2026. The shared
`voice-ai-evaluation-framework-asr:latest` image has been rebuilt from the
tracked CUDA 12.8 Dockerfile. It includes `pyctcdecode==0.5.0` and
`kenlm==0.3.0` for BookBot CTC beam search and CUDA-enabled Sherpa-ONNX and
ONNX Runtime packages for BookBot Zipformer; no separate
BookBot image is required.

### Runnable with the current image

| Model | Launcher/profile | Current status |
|---|---|---|
| BookBot orthographic CTC (greedy) | `run_transformers_inference.sh` + `bookbot-orthographic-ctc.yaml` | One-file inference passed and the existing plain-processor/argmax profile is unchanged. The local files lack Hugging Face download metadata, so rerun the pinned downloader only if snapshot certification is required. |
| BookBot orthographic CTC (packaged 5-gram) | `run_transformers_inference.sh` + `bookbot-orthographic-ctc-5gram.yaml` | Runnable now in the rebuilt shared image. Local-artifact loading and one-file GPU inference passed; metadata confirmed `Wav2Vec2ProcessorWithLM`, `BeamSearchDecoderCTC`, the 1,056,897,033-byte packaged LM, and the pinned package/settings provenance. On the same 4.096-second smoke item, greedy returned `a` and the word-LM decoder returned an empty hypothesis; use a development set, not held-out data, if settings are ever tuned. |
| BookBot phoneme CTC | `run_transformers_inference.sh` + `bookbot-phoneme-ctc.yaml` | One-file inference passed. Output is phonemic and must not be scored directly as orthographic WER. The local files lack Hugging Face download metadata. |
| Wav2Vec2-BERT Swahili | `run_transformers_inference.sh` + `w2v-bert-2.0-swahili-asr.yaml` | Complete snapshot; inference passed. |
| Paza Whisper large-v3-turbo | `run_transformers_inference.sh` + `paza-whisper-large-v3-turbo-sw.yaml` | Complete snapshot; inference passed. Above the owner-documented 448-token input limit, the profile applies the documented project policy: deterministic 30-second, zero-overlap sequential chunking in memory, followed by source-order text joining. It does not enable timestamp-based long-form decoding. |
| Whisper large | `run_transformers_inference.sh` + `whisper-large-sw.yaml` | Complete snapshot; GPU inference passed. Inputs above 30 seconds use Hugging Face's untruncated timestamp-based long-form path without modifying source audio. |
| Whisper large-v2 | `run_transformers_inference.sh` + `whisper-large-v2-sw.yaml` | Complete snapshot; inference passed. Inputs above 30 seconds use Hugging Face's untruncated timestamp-based long-form path without modifying source audio. |
| MMS-1B Swahili | `run_transformers_inference.sh` + `mms-1b-all-swh.yaml` | Complete Swahili package; inference and `swh` adapter loading passed. |
| BookBot Zipformer streaming RNN-T | `run_sherpa_onnx_inference.sh` + `zipformer-streaming-robust-sw-v4.yaml` | Moved out of the Transformers model root because it is a Sherpa-ONNX artifact. The shared CUDA image now includes Sherpa-ONNX; one-file GPU inference passed. Output is phonemic. |
| NeMo Exp41 / [AI-for-Education `sw-tz-child-egra-fastconformer-ctc-110m`](https://huggingface.co/AI-for-Education/sw-tz-child-egra-fastconformer-ctc-110m) | `run_nemo_inference.sh` + `swahili-exp41-ctc.yaml` | The complete public `model_exp41_avg.nemo` artifact is present locally. Its 463,144,960-byte size and SHA-256 `6450926bc1338827ab201b2d9f8f94bcb7a5bd06b6f72690f60ead14a067a7b0` match the model owner's published identity. Restore/inference passed, including a 7,617-segment run with zero inference errors. |
| Gemma 4 E2B | `run_multimodal_inference.sh` + `gemma-4-E2B-sw.yaml` | Uses the complete local BF16 snapshot through a separate prompt-driven audio adapter. Audio longer than Gemma's 30-second limit is decoded as ordered, non-overlapping 30-second chunks and rejoined into one shared output row. The existing image already has the required Transformers, Torch, Accelerate, and audio packages; no image rebuild or additional download is required. |

The decoder dependencies are additive and are imported only by a CTC profile
whose strategy is `beam_search`; they do not change greedy CTC, Whisper-style,
NeMo, post-processing, or evaluation selection. The current Compose inference
services reserve an NVIDIA GPU (`gpus: all`). A CPU-only deployment requires a
CPU image build plus removal or override of that Compose GPU reservation.

### Runnable through a current inference route after preparation

| Model | Required preparation |
|---|---|
| BookBot Zipformer six-row matrix | Use `bash ./run_bookbot_zipformer_matrix.sh`. It validates the pinned native/INT8 ONNX/INT8 ORT artifacts by SHA-256, rebuilds the shared ASR image, and then runs rows 30–35 in order. The optional `docs/bookbot-zipformer-runbook.md` is a local-development runbook and is not versioned. |
| HuBERT large-ls960-ft | Download the missing pinned snapshot. The existing CTC adapter and `hubert-large-ls960-ft-en.yaml` profile can then run it in the current image. This is English-only. |

### Requires another execution environment or additional model artifacts

| Model | What is still required |
|---|---|
| Paza Phi-4 multimodal | Runnable now in its dedicated image on the current hardware. Image build and one-file BF16 GPU inference passed with zero errors; the five-shard load took about 127 seconds and the 4.096-second file took about 16 seconds. The bounded auto-placement/offload path remains available if memory pressure changes. No model download is required. |
| Qwen2.5-Omni-7B | The dedicated text-only adapter/profile/image structure is implemented: the talker is disabled and `return_audio=False`. Its profile requires at least 40 GiB VRAM and deliberately rejects this 16 GiB host before loading weights. Build/run it only on suitable hardware; no model download is required. |
| Kaldi | Obtain the exact trained acoustic model, feature configuration, lexicon, language model, symbol tables, and decoding graph, then build a Kaldi service/adapter. The toolkit source alone is not a runnable ASR model. |
| `Swahili_exp1_100epochs` | Obtain complete NeMo weights or a `.nemo` artifact. The currently present tokenizer/configuration files are insufficient. |



## Straightforward steps

1. **Build the Docker image** (optional):

   GPU-enabled default:
   ```bash
   docker compose build
   ```

   CPU-only alternative:
   ```bash
   docker compose build --build-arg TORCH_CUDA=cpu
   ```

   The tracked Compose file requests `gpus: all` for all four inference services.
   CPU-only hosts must locally override or remove those requests; see
   [How to run (Docker)](#how-to-run-docker).

2. **Prepare dataset + model**
   - Copy dataset files (`0_Audio/`, `2_TextGrid/`, `Student_Full_Canonical_EGRA_*.csv`, `Student_MetaData_EGRA_*.csv`) under `input_output_data/input/<dataset_name>/`.
   - Place passages CSV at `input_output_data/input/oral_passages.csv`.
   - Place model artifacts under the `models/` directory for their inference library: `inference/nemo/`, `inference/transformers/`, `inference/sherpa_onnx/`, `inference/torch/`, or `inference/multimodal/`.
   - Select a tracked YAML inference profile from the corresponding `profiles/` directory. Profiles use relative artifact paths; inference never downloads models.

3. **Build base full manifest (input for segmentation)**

   Generic:
   ```bash
   ./run_make_ref_manifest.sh \
     --dataset_root input_output_data/input/<dataset_name> \
     --passages_csv input_output_data/input/oral_passages.csv \
     --output_jsonl input_output_data/output/experiments/<dataset_name>/manifests/ref_manifest.raw.jsonl
   ```

   Example:
   ```bash
   ./run_make_ref_manifest.sh \
     --dataset_root input_output_data/input/2_Batch3_4_Data_validation \
     --passages_csv input_output_data/input/oral_passages.csv \
     --output_jsonl input_output_data/output/experiments/2_Batch3_4_Data_validation/manifests/ref_manifest.raw.jsonl
   ```

4. **Segment manifest + generate segmented audio**

   Generic:
   ```bash
   ./run_segment.sh \
     --textgrid_root input_output_data/input/<dataset_name>/0_IAR/2_TextGrid \
     --manifest_in input_output_data/output/experiments/<dataset_name>/manifests/ref_manifest.raw.jsonl \
     --manifest_out input_output_data/output/experiments/<dataset_name>/manifests/ref_manifest.raw_segments.jsonl \
     --segments_out_root input_output_data/output/experiments/<dataset_name>/audio_segments
   ```

   Example:
   ```bash
   ./run_segment.sh \
     --textgrid_root input_output_data/input/2_Batch3_4_Data_validation/0_IAR/2_TextGrid \
     --manifest_in input_output_data/output/experiments/2_Batch3_4_Data_validation/manifests/ref_manifest.raw.jsonl \
     --manifest_out input_output_data/output/experiments/2_Batch3_4_Data_validation/manifests/ref_manifest.raw_segments.jsonl \
     --segments_out_root input_output_data/output/experiments/2_Batch3_4_Data_validation/audio_segments
   ```

5. **Run ASR inference on segmented audio**

   Run either or both inference routes from the same segmented manifest. Each launcher
   creates and prints its own `<inference_setup_id>_<timestamp>` run ID.

   NeMo:
   ```bash
   ./run_nemo_inference.sh \
     --inference_profile inference/nemo/profiles/swahili-exp41-ctc.yaml \
     --audio_manifest input_output_data/output/experiments/<dataset_name>/manifests/ref_manifest.raw_segments.jsonl
   ```

   Transformers (BookBot CTC):
   ```bash
   ./run_transformers_inference.sh \
     --inference_profile inference/transformers/profiles/bookbot-orthographic-ctc.yaml \
     --audio_manifest input_output_data/output/experiments/<dataset_name>/manifests/ref_manifest.raw_segments.jsonl \
     --batch_size 8
   ```

   For Whisper-family generation, use the same Transformers command with
   `inference/transformers/profiles/paza-whisper-large-v3-turbo-sw.yaml`.
   Both launchers require `--inference_profile` and write the same `transcriptions.jsonl`
   schema plus `run_metadata.json`.

6. **Build final segment-level manifest (attach `pred_text` from ASR) + clean**

   Generic:
   ```bash
   RUN_ID=<inference-setup-id>_<timestamp>   # copy the value printed by inference
   ./run_manifest.sh \
     --dataset_root input_output_data/input/<dataset_name> \
     --audio_manifest input_output_data/output/experiments/<dataset_name>/manifests/ref_manifest.raw_segments.jsonl \
     --prediction_manifest input_output_data/output/transcripts/$RUN_ID/transcriptions.jsonl
   ```

   This generates:
   - `input_output_data/output/evaluations/$RUN_ID/manifests/ref_manifest.raw.jsonl`
   - `input_output_data/output/evaluations/$RUN_ID/manifests/ref_manifest.clean.jsonl`

   When `--prediction_manifest` is supplied, its hypotheses are authoritative: existing
   `pred_text` values in the base manifest are not retained. Manifest building and
   evaluation also fail fast if `ref_text` or `can_text` contains a known Unicode
   replacement/mojibake marker. This check does not reject valid IPA Unicode or
   alter model-produced `pred_text`.

   Run this step once for each inference run. Smoke-test transcript paths are mapped
   to the parallel `smoke_tests/evaluations/$RUN_ID/` directory automatically.

7. **Run evaluation from cleaned segment manifest**

   Generic:
   ```bash
   ./run_eval2.sh \
     --dataset_root input_output_data/input/<dataset_name> \
     --manifest_in input_output_data/output/evaluations/$RUN_ID/manifests/ref_manifest.clean.jsonl
   ```

   Optional IPA comparison for any model:
   ```bash
   ./run_eval2.sh \
     --dataset_root input_output_data/input/<dataset_name> \
     --manifest_in input_output_data/output/evaluations/$RUN_ID/manifests/ref_manifest.clean.jsonl \
     --scoring_representation ipa \
     --g2p-tool babygruut
   ```

   Native IPA hypotheses use their reviewed inventory adapter. Orthographic
   hypotheses are converted with the same explicitly selected G2P system as
   CAN/REF. Choose either `africa_g2p` or `babygruut`; babygruut uses its pinned
   local SQLite lexicon with CRF fallback and never Turso. This writes PER under
   `input_output_data/output/evaluations/$RUN_ID/ipa/<g2p_system_id>/`; results
   produced by different exact G2P identities are never combined.

8. **Inspect the outputs** under `input_output_data/output/evaluations/$RUN_ID/`:
   - `orthographic/egra_eval_detailed.csv` and
     `orthographic/egra_eval_summary.txt` for valid WER scoring.
   - `ipa/<g2p_system_id>/egra_eval_detailed.csv` and
     `ipa/<g2p_system_id>/egra_eval_summary.txt` for valid PER scoring.
   - `evaluation_metadata.json` inside each completed representation directory,
     recording its source manifest and representation status.
9. **Explore results interactively**  
   - Dependencies: `pip install streamlit pandas numpy` (preferably inside a virtualenv).  
     - Specific example: `python3 -m venv .venv_streamlit && . .venv_streamlit/bin/activate && pip install --upgrade pip setuptools wheel && pip install streamlit pandas numpy`
   - Run: `streamlit run egra_dashboard2.py -- --csv <path/to/egra_eval_detailed.csv>`  
     - Specific example: ` . .venv_streamlit/bin/activate && streamlit run egra_dashboard2.py -- --csv input_output_data/output/evaluations/$RUN_ID/orthographic/egra_eval_detailed.csv`
   - Open the browser tab (Streamlit serves on `http://localhost:8501` by default) to sort, group and aggregate metrics.

   Leaderboard generation admits only completed, representation-compatible
   evaluations and rejects smoke-test runs before writing any CSV, dashboard, or
   shareable HTML output. Because every displayed row has already passed that
   rule, the leaderboard does not show a redundant per-run completion label.

Everything runs in Docker setup (CPU-only or GPU-enabled).


## Full example for held-out data

The example assumes that data and models have been placed in:

- Data: `input_output_data/input/heldout_combined_fixed_20260525`
- Inference profile: `inference/nemo/profiles/swahili-exp41-ctc.yaml`
- Model artifact: `inference/nemo/models/model_exp41_avg.nemo`

All the steps above can then be performed in sequence:

    ./run_make_ref_manifest.sh \
        --dataset_root input_output_data/input/heldout_combined_fixed_20260525 \
        --passages_csv input_output_data/input/oral_passages.csv \
        --output_jsonl input_output_data/output/experiments/heldout_combined_fixed_20260525_exp41/manifests/ref_manifest.raw.jsonl

    ./run_segment.sh \
        --textgrid_root input_output_data/input/heldout_combined_fixed_20260525/2_TextGrid \
        --manifest_in input_output_data/output/experiments/heldout_combined_fixed_20260525_exp41/manifests/ref_manifest.raw.jsonl \
        --manifest_out input_output_data/output/experiments/heldout_combined_fixed_20260525_exp41/manifests/ref_manifest.raw_segments.jsonl \
        --segments_out_root input_output_data/output/experiments/heldout_combined_fixed_20260525_exp41/audio_segments

    ./run_nemo_inference.sh \
        --inference_profile inference/nemo/profiles/swahili-exp41-ctc.yaml \
        --audio_manifest input_output_data/output/experiments/heldout_combined_fixed_20260525_exp41/manifests/ref_manifest.raw_segments.jsonl

    # Copy the exact value printed by "[INFO] Inference run:".
    RUN_ID=swahili-exp41-ctc_<timestamp>

    ./run_manifest.sh \
        --dataset_root input_output_data/input/heldout_combined_fixed_20260525 \
        --audio_manifest input_output_data/output/experiments/heldout_combined_fixed_20260525_exp41/manifests/ref_manifest.raw_segments.jsonl \
        --prediction_manifest input_output_data/output/transcripts/$RUN_ID/transcriptions.jsonl

    ./run_eval2.sh \
        --dataset_root input_output_data/input/heldout_combined_fixed_20260525 \
        --manifest_in input_output_data/output/evaluations/$RUN_ID/manifests/ref_manifest.clean.jsonl

    . .venv_streamlit/bin/activate && streamlit run egra_dashboard2.py -- --csv input_output_data/output/evaluations/$RUN_ID/orthographic/egra_eval_detailed.csv

---

## Working Modes

- Segment-only flow is the default documented flow:
- `run_segment.sh`: creates segment audio + segment manifest.
- `run_nemo_inference.sh`, `run_transformers_inference.sh`, `run_sherpa_onnx_inference.sh`, or `run_multimodal_inference.sh`: transcribes segment audio using a required inference profile.
- `run_manifest.sh`: builds/cleans the final segment-level manifest from `--audio_manifest` and optionally attaches `--prediction_manifest`.
- `run_eval2.sh`: scores only from an existing cleaned segment manifest (`--manifest_in`).

---

## Contents

- [Straightforward steps](#straightforward-steps)
- [Working Modes](#working-modes)
- [Contents](#contents)
- [Project structure](#project-structure)
- [What the pipeline does](#what-the-pipeline-does)
- [Input data format](#input-data-format)
- [How to run (Docker)](#how-to-run-docker)
  - [1) Build the image](#1-build-the-image)
  - [2) Run inference (ASR)](#2-run-inference-asr)
  - [3) Run standalone segmentation](#3-run-standalone-segmentation)
  - [4) Run inference on segments](#4-run-inference-on-segments)
  - [5) Build final segment manifest (REF/CAN/HYP source for evaluation)](#5-build-final-segment-manifest-refcanhyp-source-for-evaluation)
  - [6) Run evaluation](#6-run-evaluation)
  - [7) Optional: compare with NeMo offline scoring](#7-optional-compare-with-nemo-offline-scoring)
- [Outputs \& how to interpret them](#outputs--how-to-interpret-them)
- [Metrics \& definitions](#metrics--definitions)
  - [Metric ranges \& units](#metric-ranges--units)
- [Configuration knobs](#configuration-knobs)
  - [Inference profiles and launchers](#inference-profiles-and-launchers)
  - [Manifest build (`manifest_pipeline.py`)](#manifest-build-manifest_pipelinepy)
  - [Evaluation (`eval_pipeline2.py`)](#evaluation-eval_pipeline2py)
- [Troubleshooting](#troubleshooting)
- [Source files](#source-files)

---

## Project structure

```
.
├── docker/
│   └── Dockerfile                # Shared CUDA/CPU image for NeMo, Transformers, and Sherpa-ONNX
├── docker-compose.yml            # Inference, evaluation, and dashboard services
├── manifest_pipeline.py          # Build+clean manifest entrypoint (used by run_manifest.sh)
├── eval_pipeline2.py             # Evaluation entrypoint from existing manifest (used by run_eval2.sh)
├── infer.py                      # Legacy NeMo --model entrypoint
├── run_*_inference.sh            # Profile-driven launchers for every inference route
├── run_onnxruntime_prepare.sh    # Explicit ONNX artifact preparation
├── run_eval2.sh                  # Wrapper script for evaluation (supports --manifest_in)
├── run_manifest.sh               # Wrapper script to build+clean manifest only (no scoring)
├── run_segment.sh                # Wrapper script for standalone manifest/audio segmentation
├── inference/
│   ├── common.py                 # Shared audio discovery, loading, duration, and resampling
│   ├── contracts.py              # Shared adapter and transcription-result contracts
│   ├── profile.py                # Strict portable YAML profile loading and validation
│   ├── runner.py                 # Shared ordered inference and output writing
│   ├── nemo/
│   │   ├── infer.py              # NeMo inference entrypoint
│   │   ├── backend.py            # NeMo CTC/RNNT/hybrid adapter
│   │   ├── profiles/             # Tracked NeMo inference profiles
│   │   ├── models/               # Local NeMo artifacts (ignored, mounted read-only)
│   │   └── tmp/                  # Temporary NeMo segments
│   ├── transformers/
│   │   ├── infer.py              # Transformers inference entrypoint
│   │   ├── factory.py            # Adapter selection
│   │   ├── adapters/             # CTC and speech-seq2seq adapters
│   │   ├── profiles/             # Tracked Transformers inference profiles
│   │   ├── models/               # Local Hugging Face artifacts (ignored, mounted read-only)
│   │   └── tmp/                  # Temporary Transformers files
│   ├── sherpa_onnx/              # Streaming transducer route
│   ├── onnxruntime/              # Desktop and Android-parity ONNX routes
│   └── multimodal/               # Gemma, Phi-4, and Qwen audio routes
├── egra_eval2/                   # Evaluation, manifest, dataset-layout, and segmentation modules
│   ├── dataset_layout.py         # Discover dataset audio, TextGrid, and metadata paths
│   ├── manifest_builder.py       # Build reference manifests
│   ├── manifest_cleaner.py       # Normalize and clean manifest text
│   ├── prediction_manifest.py    # Load library-neutral prediction JSONL outputs
│   ├── evaluate.py               # Current row-level and aggregate scoring
│   ├── metrics.py                # WER, MER, and fine-grained error metrics
│   ├── reporting_candidates.py   # Inactive reporting logic awaiting migration
│   ├── reporting_candidate_support.py # Inactive support for those candidates
│   ├── segmenter.py              # TextGrid-driven segmentation helpers
│   └── scoring_text.py           # Orthographic/IPA scoring-view selection
├── tools/                        # Helper scripts (manifest prep, comparisons, etc.)
│   ├── migrations/               # Explicit one-off migration utilities
│   ├── make_ref_manifest.py      # Standalone reference manifest builder
│   └── ...
└── input_output_data/
│   ├── input/                    # place each dataset folder for every experiment here
│   └── output/                   # experiment results (one subfolder per run)
```

---

## What the pipeline does

**Inference (`inference/`)**
- Loads a required inference profile and resolves its artifact from the matching `models/` directory.
- Reads exact segments from `--audio_manifest` or recursively discovers `.wav` files under `--root_audio_dir`; the NeMo legacy API also retains dataset discovery and optional TextGrid-driven segmentation.
- Resamples audio to the model's required sample rate, then dispatches to the selected inference adapter.
- Emits the unchanged `transcriptions.jsonl` schema and a separate `run_metadata.json` record.

**Manifest build (`manifest_pipeline.py`)**
- In segment-only flow, loads the exact segmented audio rows from `--audio_manifest`.
- Attaches ASR hypotheses from `--prediction_manifest` when supplied.
- Writes cleaned segment manifests (for example `ref_manifest.segment.raw.jsonl` and `ref_manifest.segment.clean.jsonl`).

**Evaluation (`eval_pipeline2.py`)**
- Loads the cleaned manifest via `--manifest_in` and attaches `REF/CAN/HYP` by audio key.
- Computes REF/HYP error rates, CAN-relative agreement metrics, and task-level summaries.
- Produces a **detailed CSV**, a text summary, representation metadata, and optional scatter plots.

---

## Input data format

`input_output_data/input/` is intentionally empty. For each experiment copy exactly **one dataset folder** here. The scripts run inside the container, so use the mounted path prefix `/io/input/<dataset>` when supplying `--dataset_root`. A typical layout looks like this:

```
input_output_data/input/1_Batch2_Data-v2/
└── 1_Batch2_Data
    ├── 0_IAR
    │   ├── 0_Audio/          # learner_id subfolders containing WAV files
    │   └── 2_TextGrid/       # annotator folders
    ├── Student_Full_Canonical_EGRA_*.csv
    └── Student_MetaData_EGRA_*.csv
```

- Only `0_Audio/`, `2_TextGrid/`, the two `Student_*` CSVs, and the passages CSV are consumed; other folders (for example `1_Annotation`) are ignored.
- Evaluation walks every subdirectory under `2_TextGrid/` and chooses the first `.TextGrid` whose stem matches the audio; no annotator flag is required. Inference still accepts `--dataset_annotator` if you want to limit slicing to a specific folder.
- Evaluation now uses a cleaned manifest as explicit input (`--manifest_in`).

## How to run (Docker)

> The default setup needs Docker, an NVIDIA GPU/driver, and Docker GPU support.

### 1) Build the image

GPU-enabled by default (CUDA 12.8 wheels, including NVIDIA Blackwell / `sm_120`):
```bash
docker compose build
```

The tracked Compose configuration grants GPU access to both ASR services. The
adapters then select `cuda:0` automatically when inference starts.

To build a CPU-only image instead:
```bash
docker compose build --build-arg TORCH_CUDA=cpu
```
CPU-only hosts must also remove or locally override the tracked `gpus: all`
requests before running the ASR services.

### 2) Run inference (ASR)

Use the matching launcher for the selected inference profile. Each launcher
requires `--inference_profile`, accepts either `--audio_manifest` or `--root_audio_dir`, and
writes to `input_output_data/output/transcripts/<inference_setup_id>_<timestamp>/` by default.
The scripts run `docker compose run` with your current `uid:gid`, so generated
files inside `input_output_data` are owned by the host user.

NeMo example:
```bash
./run_nemo_inference.sh \
  --inference_profile inference/nemo/profiles/swahili-exp41-ctc.yaml \
  --root_audio_dir input_output_data/input/<dataset>/0_Audio
```

Transformers speech-seq2seq example:
```bash
./run_transformers_inference.sh \
  --inference_profile inference/transformers/profiles/paza-whisper-large-v3-turbo-sw.yaml \
  --audio_manifest input_output_data/output/<experiment>/manifests/ref_manifest.raw_segments.jsonl \
  --batch_size 8
```

Speech-seq2seq profiles assume pre-segmented audio. Long-form chunking is not
implemented in this delivery.

Root `infer.py` remains available only for legacy NeMo commands that pass
`--model` directly. New workflows should use a profile-driven launcher.

### 3) Run standalone segmentation

We provide `run_segment.sh` to run `segment_manifests.py` directly.
The script is now strict segment-only:
- rows without valid audio/TextGrid are skipped (not copied as full utterances);
- even a single valid interval produces `*_segment1.wav`.

Usage:
```bash
./run_segment.sh \
  --textgrid_root /io/input/<dataset>/0_IAR/2_TextGrid \
  --manifest_in /io/output/<experiment>/manifests/ref_manifest.raw.jsonl \
  --manifest_out /io/output/<experiment>/manifests/ref_manifest.raw_segments.jsonl \
  --segments_out_root /io/output/<dataset>/audio_segments
```

Example:
```bash
./run_segment.sh \
  --textgrid_root input_output_data/input/1_Batch2_Data_16spk_subset/0_IAR/2_TextGrid \
  --manifest_in input_output_data/output/experiments/1_Batch2_Data_16spk_subset/manifests/ref_manifest.raw.jsonl \
  --manifest_out input_output_data/output/experiments/1_Batch2_Data_16spk_subset/manifests/ref_manifest.raw_segments.jsonl \
  --segments_out_root input_output_data/output/experiments/1_Batch2_Data_16spk_subset/audio_segments
```

### 4) Run inference on segments

Pass the segment manifest to either inference launcher. This preserves manifest
ordering and avoids rediscovering files:

```bash
./run_nemo_inference.sh \
  --inference_profile inference/nemo/profiles/swahili-exp41-ctc.yaml \
  --audio_manifest input_output_data/output/<experiment>/manifests/ref_manifest.raw_segments.jsonl
```

```bash
./run_transformers_inference.sh \
  --inference_profile inference/transformers/profiles/bookbot-orthographic-ctc.yaml \
  --audio_manifest input_output_data/output/<experiment>/manifests/ref_manifest.raw_segments.jsonl \
  --batch_size 8
```

### 5) Build final segment manifest (REF/CAN/HYP source for evaluation)

We provide `run_manifest.sh`. It will:
- Load the segmented audio manifest from `--audio_manifest` (preserve segment granularity).
- Attach ASR hypotheses from `--prediction_manifest`.
- Derive `evaluations/<inference_setup_id>_<timestamp>/` from a standard transcript path.
- Write raw and cleaned segment manifests under that run's `manifests/` folder.

Usage:
```bash
./run_manifest.sh \
  --dataset_root /io/input/<dataset> \
  --audio_manifest /io/output/<experiment>/manifests/ref_manifest.raw_segments.jsonl \
  --prediction_manifest /io/output/transcripts/<inference_setup_id>_<timestamp>/transcriptions.jsonl
```

### 6) Run evaluation

We provide `run_eval2.sh`. It will:
- Load segment rows from the cleaned manifest passed via `--manifest_in` and score at segment level.
- Attach learner metadata from dataset CSVs via `learner_id`.
- Keep the detailed CSV and text summary in the evaluation run that owns the manifest.
Like the inference wrapper, it executes the container with your user ID so the resulting CSVs and summaries remain writable without sudo.

Usage:
```bash
./run_eval2.sh \
  --dataset_root /io/input/<dataset> \
  --manifest_in /io/output/evaluations/<inference_setup_id>_<timestamp>/manifests/ref_manifest.clean.jsonl
```


### 7) Optional: compare with NeMo offline scoring

`run_nemo_offline_eval.sh` normalizes the dataset into NeMo manifests and invokes NVIDIA’s
`speech_to_text_eval.py` for both REF↔HYP and CAN↔HYP scoring.

Usage example:
```bash
./run_nemo_offline_eval.sh \
  --dataset_root /io/input/1_Batch2_Data \
  --dataset_annotator Flora \
  --output_dir /io/output/1_Batch2_Data/nemo_asr_output \
  --nemo_hyp_manifest /io/output/1_Batch2_Data-v2/nemo_asr_output/transcriptions.jsonl
```

This writes `ref_manifest_norm.jsonl` and `can_manifest_norm.jsonl` alongside the supplied output
directory and enriches them with per-sample NeMo WER scores. Pass either `--output_dir` or
`--dataset_root`; without one of these the script aborts.

---

## Outputs & how to interpret them

By default, all evaluation outputs land in
`input_output_data/output/evaluations/<inference_setup_id>_<timestamp>/` (or the parallel
`smoke_tests/evaluations/` path). Each representation is written separately
under `orthographic/` or `ipa/<g2p_system_id>/` and contains:

1. **`egra_eval_detailed.csv`** — one row per merged EGRA item. It contains
   `learner_id`, `audio_type`, `audio_file`, normalized `CAN`/`REF`/`HYP`,
   learner metadata, REF/HYP alignment counts (`S_ref_hyp`, `D_ref_hyp`,
   `I_ref_hyp`, `C_ref_hyp`, `N_ref_hyp`), CAN-relative correct counts
   (`C_can_ref`, `C_can_hyp`), mistake-alignment counts (`S_mer`, `D_mer`,
   `I_mer`, `C_mer`, `N_mer`), and fine-grained substitution/deletion/insertion
   classification counts.

2. **`egra_eval_summary.txt`** — global and per-task aggregate WER or PER,
   mistake error rate (MER), and the task-specific agreement metrics supported
   by the evaluator. Error rates are percentages. Passing `--detailed` also
   writes the supported scatter plots into the same representation directory.

3. **`evaluation_metadata.json`** — the scoring representation, source
   manifest, reference-view identity, and evaluation provenance.

The richer per-speaker and pair-specific reports from the former evaluator are
not produced by the active workflow. Their source is isolated as migration
material in `egra_eval2/reporting_candidates.py`.

---

## Metrics & definitions

All metrics are computed after text normalization in
`egra_eval2/eval_utils.py`: NFC Unicode, lowercase, punctuation removed, and
whitespace collapsed.

We compute standard REF/HYP alignment counts via `jiwer`:
- **S** — substitutions  
- **D** — deletions  
- **I** — insertions  
- **C** — correct matches 
- **N** — number of reference tokens

From those we derive:

- **WER/PER** = `(S + D + I) / N × 100`. Orthographic evaluation reports
  WER; IPA evaluation reports PER.
- **MER** compares the CAN-relative error sequence of REF with the corresponding
  sequence for HYP, then applies the same aggregate error-rate formula.
- `C_can_ref` and `C_can_hyp` are the correct-token counts from CAN→REF and
  CAN→HYP alignments. For passage and grid tasks, their correlation measures
  whether the model preserves learner-level variation in correct responses.
- Fine-grained substitution, deletion, and insertion precision/recall/F1 are
  calculated for the supported passage and grid task summaries.
- Isolated-letter, isolated-syllable, and isolated-nonword summaries treat a
  CAN-relative error as the positive class and report classification accuracy.

---

### Metric ranges & units

Aggregate error rates and summary precision/recall/F1 values are emitted as
percentages. Count columns remain raw integers.

| Metric | Description | Typical Range / Unit | Interpretation |
|:--|:--|:--|:--|
| **WER/PER** | REF/HYP substitutions + deletions + insertions, divided by reference-token count | Percentage; can exceed 100 with many insertions | Lower is better |
| **MER** | Error rate between the human and model CAN-relative error sequences | Percentage | Lower is better |
| **Correlation** | Pearson correlation between `C_can_ref` and `C_can_hyp` for passage/grid tasks | −1 to 1, or undefined when variance is zero | Higher is better |
| **Fine-grained F1** | Agreement on substitution, deletion, or insertion events | 0–100% | Higher is better |
| **Isolated-task accuracy** | Agreement on whether an isolated response contains a CAN-relative error | 0–100% | Higher is better |
| **S/D/I/C/N fields** | Raw alignment counts used by the corresponding aggregate metric | Integer counts | Diagnostic evidence |

**Note:**  
If an aggregate reference-token count is zero, its error rate is undefined and
appears as `NaN` in the summary.

---


## Configuration knobs

### Inference profiles and launchers

- **Inference profile**: use `--inference_profile <profile.yaml>` with every current inference launcher. Schema v2 profiles define `inference_setup_id`, `inference_library`, adapter, relative model artifact, language/task, loading settings, decoding, and structured parameter evidence. Container images supply compatible dependencies but do not select model-specific behavior.
- **Invocation controls**: batch size, thread/worker counts, input selection, and output roots remain launcher arguments and are recorded in run metadata rather than being hidden in an image or treated as model hyperparameters.
- **Model storage**: place artifacts below the owning inference library's `models/` directory. `ASR_MODEL_ROOT` overrides that default root; Compose sets it to the read-only `/models` mount.
- **Input**: provide exactly one of `--audio_manifest <segments.jsonl>` or `--root_audio_dir <audio-directory>`. NeMo retains `--dataset_root` and `--dataset_annotator` for legacy dataset discovery and optional TextGrid segmentation.
- **Output base**: inference defaults to `input_output_data/output`; `--output_root <directory>` changes that base. The runner creates `transcripts/<inference_setup_id>_<UTC timestamp>/` below it, or `smoke_tests/transcripts/...` with `--smoke_test`.
- **Execution controls**: NeMo retains its CPU worker, temporary-segment, decoder, and debug controls; Transformers retains `--batch_size`; Sherpa-ONNX adds `--num_threads`. Sources, evidence strength, and hardware assumptions are recorded beside the relevant launcher. Optional extended notes in `docs/inference-execution-parameter-provenance.md` are local-development documentation and are not versioned.
- **Offline operation**: profiles require local artifacts and `local_files_only: true`; downloading a model is a separate preparation step.

Root `infer.py` is the only legacy NeMo API and continues to accept `--model`
directly. It also remaps the former NeMo model-directory prefix when that old
path no longer exists. Do not use direct model paths with the new launchers.

### Manifest build (`manifest_pipeline.py`)
The dataset and segmented audio manifest remain explicit. The evaluation output is
derived from a standard transcript path unless you override it.

Run `python3 manifest_pipeline.py --help` to see available options. Highlights:
- `--dataset_root /io/input/<dataset>` — required; automatically discovers the `Student_*` CSVs plus `0_Audio/` and `2_TextGrid/`.
- `--audio_manifest /io/output/<experiment>/manifests/ref_manifest.raw_segments.jsonl` — required in segment-only flow; defines and preserves the exact audio segment rows used for inference.
- `--output_root /io/output/evaluations/<inference_setup_id>_<timestamp>` — optional exact destination; otherwise derived from `--prediction_manifest` by replacing `transcripts` with `evaluations`.
- `--prediction_manifest /path/to/transcriptions.jsonl` — optional; if provided, `pred_text` is attached from any inference route. Repeat the argument to merge multiple prediction files.

### Evaluation (`eval_pipeline2.py`)
Run `python3 eval_pipeline2.py --help` to see available options. Highlights:
- `--dataset_root /io/input/<dataset>` — required.
- `--manifest_in /io/output/<experiment>/manifests/ref_manifest.segment.clean.jsonl` — required.
- `--output_root /io/output/evaluations/<inference_setup_id>_<timestamp>` — optional run
  destination; otherwise the owning evaluation run is derived from
  `--manifest_in`. The evaluator appends `orthographic/` or
  `ipa/<g2p_system_id>/`. A custom `--out_csv` path is rejected if it escapes
  that representation directory.

---

## Troubleshooting

- **No GPU used**: Rebuild the default CUDA image, confirm Docker can access the NVIDIA GPU, and check that the resolved ASR service retains `gpus: all`.
- **Empty or short `pred_text`**: Check that the model matches the language/domain. Also verify sample rate conversion (the script resamples to 16 kHz automatically).
- **Missing REF text in segment base manifest**: Ensure `run_segment.sh` used the correct `--textgrid_root` and that audio/TextGrid stems align.
- **Segment ASR not attached**: Check that `--prediction_manifest` in `run_manifest.sh` points to segmented ASR output and that `--match_on` is appropriate.
- **Segmentation not applied in inference**: Ensure matching `.TextGrid` files exist under `2_TextGrid/` and names align with audio stems; segmentation follows all parsed intervals from TextGrid.
- **Unexpected full rows in segment manifest**: re-run `run_segment.sh`; strict mode drops non-segmentable rows and writes only `*_segmentN.wav` entries.
- **Profile rejected before loading**: Check for unknown keys, an invalid inference-library/adapter/decoding combination, an absolute or escaping artifact path, or a missing local artifact.
- **Wrong Transformers decoding path**: CTC profiles must use the `ctc` adapter with greedy decoding; Whisper-family profiles must use `speech_seq2seq` with `generate`.
- **Permissions**: The repo root and `input_output_data` are mounted read-write. Each inference service mounts its own model-artifact directory read-only at `/models`.

---

## Source files

- **`inference/common.py`, `inference/contracts.py`, `inference/profile.py`, `inference/runner.py`**
  Define common audio handling, adapter/result contracts, strict portable inference profiles, ordered inference, and the shared `transcriptions.jsonl` plus `run_metadata.json` outputs.

- **`inference/nemo/`**
  Implements profile-driven NeMo CTC/RNNT/hybrid inference while retaining existing dataset discovery and optional TextGrid behaviour.

- **`inference/transformers/`**
  Selects between the greedy CTC adapter and the Whisper-style speech-seq2seq adapter. Model-specific behaviour, including BookBot plain processing and MMS language-adapter selection, is profile-driven.

- **`inference/sherpa_onnx/`**
  Runs streaming ONNX transducers through Sherpa-ONNX while preserving the same profile, manifest, progress, and output contracts.

- **`infer.py`**  
  Preserves the legacy NeMo `--model` command and delegates to the reorganized NeMo implementation.

- **`manifest_pipeline.py`**  
  Builds and cleans final manifests; in segment-only flow it loads `--audio_manifest` and attaches ASR `pred_text` from `--prediction_manifest`.

- **`eval_pipeline2.py`**
  Runs scoring and report generation using only a cleaned manifest (`--manifest_in`) plus dataset metadata CSVs.

- **`egra_eval2/evaluate.py` and `egra_eval2/metrics.py`**
  Produce row-level and aggregate counts (**S, D, I, C, N**), WER/PER, MER,
  agreement metrics, and fine-grained substitution/deletion/insertion scores.

- **`egra_eval2/textgrid_io.py`**
  Finds the requested tier case-insensitively (default `child`), gathers labeled intervals, strips filler tags (`<unk>`, `<noise>`, etc.), and concatenates labels to form **REF** per item while searching recursively across annotator folders.

- **`egra_eval2/linking.py`**
  Builds join keys from the EGRA CSV (`audio_name`, `audio_stem`) and attaches ASR HYPs by the chosen key (`stem` by default).

- **`egra_eval2/prediction_manifest.py`**
  Loads one or many prediction manifests (JSONL), extracting `audio_path`, `audio_name`, `audio_stem` and `hyp_text`.

- **`egra_eval2/dataset_layout.py`**
  Utility helpers that discover dataset packages containing `0_Audio/`, `2_TextGrid/` and the `Student_*` CSVs.

- **`egra_eval2/passage_merge.py`**
  Parses the passages CSV (various encodings handled), extracts `passage_num` and fills missing `canonical_text` for `passage_numX` rows.

- **`egra_eval2/eval_utils.py`**
  Shared text normalization, aggregation, and letter-canonical helpers.

- **`egra_eval2/reporting_candidates.py`**
  Inactive migration material for richer historical reports. The supported
  evaluator does not import it. Optional notes in
  `docs/legacy-code-migration.md` are local-development documentation and are
  not versioned.

- **`egra_eval2/segmenter.py`**
  Shared TextGrid segmentation module used by inference; parses interval blocks and cuts audio without label/tier filtering.

- **`docker/Dockerfile`**  
  Debian 12 base with PyTorch (CPU or CUDA), NeMo ASR 2.3, Transformers, Sherpa-ONNX, and pinned inference dependencies.

- **`docker-compose.yml`**
  Defines the NeMo, Transformers, Sherpa-ONNX, desktop ONNX, Android-parity
  ONNX, Gemma, Phi-4, and Qwen inference services, plus ONNX preparation,
  evaluation, and the isolated leaderboard dashboard. Inference services mount
  the repository at `/work`, data at `/io`, and the matching model-artifact
  directory read-only at `/models`.

- **`run_nemo_inference.sh` / `run_transformers_inference.sh` / `run_sherpa_onnx_inference.sh` / `run_multimodal_inference.sh` / `run_phi4_multimodal_inference.sh` / `run_qwen_omni_inference.sh` / `run_segment.sh` / `run_manifest.sh` / `run_eval2.sh`**
  Thin wrappers that run the appropriate Compose service and command. Every current inference wrapper requires `--inference_profile`.
- **`run_nemo_offline_eval.sh`**  
  Generates normalized REF/CAN manifests and runs NVIDIA NeMo’s own `speech_to_text_eval.py` script for REF↔HYP and CAN↔HYP scoring. Handy for cross-checking the internal metrics against the official NeMo implementation.
