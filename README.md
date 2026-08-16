## Overview

The purpose of this project is to evaluate locally deployed ASR models on the task of early grade reading assessments (EGRA) for Kiswahili child speech. Inference is currently supported through NVIDIA NeMo, Hugging Face Transformers, ONNX Runtime, Sherpa-ONNX, and a dedicated multimodal generation flow.

**Input (you need to provide)**

- A local ASR model artifact and its tracked model profile. A first NeMo model is [provided here](https://drive.google.com/file/d/1NQTC8532QluX7KXQNGcebKj9FseUzrO-), and the broader candidate list is documented in the [ASR benchmark candidate-model spreadsheet](https://www.dropbox.com/scl/fi/t4f76nnj14n7gog2zm6du/ASR_benchmark_candidate_models.xlsx?rlkey=62h00z8ks4aan9k70vv03cp7u&st=u2zd2pnu&dl=0).

- Dataset of Kiswahili child speech comprising:
  - audio files, 
  - cannonical texts, i.e. what the child should have uttered and 
  - reference text, i.e. what the child actually uttered

**Output**

Given the above, evaluation will be performed, producing a final `egra_eval_summary.txt` report.

## Inference backends

Inference is organized under `inference/` by model framework while sharing the same audio, manifest, result, and evaluation contracts:

- `inference/nemo/` contains NVIDIA NeMo inference and its profiles.
- `inference/transformers/` contains offline Hugging Face CTC and speech-seq2seq inference.
- `inference/sherpa_onnx/` contains offline Sherpa-ONNX streaming-transducer inference.
- `inference/onnxruntime/` contains PC CPU validation of FP32 and INT8 exports plus a separate, controlled Android-parity accuracy proxy for the packaged mobile artifact.
- `inference/multimodal/` contains prompt-driven audio-to-text generation for multimodal models, beginning with Gemma 4 E2B.
- `inference/common.py`, `inference/contracts.py`, and `inference/runner.py` provide the shared input, audio, result, and output behaviour.

Segment audio once with `run_segment.sh`, then pass the same segment manifest to any backend. All write unchanged `transcriptions.jsonl` rows with `audio_filepath`, `duration`, and `pred_text`, plus `run_metadata.json` describing the effective profile and runtime. Use `run_nemo_inference.sh`, `run_transformers_inference.sh`, `run_onnxruntime_inference.sh`, `run_onnxruntime_android_inference.sh`, `run_sherpa_onnx_inference.sh`, or `run_multimodal_inference.sh` for new runs; root `infer.py` is the only legacy NeMo `--model` API. See the README in each framework directory for model-specific details.

When inference adjusts a hypothesis after decoding, the transcript row keeps
both the scored `pred_text` and the model's original `raw_pred_text`.
`run_metadata.json` adds a concise optional `postprocessing` summary. Older
runs and backends without this block remain valid, and evaluation continues to
read `pred_text` exactly as before.

### Standard output layout

New profile-driven runs use one model/timestamp name throughout the pipeline:

```text
input_output_data/output/
├─ transcripts/
│  └─ <model>_<YYYY_MM_DD_HH_MM_SS_UTC>/
│     ├─ transcriptions.jsonl
│     └─ run_metadata.json
├─ evaluations/
│  └─ <model>_<YYYY_MM_DD_HH_MM_SS_UTC>/
│     ├─ manifests/
│     ├─ orthographic/             # Valid orthographic WER result
│     │  ├─ egra_eval_detailed.csv
│     │  ├─ egra_eval_summary.txt
│     │  └─ evaluation_metadata.json
│     ├─ ipa/                      # Valid IPA/PER result
│     │  ├─ egra_eval_detailed.csv
│     │  ├─ egra_eval_summary.txt
│     │  └─ evaluation_metadata.json
│     └─ orthographic_legacy/      # Optional historical mismatch only
└─ smoke_tests/
   ├─ transcripts/
   │  └─ <model>_<YYYY_MM_DD_HH_MM_SS_UTC>/
   └─ evaluations/
      └─ <model>_<YYYY_MM_DD_HH_MM_SS_UTC>/
```

Inference creates the model/UTC-timestamp name automatically. `run_manifest.sh`
derives the matching evaluation directory from the standard
`--asr_manifest` path. `run_eval2.sh` writes each scoring representation into
its own child directory, so WER and PER results cannot replace one another.
`auto` writes orthographic models under `orthographic/` and native phoneme
models under `ipa/`. The explicit `legacy_orthographic` mode exists only to
restore earlier phoneme-vs-orthography diagnostics under
`orthographic_legacy/`; those WER values are not representation-compatible
quality measurements.
For faithful archive recovery, that explicit legacy mode also preserves known
pre-fix reference corruption instead of applying the modern integrity gate;
`evaluation_metadata.json` records that the gate was disabled.
Use `--smoke_test` on either inference launcher for the smoke-test branch.
`--output_root` remains available when a different output base or an explicit
evaluation destination is required.

All inference backends print model-loading milestones and a shared file
progress bar. The bar advances after each completed batch and ends with result
and error counts; the exact output path is also printed.

| Stage | Preferred argument | File role | Compatibility name |
|---|---|---|---|
| NeMo inference input | `--audio_manifest` | Exact audio segments to transcribe | `--manifest_in` (legacy NeMo calls only) |
| Transformers inference input | `--audio_manifest` | Exact audio segments to transcribe | — |
| Sherpa-ONNX inference input | `--audio_manifest` | Exact audio segments to transcribe | — |
| Multimodal inference input | `--audio_manifest` | Exact audio segments to transcribe | N/A |
| Manifest merge input | `--asr_manifest` | Backend predictions in `transcriptions.jsonl` | `--nemo_manifest` (legacy NeMo-oriented merge calls only) |
| Evaluation input | `--manifest_in` | Cleaned manifest containing references and predictions | — |

Use the preferred names for every new command. The compatibility names exist
only so previously working NeMo commands do not break; Transformers commands should
not use them.

Run the same merge and evaluation steps once per model output. Report each
model's metrics separately; phoneme-model WER/MER is not directly comparable
with orthographic-model WER/MER because the scored units are different.

## Model generalisation readiness

The status below records local artifact checks and one-file inference smoke tests
performed on 12-13 August 2026. The shared
`voice-ai-evaluation-framework-asr:latest` image has been rebuilt from the
tracked CUDA 12.8 Dockerfile. It includes `pyctcdecode==0.5.0` and
`kenlm==0.3.0` for BookBot CTC beam search and a CUDA-enabled Sherpa-ONNX
runtime for BookBot Zipformer; no separate BookBot image is required.

### Runnable with the current image

| Model | Launcher/profile | Current status |
|---|---|---|
| BookBot orthographic CTC (greedy) | `run_transformers_inference.sh` + `bookbot-orthographic-ctc.yaml` | One-file inference passed and the existing plain-processor/argmax profile is unchanged. The local files lack Hugging Face download metadata, so rerun the pinned downloader only if snapshot certification is required. |
| BookBot orthographic CTC (packaged 5-gram) | `run_transformers_inference.sh` + `bookbot-orthographic-ctc-5gram.yaml` | Runnable now in the rebuilt shared image. Local-artifact loading and one-file GPU inference passed; metadata confirmed `Wav2Vec2ProcessorWithLM`, `BeamSearchDecoderCTC`, the 1,056,897,033-byte packaged LM, and the pinned package/settings provenance. On the same 4.096-second smoke item, greedy returned `a` and the word-LM decoder returned an empty hypothesis; use a development set, not held-out data, if settings are ever tuned. |
| BookBot phoneme CTC | `run_transformers_inference.sh` + `bookbot-phoneme-ctc.yaml` | One-file inference passed. Output is phonemic and must not be scored directly as orthographic WER. The local files lack Hugging Face download metadata. |
| Wav2Vec2-BERT Swahili | `run_transformers_inference.sh` + `w2v-bert-2.0-swahili-asr.yaml` | Complete snapshot; inference passed. |
| Paza Whisper large-v3-turbo | `run_transformers_inference.sh` + `paza-whisper-large-v3-turbo-sw.yaml` | Complete snapshot; inference passed. Caller must provide WAV files no longer than 30 seconds. The profile has no automatic segmentation or timestamp-based long-form fallback, so longer inputs may be truncated. |
| Whisper large | `run_transformers_inference.sh` + `whisper-large-sw.yaml` | Complete snapshot; GPU inference passed. Inputs above 30 seconds use Hugging Face's untruncated timestamp-based long-form path without modifying source audio. |
| Whisper large-v2 | `run_transformers_inference.sh` + `whisper-large-v2-sw.yaml` | Complete snapshot; inference passed. Inputs above 30 seconds use Hugging Face's untruncated timestamp-based long-form path without modifying source audio. |
| MMS-1B Swahili | `run_transformers_inference.sh` + `mms-1b-all-swh.yaml` | Complete Swahili package; inference and `swh` adapter loading passed. |
| BookBot Zipformer streaming RNN-T | `run_sherpa_onnx_inference.sh` + `zipformer-streaming-robust-sw-v4.yaml` | Moved out of the Transformers model root because it is a Sherpa-ONNX artifact. The shared CUDA image now includes Sherpa-ONNX; one-file GPU inference passed. Output is phonemic. |
| NeMo Exp41 | `run_nemo_inference.sh` + `swahili-exp41-ctc.yaml` | `model_exp41_avg.nemo` restored and inference passed. |
| Gemma 4 E2B | `run_multimodal_inference.sh` + `gemma-4-E2B-sw.yaml` | Uses the complete local BF16 snapshot through a separate prompt-driven audio adapter. Audio longer than Gemma's 30-second limit is decoded as ordered, non-overlapping 30-second chunks and rejoined into one shared output row. The existing image already has the required Transformers, Torch, Accelerate, and audio packages; no image rebuild or additional download is required. |

The decoder dependencies are additive and are imported only by a CTC profile
whose strategy is `beam_search`; they do not change greedy CTC, Whisper-style,
NeMo, post-processing, or evaluation selection. The current Compose inference
services reserve an NVIDIA GPU (`gpus: all`). A CPU-only deployment requires a
CPU image build plus removal or override of that Compose GPU reservation.

### Runnable in the current framework after preparation

| Model | Required preparation |
|---|---|
| HuBERT large-ls960-ft | Download the missing pinned snapshot. The existing CTC adapter and `hubert-large-ls960-ft-en.yaml` profile can then run it in the current image. This is English-only. |

### Requires a separate runtime image or additional model artifacts

| Model | What is still required |
|---|---|
| Paza Phi-4 multimodal | Runnable now in its dedicated image on the current hardware. Image build and one-file BF16 GPU inference passed with zero errors; the five-shard load took about 127 seconds and the 4.096-second file took about 16 seconds. The bounded auto-placement/offload path remains available if memory pressure changes. No model download is required. |
| Qwen2.5-Omni-7B | The dedicated text-only adapter/profile/image structure is implemented: the talker is disabled and `return_audio=False`. Its profile requires at least 40 GiB VRAM and deliberately rejects this 16 GiB host before loading weights. Build/run it only on suitable hardware; no model download is required. |
| Kaldi | Obtain the exact trained acoustic model, feature configuration, lexicon, language model, symbol tables, and decoding graph, then build a Kaldi service/adapter. The toolkit source alone is not a runnable ASR model. |
| `Swahili_exp1_100epochs` and `sw-tz-child-egra-fastconformer-ctc-110m` | Obtain complete NeMo weights or a `.nemo` artifact. The currently present tokenizer/configuration or documentation files are insufficient. |



## Straight forward steps

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
   - Place model artifacts under the `models/` directory owned by their runtime: `inference/nemo/`, `inference/transformers/`, `inference/sherpa_onnx/`, or `inference/multimodal/`.
   - Select a tracked YAML profile from the corresponding framework's `profiles/` directory. Profiles use relative artifact paths; inference never downloads models.

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

   Run either or both backends from the same segmented manifest. Each launcher
   creates and prints its own `<model>_<timestamp>` run name.

   NeMo:
   ```bash
   ./run_nemo_inference.sh \
     --model_config inference/nemo/profiles/swahili-exp41-ctc.yaml \
     --audio_manifest input_output_data/output/experiments/<dataset_name>/manifests/ref_manifest.raw_segments.jsonl
   ```

   Transformers (BookBot CTC):
   ```bash
   ./run_transformers_inference.sh \
     --model_config inference/transformers/profiles/bookbot-orthographic-ctc.yaml \
     --audio_manifest input_output_data/output/experiments/<dataset_name>/manifests/ref_manifest.raw_segments.jsonl \
     --batch_size 8
   ```

   For Whisper-family generation, use the same Transformers command with
   `inference/transformers/profiles/paza-whisper-large-v3-turbo-sw.yaml`.
   Both launchers require `--model_config` and write the same `transcriptions.jsonl`
   schema plus `run_metadata.json`.

6. **Build final segment-level manifest (attach `pred_text` from ASR) + clean**

   Generic:
   ```bash
   RUN_NAME=<model>_<timestamp>   # copy the value printed by inference
   ./run_manifest.sh \
     --dataset_root input_output_data/input/<dataset_name> \
     --manifest_base_in input_output_data/output/experiments/<dataset_name>/manifests/ref_manifest.raw_segments.jsonl \
     --asr_manifest input_output_data/output/transcripts/$RUN_NAME/transcriptions.jsonl
   ```

   This generates:
   - `input_output_data/output/evaluations/$RUN_NAME/manifests/ref_manifest.raw.jsonl`
   - `input_output_data/output/evaluations/$RUN_NAME/manifests/ref_manifest.clean.jsonl`

   When `--asr_manifest` is supplied, its hypotheses are authoritative: existing
   `pred_text` values in the base manifest are not retained. Manifest building and
   evaluation also fail fast if `ref_text` or `can_text` contains a known Unicode
   replacement/mojibake marker. This check does not reject valid IPA Unicode or
   alter model-produced `pred_text`.

   Run this step once for each model run. Smoke-test transcript paths are mapped
   to the parallel `smoke_tests/evaluations/$RUN_NAME/` directory automatically.

7. **Run evaluation from cleaned segment manifest**

   Generic:
   ```bash
   ./run_eval2.sh \
     --dataset_root input_output_data/input/<dataset_name> \
     --manifest_in input_output_data/output/evaluations/$RUN_NAME/manifests/ref_manifest.clean.jsonl
   ```

   Optional IPA comparison for any model:
   ```bash
   ./run_eval2.sh \
     --dataset_root input_output_data/input/<dataset_name> \
     --manifest_in input_output_data/output/evaluations/$RUN_NAME/manifests/ref_manifest.clean.jsonl \
     --scoring_representation ipa
   ```

   Native IPA hypotheses use their reviewed inventory adapter. Orthographic
   hypotheses are converted with the same pinned Africa G2P language and IPA
   inventory as CAN/REF. This writes a separate PER view under
   `input_output_data/output/evaluations/$RUN_NAME/ipa/`; it does not replace WER.

8. **Inspect the outputs** under `input_output_data/output/evaluations/$RUN_NAME/`:
   - `orthographic/egra_eval_detailed.csv` and
     `orthographic/egra_eval_summary.txt` for valid WER scoring.
   - `ipa/egra_eval_detailed.csv` and `ipa/egra_eval_summary.txt` for valid PER
     scoring.
   - `evaluation_metadata.json` inside each completed representation directory,
     recording its source manifest and compatibility status.
   - `orthographic_legacy/`, when explicitly requested, is retained for audit
     history only and must not be compared as a valid phoneme-model WER.
   - Summary folders inside each representation: `can_ref/`, `can_hyp/`, `ref_hyp/`.
9. **Explore results interactively**  
   - Dependencies: `pip install streamlit pandas numpy` (preferably inside a virtualenv).  
     - Specific example: `python3 -m venv .venv_streamlit && . .venv_streamlit/bin/activate && pip install --upgrade pip setuptools wheel && pip install streamlit pandas numpy`
   - Run: `streamlit run egra_dashboard2.py -- --csv <path/to/egra_eval_detailed.csv>`  
     - Specific example: ` . .venv_streamlit/bin/activate && streamlit run egra_dashboard2.py -- --csv input_output_data/output/evaluations/$RUN_NAME/orthographic/egra_eval_detailed.csv`
   - Open the browser tab (Streamlit serves on `http://localhost:8501` by default) to sort, group and aggregate metrics.

Everything runs in Docker setup (CPU-only or GPU-enabled).


## Full example for held-out data

The example assumes that data and models have been placed in:

- Data: `input_output_data/input/heldout_combined_fixed_20260525`
- Model profile: `inference/nemo/profiles/swahili-exp41-ctc.yaml`
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
        --model_config inference/nemo/profiles/swahili-exp41-ctc.yaml \
        --root_audio_dir input_output_data/output/experiments/heldout_combined_fixed_20260525_exp41/audio_segments

    # Copy the exact value printed by "[INFO] Model run:".
    RUN_NAME=swahili-exp41-ctc_<timestamp>

    ./run_manifest.sh \
        --dataset_root input_output_data/input/heldout_combined_fixed_20260525 \
        --manifest_base_in input_output_data/output/experiments/heldout_combined_fixed_20260525_exp41/manifests/ref_manifest.raw_segments.jsonl \
        --asr_manifest input_output_data/output/transcripts/$RUN_NAME/transcriptions.jsonl

    ./run_eval2.sh \
        --dataset_root input_output_data/input/heldout_combined_fixed_20260525 \
        --manifest_in input_output_data/output/evaluations/$RUN_NAME/manifests/ref_manifest.clean.jsonl

    . .venv_streamlit/bin/activate && streamlit run egra_dashboard2.py -- --csv input_output_data/output/evaluations/$RUN_NAME/orthographic/egra_eval_detailed.csv

---

## Working Modes

- Segment-only flow is the default documented flow:
- `run_segment.sh`: creates segment audio + segment manifest.
- `run_nemo_inference.sh`, `run_transformers_inference.sh`, `run_sherpa_onnx_inference.sh`, or `run_multimodal_inference.sh`: transcribes segment audio using a required model profile.
- `run_manifest.sh`: builds/cleans final segment-level manifest from `--manifest_base_in`.
- `run_eval2.sh`: scores only from an existing cleaned segment manifest (`--manifest_in`).

---

## Contents

- [Straight forward steps](#straight-forward-steps)
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
├── docker-compose.yml            # Three inference services plus egra-eval
├── manifest_pipeline.py          # Build+clean manifest entrypoint (used by run_manifest.sh)
├── eval_pipeline2.py             # Evaluation entrypoint from existing manifest (used by run_eval2.sh)
├── evaluation.py                 # Shared evaluation utilities and legacy combined entrypoint
├── infer.py                      # Legacy NeMo --model entrypoint
├── run_nemo_inference.sh         # Profile-driven NeMo launcher
├── run_transformers_inference.sh # Profile-driven Transformers launcher
├── run_sherpa_onnx_inference.sh  # Profile-driven Sherpa-ONNX launcher
├── run_eval2.sh                  # Wrapper script for evaluation (supports --manifest_in)
├── run_manifest.sh               # Wrapper script to build+clean manifest only (no scoring)
├── run_segment.sh                # Wrapper script for standalone manifest/audio segmentation
├── inference/
│   ├── common.py                 # Shared audio discovery, loading, duration, and resampling
│   ├── contracts.py              # Shared backend and transcription-result contracts
│   ├── profile.py                # Strict portable YAML profile loading and validation
│   ├── runner.py                 # Shared ordered inference and output writing
│   ├── nemo/
│   │   ├── infer.py              # NeMo framework entrypoint
│   │   ├── backend.py            # NeMo CTC/RNNT/hybrid backend
│   │   ├── profiles/             # Tracked NeMo model profiles
│   │   ├── models/               # Local NeMo artifacts (ignored, mounted read-only)
│   │   └── tmp/                  # Temporary NeMo segments
│   ├── transformers/
│   │   ├── infer.py              # Transformers framework entrypoint
│   │   ├── factory.py            # Adapter selection
│   │   ├── adapters/             # CTC and speech-seq2seq adapters
│   │   ├── profiles/             # Tracked Transformers model profiles
│   │   ├── models/               # Local Hugging Face artifacts (ignored, mounted read-only)
│   │   └── tmp/                  # Temporary Transformers files
│   └── sherpa_onnx/
│       ├── infer.py              # Sherpa-ONNX framework entrypoint
│       ├── backend.py            # Streaming transducer adapter
│       ├── profiles/             # Tracked Sherpa-ONNX profiles
│       └── models/               # Local ONNX artifacts (ignored, mounted read-only)
├── egra_eval2/                   # Evaluation, manifest, dataset-layout, and segmentation modules
│   ├── dataset_layout.py         # Discover dataset audio, TextGrid, and metadata paths
│   ├── manifest_builder.py       # Build reference manifests
│   ├── manifest_cleaner.py       # Normalize and clean manifest text
│   ├── nemo_manifest.py          # Load framework-neutral ASR JSONL outputs
│   ├── run_eval.py               # Core CAN/REF/HYP scoring
│   ├── scoring.py                # WER counts and accuracy metrics
│   ├── segmenter.py              # TextGrid-driven segmentation helpers
│   └── summarize.py              # Overall, macro, and per-learner summaries
├── tools/                        # Helper scripts (NeMo manifest prep, comparisons, etc.)
│   ├── make_ref_manifest.py      # Standalone reference manifest builder
│   └── ...
└── input_output_data/
│   ├── input/                    # place each dataset folder for every experiment here
│   └── output/                   # experiment results (one subfolder per run)
```

---

## What the pipeline does

**Inference (`inference/`)**
- Loads a required, framework-specific YAML profile and resolves its artifact from the matching `models/` directory.
- Reads exact segments from `--audio_manifest` or recursively discovers `.wav` files under `--root_audio_dir`; the NeMo legacy API also retains dataset discovery and optional TextGrid-driven segmentation.
- Resamples audio to the model's required sample rate, then dispatches to the selected NeMo, Transformers CTC, or Transformers speech-seq2seq backend.
- Emits the unchanged `transcriptions.jsonl` schema and a separate `run_metadata.json` record.

**Manifest build (`manifest_pipeline.py`)**
- In segment-only flow, loads a base segment manifest from `--manifest_base_in`.
- Attaches ASR hypotheses from `--asr_manifest` (optional; `--nemo_manifest` is a legacy alias).
- Writes cleaned segment manifests (for example `ref_manifest.segment.raw.jsonl` and `ref_manifest.segment.clean.jsonl`).

**Evaluation (`eval_pipeline2.py`)**
- Loads the cleaned manifest via `--manifest_in` and attaches `REF/CAN/HYP` by audio key.
- Computes CAN/REF, CAN/HYP, REF/HYP metrics and advanced summaries.
- Produces a **detailed CSV**, text summary, and per-alignment summary folders (`can_ref/`, `can_hyp/`, `ref_hyp/`).

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

Use the framework launcher for the selected profile. Each launcher requires
`--model_config`, accepts either `--audio_manifest` or `--root_audio_dir`, and
writes to `input_output_data/output/transcripts/<model>_<timestamp>/` by default.
The scripts run `docker compose run` with your current `uid:gid`, so generated
files inside `input_output_data` are owned by the host user.

NeMo example:
```bash
./run_nemo_inference.sh \
  --model_config inference/nemo/profiles/swahili-exp41-ctc.yaml \
  --root_audio_dir input_output_data/input/<dataset>/0_Audio
```

Transformers speech-seq2seq example:
```bash
./run_transformers_inference.sh \
  --model_config inference/transformers/profiles/paza-whisper-large-v3-turbo-sw.yaml \
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

Pass the segment manifest to either framework launcher. This preserves manifest
ordering and avoids rediscovering files:

```bash
./run_nemo_inference.sh \
  --model_config inference/nemo/profiles/swahili-exp41-ctc.yaml \
  --audio_manifest input_output_data/output/<experiment>/manifests/ref_manifest.raw_segments.jsonl
```

```bash
./run_transformers_inference.sh \
  --model_config inference/transformers/profiles/bookbot-orthographic-ctc.yaml \
  --audio_manifest input_output_data/output/<experiment>/manifests/ref_manifest.raw_segments.jsonl \
  --batch_size 8
```

### 5) Build final segment manifest (REF/CAN/HYP source for evaluation)

We provide `run_manifest.sh`. It will:
- Load base segment manifest from `--manifest_base_in` (preserve segment granularity).
- Attach ASR hypotheses from `--asr_manifest`.
- Derive `evaluations/<model>_<timestamp>/` from a standard transcript path.
- Write raw and cleaned segment manifests under that run's `manifests/` folder.

Usage:
```bash
./run_manifest.sh \
  --dataset_root /io/input/<dataset> \
  --manifest_base_in /io/output/<experiment>/manifests/ref_manifest.raw_segments.jsonl \
  --asr_manifest /io/output/transcripts/<model>_<timestamp>/transcriptions.jsonl
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
  --manifest_in /io/output/evaluations/<model>_<timestamp>/manifests/ref_manifest.clean.jsonl
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
`input_output_data/output/evaluations/<model>_<timestamp>/` (or the parallel
`smoke_tests/evaluations/` path). Each run folder contains:

1. **`egra_eval_detailed.csv`** — One row per EGRA item with:
   - Keys: `learner_id`, `audio_type`, `audio_file`.
   - Texts: `CAN` (canonical), `REF` (annotator), `HYP` (ASR).
   - **CAN vs REF** metrics: `WER_can_ref`, `ACC_can_ref (EGRA_ACC)` plus counts `S_can_ref`, `D_can_ref`, `I_can_ref`, `C_can_ref (EGRA_COR)`, `N_can_ref`.
   - **CAN vs HYP** metrics: `WER_can_hyp`, `ACC_can_hyp (ASR_EGRA_ACC)` plus counts `S_can_hyp`, `D_can_hyp`, `I_can_hyp`, `C_can_hyp (ASR_EGRA_COR)`, `N_can_hyp`.
   - **REF vs HYP** metrics: `WER_ref_hyp`, `ACC_ref_hyp` plus counts `S_ref_hyp`, `D_ref_hyp`, `I_ref_hyp`, `C_ref_hyp`, `N_ref_hyp`.
   - Per-row aggregates: `EGRA-COR`, `EGRA-ACC`, `ASR-EGRA-COR`, `ASR-EGRA-ACC`, `MAE_EGRA_COR`, `ASR_WER`.
   - Column names that include aliases (e.g., `ACC_can_ref (EGRA_ACC)`) expose both the base metric and the specific EGRA naming.
   - WER and ACC values are percentages (0–100); the raw counts are absolute integers.
   - **Agreement**: `MAE_EGRA_COR = |EGRA_COR − ASR_EGRA_COR|` which represents the absolute difference in number of correct tokens between annotator-based and ASR-based evaluations.
   - All learner metadata merged in (e.g., `gender`, `age`).

2. **`egra_eval_summary.txt`** — Six-line global snapshot with the metrics `EGRA-COR`, `EGRA-ACC`, `ASR-EGRA-COR`, `ASR-EGRA-ACC`, `MAE_EGRA_COR`, and `ASR_WER` (averages where applicable), rounded to two decimals.

3. **Pair-specific summary folders** — within the same experiment directory you will find three
   subfolders:

   | Folder | Alignment pair | Files inside |
   |--|--|--|
   | `can_ref/` | Canonical vs Reference (annotator EGRA) | `egra_eval_summary_per_speaker_global.csv`, `egra_eval_summary_per_speaker_macro.csv`, `egra_eval_summary_per_speaker_subcat.csv` |
   | `can_hyp/` | Canonical vs ASR hypothesis (automated EGRA) | same filenames as above |
   | `ref_hyp/` | Reference vs ASR hypothesis (ASR quality) | same filenames as above |

   Each summary file reports **micro-averages** derived from the raw counts:
   - `*_per_speaker_global.csv` — one row per `learner_id` plus a leading `__GLOBAL__` row aggregating every sample.
   - `*_per_speaker_macro.csv` — per learner × macro category (letters / syllables / nonwords / passage).
   - `*_per_speaker_subcat.csv` — per learner × macro category × subcategory (e.g., `letters` + `isolated`).

   The columns mirror the metric block in the detailed CSV (WER, ACC, counts). Use them to compare annotator vs ASR EGRA scores or inspect performance by task type.

Use these artifacts to track:
- Human annotator performance (`can_ref`).
- Automated EGRA performance (`can_hyp`).
- ASR quality with respect to the human reference (`ref_hyp`).
- Agreement between automated and human EGRA via `MAE_EGRA_COR` (closer to 0 is better).

---

## Metrics & definitions

All metrics are computed after **text normalization** (`normalize/textnorm.py`): NFC Unicode, lowercase, punctuation removed, whitespace collapsed.

We compute standard ASR alignment counts via `jiwer`:
- **S** — substitutions  
- **D** — deletions  
- **I** — insertions  
- **C** — correct matches 
- **N** — number of total reference tokens (groundtruth)

From those we derive:

- **WER** = (S + D + I) / N → reported in the CSVs as a **percentage** (value × 100).
- **ACC** = C / N → also reported as a **percentage** in the detailed and summary files.

We apply the same counts to derive **EGRA-style** KPIs:

- **EGRA (Annotator-based)** from **ANN/REF as truth vs CAN as hypothesis**  
  - `C_can_ref (EGRA_COR) = N_ann − S_can_ref − D_can_ref`
  - `ACC_can_ref (EGRA_ACC) = EGRA_COR / N_ann`
  - In segment mode, `REF`/`HYP` are concatenated per original audio item before CAN-side scoring.

- **ASR-based EGRA** from **CAN vs HYP**  
  - `C_can_hyp (ASR_EGRA_COR) = N_can − S_can_hyp − D_can_hyp`
  - `ACC_can_hyp (ASR_EGRA_ACC) = ASR_EGRA_COR / N_can`

- **Agreement** between annotator- and ASR-based correctness  
  - `MAE_EGRA_COR = |EGRA_COR − ASR_EGRA_COR|`

- **ASR quality snapshot**  
  - `ASR_WER = WER_ref_hyp` (same computation exposed for convenience in the detailed CSV and summary text).

- **ASR quality vs human** from **REF vs HYP**  
  - `WER_ref_hyp`, `ACC_ref_hyp` and the count fields `S_ref_hyp`, `D_ref_hyp`, `I_ref_hyp`, `C_ref_hyp`, `N_ref_hyp`.

---

### Metric ranges & units

WER and ACC values are emitted as **percentages** (0.0–100.0). Count-based columns (`S/D/I/C/N`) remain raw integers.

| Metric | Description | Typical Range / Unit | Interpretation |
|:--|:--|:--|:--|
| **WER_can_ref**, **WER_can_hyp**, **WER_ref_hyp** | Word Error Rate (substitutions + deletions + insertions) / N | 0.0–100.0 (%); can exceed 100 with many insertions | Lower is better |
| **ACC_can_ref (EGRA_ACC)**, **ACC_can_hyp (ASR_EGRA_ACC)**, **ACC_ref_hyp** | Accuracy = C / N | 0.0–100.0 (%) | Higher is better |
| **C_can_ref (EGRA_COR)**, **C_can_hyp (ASR_EGRA_COR)** | Correctness count = N − S − D | Integer ≥ 0 | Count of correct tokens |
| **S_\***, **D_\***, **I_\***, **C_\***, **N_\*** | Alignment counts (Substitutions, Deletions, Insertions, Correct, Total) | Integers ≥ 0 | Raw counts |
| **MAE_EGRA_COR** | Absolute difference between EGRA_COR and ASR_EGRA_COR per row | Integer ≥ 0 | Lower indicates better agreement |
| **ASR_WER** | Word error rate from REF vs HYP (duplicate of `WER_ref_hyp`) | 0.0–100.0 (%) | Lower is better |

**Note:**  
If the canonical or reference text has `N = 0`, ratio-based metrics (WER, ACC) are undefined and will appear as `NaN` in the output CSVs.

---


## Configuration knobs

### Inference profiles and launchers

- **Model profile**: `--model_config <profile.yaml>` is required by all framework launchers. Profiles define `framework`, `adapter`, relative `artifact`, language/task, loader settings, decoding strategy, and structured parameter evidence. Images supply compatible dependencies but do not choose model-specific inference behavior; see [model profile evidence implementation](docs/model-profile-evidence-implementation.md).
- **Invocation controls**: batch size, thread/worker counts, input selection, and output roots remain launcher arguments and are recorded in run metadata rather than being hidden in an image or treated as model hyperparameters.
- **Model storage**: place artifacts below the owning framework's `models/` directory. `ASR_MODEL_ROOT` overrides that default root; Compose sets it to the read-only `/models` mount.
- **Input**: provide exactly one of `--audio_manifest <segments.jsonl>` or `--root_audio_dir <audio-directory>`. NeMo retains `--dataset_root` and `--dataset_annotator` for legacy dataset discovery and optional TextGrid segmentation.
- **Output base**: inference defaults to `input_output_data/output`; `--output_root <directory>` changes that base. The runner creates `transcripts/<model>_<UTC timestamp>/` below it, or `smoke_tests/transcripts/...` with `--smoke_test`.
- **Runtime controls**: NeMo retains its CPU worker, temporary-segment, decoder, and debug controls; Transformers retains `--batch_size`; Sherpa-ONNX adds `--num_threads`. The source, evidence strength, hardware assumptions, and known historical gaps for these values are recorded in [inference runtime parameter provenance](docs/inference-runtime-parameter-provenance.md).
- **Offline operation**: profiles require local artifacts and `local_files_only: true`; downloading a model is a separate preparation step.

Root `infer.py` is the only legacy NeMo API and continues to accept `--model`
directly. It also remaps the former NeMo model-directory prefix when that old
path no longer exists. Do not use direct model paths with the new launchers.

### Manifest build (`manifest_pipeline.py`)
The dataset and base segment manifest remain explicit. The evaluation output is
derived from a standard transcript path unless you override it.

Run `python3 manifest_pipeline.py --help` to see available options. Highlights:
- `--dataset_root /io/input/<dataset>` — required; automatically discovers the `Student_*` CSVs plus `0_Audio/` and `2_TextGrid/`.
- `--manifest_base_in /io/output/<experiment>/manifests/ref_manifest.raw_segments.jsonl` — required in segment-only flow; preserves segment rows.
- `--output_root /io/output/evaluations/<model>_<timestamp>` — optional exact destination; otherwise derived from `--asr_manifest` by replacing `transcripts` with `evaluations`.
- `--asr_manifest /path/to/transcriptions.jsonl` — optional; if provided, `pred_text` is attached from either backend. `--nemo_manifest` remains an alias.

### Evaluation (`eval_pipeline2.py`)
Run `python3 eval_pipeline2.py --help` to see available options. Highlights:
- `--dataset_root /io/input/<dataset>` — required.
- `--manifest_in /io/output/<experiment>/manifests/ref_manifest.segment.clean.jsonl` — required.
- `--output_root /io/output/evaluations/<model>_<timestamp>` — optional run
  destination; otherwise the owning evaluation run is derived from
  `--manifest_in`. The evaluator appends `orthographic/`, `ipa/`, or
  `orthographic_legacy/`. Custom CSV and summary paths are rejected if they
  escape that representation directory.

---

## Troubleshooting

- **No GPU used**: Rebuild the default CUDA image, confirm Docker can access the NVIDIA GPU, and check that the resolved ASR service retains `gpus: all`.
- **Empty or short `pred_text`**: Check that the model matches the language/domain. Also verify sample rate conversion (the script resamples to 16 kHz automatically).
- **Missing REF text in segment base manifest**: Ensure `run_segment.sh` used the correct `--textgrid_root` and that audio/TextGrid stems align.
- **Segment ASR not attached**: Check that `--asr_manifest` in `run_manifest.sh` points to segmented ASR output and that `--match_on` is appropriate.
- **Segmentation not applied in inference**: Ensure matching `.TextGrid` files exist under `2_TextGrid/` and names align with audio stems; segmentation follows all parsed intervals from TextGrid.
- **Unexpected full rows in segment manifest**: re-run `run_segment.sh`; strict mode drops non-segmentable rows and writes only `*_segmentN.wav` entries.
- **Profile rejected before loading**: Check for unknown keys, an invalid framework/adapter/decoding combination, an absolute or escaping artifact path, or a missing local artifact.
- **Wrong Transformers decoding path**: CTC profiles must use the `ctc` adapter with greedy decoding; Whisper-family profiles must use `speech_seq2seq` with `generate`.
- **Permissions**: The repo root and `input_output_data` are mounted read-write. Each inference service mounts its own framework's `models/` directory read-only at `/models`.

---

## Source files

- **`inference/common.py`, `inference/contracts.py`, `inference/profile.py`, `inference/runner.py`**
  Define common audio handling, backend/result contracts, strict portable model profiles, ordered inference, and the shared `transcriptions.jsonl` plus `run_metadata.json` outputs.

- **`inference/nemo/`**
  Implements profile-driven NeMo CTC/RNNT/hybrid inference while retaining existing dataset discovery and optional TextGrid behaviour.

- **`inference/transformers/`**
  Selects between the greedy CTC adapter and the Whisper-style speech-seq2seq adapter. Model-specific behaviour, including BookBot plain processing and MMS language-adapter selection, is profile-driven.

- **`inference/sherpa_onnx/`**
  Runs streaming ONNX transducers through Sherpa-ONNX while preserving the same profile, manifest, progress, and output contracts.

- **`infer.py`**  
  Preserves the legacy NeMo `--model` command and delegates to the reorganized NeMo implementation.

- **`manifest_pipeline.py`**  
  Builds and cleans final manifests; in segment-only flow it loads `--manifest_base_in` and attaches ASR `pred_text` from `--asr_manifest`.

- **`eval_pipeline2.py`**
  Runs scoring and report generation using only a cleaned manifest (`--manifest_in`) plus dataset metadata CSVs.

- **`egra_eval2/scoring.py`**
  Wraps `jiwer` to produce counts (**S, D, I, C, N**), **WER** and **ACC** (all expressed as percentages in downstream outputs). Uses `egra_eval2/textnorm.py` for simple text normalization.

- **`egra_eval2/textgrid_io.py`**
  Finds the requested tier case-insensitively (default `child`), gathers labeled intervals, strips filler tags (`<unk>`, `<noise>`, etc.), and concatenates labels to form **REF** per item while searching recursively across annotator folders.

- **`egra_eval2/linking.py`**
  Builds join keys from the EGRA CSV (`audio_name`, `audio_stem`) and attaches ASR HYPs by the chosen key (`stem` by default).

- **`egra_eval2/nemo_manifest.py`**
  Loads one or many NeMo manifests (JSONL), extracting `audio_path`, `audio_name`, `audio_stem` and `hyp_text`.

- **`egra_eval2/dataset_layout.py`**
  Utility helpers that discover dataset packages containing `0_Audio/`, `2_TextGrid/` and the `Student_*` CSVs.

- **`egra_eval2/passage_merge.py`**
  Parses the passages CSV (various encodings handled), extracts `passage_num` and fills missing `canonical_text` for `passage_numX` rows.

- **`egra_eval2/summarize.py`**
  Builds micro-averaged summaries for each alignment pair:
  - `summary_for_pair(df, prefix, by=None)` — aggregates metrics for one of `can_ref`, `can_hyp`, or `ref_hyp` (optionally grouped by columns).
  - `summary_per_speaker(df, prefix)` — per learner.
  - `summary_per_speaker_macro(df, prefix)` — per learner × macro category.
  - `summary_per_speaker_subcategory(df, prefix)` — per learner × macro category × subcategory.

- **`egra_eval2/eval_utils.py`**
  Shared evaluation helpers for letter canonical normalization and advanced metrics (MER + fine-grained P/R/F1 via `dp_align`).

- **`egra_eval2/segmenter.py`**
  Shared TextGrid segmentation module used by inference; parses interval blocks and cuts audio without label/tier filtering.

- **`docker/Dockerfile`**  
  Debian 12 base with PyTorch (CPU or CUDA), NeMo ASR 2.3, Transformers, Sherpa-ONNX, and pinned inference dependencies.

- **`docker-compose.yml`**  
  Seven services:
  - `nemo-asr`: run profile-driven NeMo inference.
  - `transformers-asr`: run profile-driven CTC or speech-seq2seq inference.
  - `sherpa-onnx-asr`: run profile-driven streaming ONNX transducer inference.
  - `multimodal-asr`: run profile-driven multimodal audio-to-text generation.
  - `phi4-multimodal-asr`: run Paza Phi-4 with bounded automatic GPU/CPU/disk placement.
  - `qwen-omni-asr`: run Qwen text-only inference on a GPU with at least 40 GiB VRAM.
  - `egra-eval`: run manifest build/evaluation (`manifest_pipeline.py`, `eval_pipeline2.py`).
  Mounts the repo as `/work`, data as `/io`, and each inference framework's models read-only as `/models`.

- **`run_nemo_inference.sh` / `run_transformers_inference.sh` / `run_sherpa_onnx_inference.sh` / `run_multimodal_inference.sh` / `run_phi4_multimodal_inference.sh` / `run_qwen_omni_inference.sh` / `run_segment.sh` / `run_manifest.sh` / `run_eval2.sh`**
  Thin wrappers that run the appropriate Compose service and command. Every inference wrapper requires `--model_config`.
- **`run_nemo_offline_eval.sh`**  
  Generates normalized REF/CAN manifests and runs NVIDIA NeMo’s own `speech_to_text_eval.py` script for REF↔HYP and CAN↔HYP scoring. Handy for cross-checking the internal metrics against the official NeMo implementation.
