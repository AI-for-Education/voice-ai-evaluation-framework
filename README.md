The purpose of this project is to evaluate NeMo ASR models on the task of early grade reading assessments (EGRA) for kiswahili child speech.

**Input (you need to provide!):**
- NeMo ASR model ([a first model provided](https://drive.google.com/file/d/1NQTC8532QluX7KXQNGcebKj9FseUzrO-))
- dataset of kiswahili child speech comprising:
  - audio files, 
  - cannonical texts, i.e. what the child should have uttered and 
  - reference text, i.e. what the child actually uttered

**Output:**
- KPIs to evaluate child:
  - EGRA-COR - the EGRA-style correctness; based on the canonical text and the reference text
  - EGRA-ACC - the EGRA accuracy; based on the canonical text and the reference text
- KPIs to evaluate the ASR (used instead of an ennumerator):
  - ASR-EGRA-COR - the EGRA-style correctness; uses ASR transcripts instead of reference texts
  - ASR-EGRA-ACC - the EGRA accuracy; uses ASR transcripts are instead of reference texts
  - MAE_EGRA_COR - the mean absolute error of the EGRA correctness when using ASR transcripts instead of reference texts
  - ASR_WER - the word error rate for the ASR model

The project transcribes the audio files in the input dataset using the input ASR model and computes the KPIs listed above.

There are 43 recordings per child that can be categorised into 7 tasks being tested. The [Task Mapping TSV](tools/task_mapping.tsv) organises the recordings into the respective task categories to be used for calculating the egra_eval_summary.txt.


Metrics needed in summary.txt
Passage and grid reading (T1, T2, T4, T6 each need all of these metrics: wer_ref_hyp, r, scatter plot, MAE_correct_counts, MER, subs_prec, subs_r, subs_f1, insert_p, insert_r, insert_f1, del_prec, del_r, del_f1, mistakes_prec, mistakes_r, mistakes_f1). More info about these are given below:
ASR WER: wer_ref_hyp
EGRA:
Correlation coefficient r (see next slide) between predicted (hyp-ref) and actual number of correct (can-ref) words per utterance (segment)
Scatter plot in code
MAE between correct counts. This is MAE between EGRA_ACC and ASR_EGRA_ACC (not normalised)
EGRA_ACC = can_ref
ASR_EGRA_ACC = hyp_ref
Finer grained:
Mistake error rate (MER)
Substitution P, R, F1
Insertion P, R, F1
Deletion P, R, F1
All mistakes P, R, F1
Isolated letters, syllables and non-words (T3, T5, T7 and each has the metrics: wer_ref_hyp, egra_acc, corr_mistake_pred_prec,  corr_mistake_pred_r,  corr_mistake_pred_f1, baseline_prec, baseline_r, baseline_f1):
ASR WER: wer_ref_hyp
EGRA accuracy = {TP + TN}/{N}
P, R, F1 for correct mistake prediction (label 1 = {mistake})
P, R, F1 for majority baseline

Context

This repository evaluates an ASR (Automatic Speech Recognition) system for Swahili children completing EGRA-style speech tests.

Each child performs 42 tests, grouped into 7 categories (T1–T7).
The mapping from test → category is defined in the project README.

For each test item, we have three string forms:

Canonical — the target / intended word shown to the child.

Reference (REF) — what the child actually said, human-annotated.

Hypothesis (HYP) — what the ASR system predicted the child said.

The current evaluation pipeline already produces an egra_eval_summary.txt file with several metrics.

Additional metrics (described in the README under Task for New Metrics) and ensure they appear in the generated egra_eval_summary.txt, aggregated per test category (T1–T7) and overall.

New Metrics to Add

The README defines multiple phonological metrics that must now be computed using the canonical, reference, and hypothesis forms.

A typical example:

Substitution Precision Example

True substitutions = differences between reference and canonical
Predicted substitutions = differences between hypothesis and canonical
Metric = How well HYP predicts the same substitutions that REF made.

Each metric follows this pattern:
Compare REF vs CANONICAL → child’s true phonological process
Compare HYP vs CANONICAL → system’s predicted phonological process

Compute true positives, false positives, false negatives

Derive:
Precision
Recall
F1-score

Counts as needed (TP, FP, FN)

These metrics must be computed inside each test category and optionally aggregated across all tests.

---

## Straight forward steps


1. **Build the Docker image** (CPU by default):  
   `docker compose build`
2. **Prepare the dataset and model**  
   - Copy the dataset (including `0_Audio/`, `2_TextGrid/`, `Student_Full_Canonical_EGRA_*.csv`, `Student_MetaData_EGRA_*.csv` into `input_output_data/input/<dataset_name>/`. Use the oral passages file from [this link](https://drive.google.com/file/d/1n50XR0TD557eYD2bkKmJ6Uem5n3SLJOc/view?usp=sharing) and place it in `input_output_data/input/oral_passages.csv`.  
   - Download your NeMo ASR model (the default scripts expect [Swahili_exp1_100epochs.nemo](https://drive.google.com/file/d/1NQTC8532QluX7KXQNGcebKj9FseUzrO-/view?usp=sharing)) and place it in `nemo_inference/models/`.
3. **Run inference**  
   ```bash
   ./run_inference.sh \
     --dataset_root input_output_data/input/<dataset_name> \
     --output_dir input_output_data/output/<dataset_name>/nemo_asr_output \
     --model nemo_inference/models/<model>.nemo
   ```
  Example
  ```bash
  ./run_inference.sh \
     --dataset_root input_output_data/input/1_Batch2_Data_16spk_subset \
     --output_dir input_output_data/output/1_Batch2_Data_16spk_subset/nemo_asr_output \
     --model nemo_inference/models/Swahili_exp1_100epochs.nemo
   ```
4. **Run evaluation**  
   ```bash
   ./run_eval.sh \
     --dataset_root input_output_data/input/<dataset_name> \
     --passages_csv input_output_data/input/oral_passages.csv \
     --nemo_manifest input_output_data/output/<dataset_name>/nemo_asr_output/transcriptions.jsonl
   ```
    Example
    ```bash
    ./run_eval.sh \
      --dataset_root input_output_data/input/1_Batch2_Data_16spk_subset \
      --passages_csv input_output_data/input/oral_passages.csv \
      --nemo_manifest input_output_data/output/1_Batch2_Data_16spk_subset/nemo_asr_output/transcriptions.jsonl
    ```
    
5. **Inspect the outputs** under `input_output_data/output/experiments/<experiment>/`:  
   - `egra_eval_detailed.csv` (very detailed evaluation, all metrics for each audio file)  
   - `egra_eval_summary.txt` (6-line metrics global summary)  
   - Summary folders: `can_ref/`, `can_hyp/`, `ref_hyp/`
6. **Explore results interactively**  
   - Dependencies: `pip install streamlit pandas numpy` (preferably inside a virtualenv).  
     - Specific example: `python3 -m venv .venv_streamlit && . .venv_streamlit/bin/activate && pip install --upgrade pip setuptools wheel && pip install streamlit pandas numpy`
   - Run: `streamlit run egra_dashboard.py -- --csv <path/to/egra_eval_detailed.csv>`  
     - Specific example: ` . .venv_streamlit/bin/activate && streamlit run egra_dashboard.py -- --csv input_output_data/output/experiments/exp1/egra_eval_detailed.csv`
   - Open the browser tab (Streamlit serves on `http://localhost:8501` by default) to sort, group and aggregate metrics.

Everything runs in Docker setup (CPU-only or GPU-enabled).

---

## Contents

- [Straight forward steps](#straight-forward-steps)
- [Contents](#contents)
- [Project structure](#project-structure)
- [What the pipeline does](#what-the-pipeline-does)
- [Input data format](#input-data-format)
- [How to run (Docker)](#how-to-run-docker)
  - [1) Build the image](#1-build-the-image)
  - [2) Run inference (ASR)](#2-run-inference-asr)
  - [3) Run evaluation](#3-run-evaluation)
  - [4) Optional: compare with NeMo offline scoring](#4-optional-compare-with-nemo-offline-scoring)
- [Outputs \& how to interpret them](#outputs--how-to-interpret-them)
- [Metrics \& definitions](#metrics--definitions)
  - [Metric ranges \& units](#metric-ranges--units)
- [Configuration knobs](#configuration-knobs)
  - [Inference (`infer.py`)](#inference-inferpy)
  - [Evaluation (`evaluation.py`)](#evaluation-evaluationpy)
- [Troubleshooting](#troubleshooting)
- [Source files](#source-files)

---

## Project structure

```
.
├── docker/
│   └── Dockerfile                # Base image with PyTorch, NeMo, audio libs, pandas, jiwer, praatio, librosa, etc.
├── docker-compose.yml            # Compose with two services: nemo-asr (inference), egra-eval (evaluation)
├── evaluation.py                 # Main entrypoint for evaluation & summaries
├── infer.py                      # Main entrypoint for NeMo-based transcription
├── run_eval.sh                   # Wrapper script for evaluation (dataset_root + output_root mandatory)
├── run_inference.sh              # Wrapper script for inference (dataset_root + output_root + model mandatory)
├── egra_eval/
│   ├── data/
│   │   ├── linking.py            # Build keys, attach HYPs to EGRA rows
│   │   ├── nemo_manifest.py      # Load NeMo manifests (JSONL)
│   │   ├── passage_merge.py      # Fill missing canonical passages from CSV
│   │   └── textgrid_io.py        # Read REF text from TextGrid tiers (recursive search, filler-tag filtering)
│   ├── eval/
│   │   └── run_eval.py           # Core scoring module (CAN/REF/HYP)
│   ├── metrics/
│   │   └── scoring.py            # Normalization + WER counts + ACC, P/R/F1
│   ├── normalize/
│   │   └── textnorm.py           # Simple text normalization (lowercase, remove punctuation, collapse spaces)
│   └── report/
│       └── summarize.py          # Summaries: macro, per-learner, overall, etc.
├── tools/                        # Helper scripts (NeMo manifest prep, comparisons, etc.)
├── input_output_data/
│   ├── input/                    # place each dataset folder for every experiment here
│   └── output/                   # experiment results (one subfolder per run)
└── nemo_inference/
    ├── models/                   # NeMo .nemo models (mounted read-only in container)
    └── tmp/                      # Temporary 16kHz segments dumped during inference (passage slicing). Use "debug" argument for inference (infer.py) in order to keep them for inspection.
```

---

## What the pipeline does

**Inference (`infer.py`)**
- Recursively scans the dataset root (either `--dataset_root` or `--root_audio_dir`) for `.wav` files.
- Resamples audio to 16 kHz as needed and, if a matching TextGrid exists (default tier `child`), slices the audio according to the intervals before transcription.
- Emits a NeMo-style JSONL manifest containing `audio_filepath`, `duration` and `pred_text`.

**Evaluation (`evaluation.py`)**
- Discovers the student CSVs, audio and TextGrid folders from `--dataset_root` (or explicit `--egra_csv`, `--meta_csv`, etc.).
- Recursively searches `2_TextGrid/` for `.TextGrid` files, selects the first match for each audio stem, and strips filler tags such as `<unk>`, `<noise>`, `<um>`, etc. from the REF transcript.
- Automatically normalizes canonical letter prompts so consonants receive a trailing `a` (e.g., `g -> ga`) prior to scoring.
- Attaches ASR hypotheses from the provided manifest(s) and computes metrics for:
  - **CAN vs REF** (annotator-based EGRA).
  - **CAN vs HYP** (ASR-based EGRA).
  - **REF vs HYP** (ASR quality vs human).
- Produces a **detailed CSV**, a 6-line text summary, and per-alignment summary folders (`can_ref/`, `can_hyp/`, `ref_hyp/`).

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
- If the dataset root already contains `nemo_asr_output/transcriptions.jsonl`, `evaluation.py` will attach it automatically unless you override with `--nemo_manifest`.

## How to run (Docker)

> You need Docker and (optionally) NVIDIA Container Toolkit for GPU.

### 1) Build the image

CPU-only (default):
```bash
docker compose build
```

GPU-enabled build (CUDA 12.1 wheels):
```bash
docker compose build --build-arg TORCH_CUDA=cu121
```
> At runtime, enable GPU by uncommenting `gpus: "all"` in `docker-compose.yml` (service `nemo-asr`) **or** pass `--gpus all` to `docker compose run`.

### 2) Run inference (ASR)

We provide `run_inference.sh`. It will:
- Discover audio/TextGrid folders from `--dataset_root` (or use the explicit paths you supply).
- Resample/slice audio as needed and run the NeMo model (`--model`).
- Write a manifest (`transcriptions.jsonl`) under the chosen output folder.
The script runs `docker compose run` with your current `uid:gid`, so all generated files inside `input_output_data` are owned by the host user.

Usage:
```bash
./run_inference.sh \
  --dataset_root /io/input/<dataset> \
  --output_dir /io/output/<dataset>/nemo_asr_output \
  --model /models/<model>.nemo
```



> To use GPU at run time: add `--gpus all` after `docker compose run` or enable `gpus: "all"` in the compose file.

### 3) Run evaluation

We provide `run_eval.sh`. It will:
- Locate the canonical/meta CSVs plus the audio/TextGrid folders (via `--dataset_root` or explicit paths).
- Attach ASR hypotheses from the given manifest(s).
- Read reference transcripts by searching all TextGrid folders and matching on audio stem.
- Produce the detailed CSV, the text summary, and per-pair summary folders in the chosen output directory.
Like the inference wrapper, it executes the container with your user ID so the resulting CSVs and summaries remain writable without sudo.

Usage:
```bash
./run_eval.sh \
  --dataset_root /io/input/<dataset> \
  --output_root /io/output/<experiment> \
  --passages_csv /io/input/<dataset>/oral_passages.csv \
  --nemo_manifest /io/output/<dataset>/nemo_asr_output/transcriptions.jsonl
```


### 4) Optional: compare with NeMo offline scoring

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

All evaluation outputs land in the experiment folder you pass as `<output_root>`. Each experiment
folder contains:

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

- **EGRA (Annotator-based)** from **CAN vs REF**  
  - `C_can_ref (EGRA_COR) = number of correct tokens = N_ref − S_can_ref − D_can_ref` 
  - `ACC_can_ref (EGRA_ACC) = EGRA_COR / N_ref`

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

### Inference (`infer.py`)
- **Model path**: `--model /models/your_model.nemo`
- **Dataset root**: `--dataset_root /io/input/<dataset>` — required.
- **Annotator**: `--dataset_annotator <annotatorName>` to pick a specific annotator (defaults to the first alphabetically).
- **Output root**: `--output_root /io/input/<dataset>/nemo_asr_output` — where `transcriptions.jsonl` is written.
- **TextGrid tier**: `--tier_name child`; change this if your intervals live on a different tier.
- **CPU workers**: `--cpu_workers N` sets the number of CPU threads used when no GPU is available.
- **Temp segments**: `--tmp_dir /work/nemo_inference/tmp` lets you keep the 16 kHz segments around for debugging.

> `run_inference.sh` requires the named options `--dataset_root`, `--output_dir`, and `--model`; add any extra flags after those.

### Evaluation (`evaluation.py`)
No implicit defaults are applied to dataset/output paths—provide them explicitly.

Run `python3 evaluation.py --help` to see available options. Highlights:
- `--dataset_root /io/input/<dataset>` — required; automatically discovers the `Student_*` CSVs plus `0_Audio/` and `2_TextGrid/`.
- `--passages_csv /io/input/<dataset>/oral_passages.csv` — required; supplies the passage text mapping for passage tasks.
- `--output_root /io/output/<experiment>` — required; directory where results are written.
- `--nemo_manifest /path/to/transcriptions.jsonl` — attach one or more ASR manifests.
- `--summary_can_ref_dir`, `--summary_can_hyp_dir`, `--summary_ref_hyp_dir` — optional overrides for the summary output destinations.

---

## Troubleshooting

- **No GPU used**: Ensure the image was built with `--build-arg TORCH_CUDA=cu121` **and** you run with `--gpus all` or `gpus: "all"` in compose.
- **Empty or short `pred_text`**: Check that the model matches the language/domain. Also verify sample rate conversion (the script resamples to 16 kHz automatically).
- **Missing REF text**: Ensure a `.TextGrid` with the same stem as the audio exists somewhere under `2_TextGrid/`; the evaluator searches recursively but still needs matching filenames.
- **Passage text missing**: Double-check that `--passages_csv` points to the oral passages file bundled with the dataset.
- **Passage segmentation not applied**: Make sure the TextGrid files contain the `child` tier and that audio/TextGrid names align; if needed, point `--tier_name` to the tier that carries spoken intervals.
- **Manifests don’t match**: Joins default to the file stem; switch `--match_on` to `name` or `path` (or rename files consistently) if the stems differ.
- **Permissions**: The repo root and `input_output_data` are mounted read-write. Models are mounted read-only from `nemo_inference/models`.

---

## Source files

- **`infer.py`**  
  Automatically discovers `0_Audio/` and `2_TextGrid/` under `--dataset_root`, resamples to 16 kHz,
  slices by TextGrid intervals when present, and writes `transcriptions.jsonl` to the output folder.

- **`evaluation.py`**  
  Orchestrates the evaluation pipeline: loads the `Student_*` CSVs, attaches the ASR manifest, adds
  canonical passage and letter adjustments, searches `2_TextGrid/` recursively for matching `.TextGrid`
  files, computes metrics, and writes the detailed CSV, text summary, and per-pair summaries.

- **`egra_eval/metrics/scoring.py`**  
  Wraps `jiwer` to produce counts (**S, D, I, C, N**), **WER** and **ACC** (all expressed as percentages in downstream outputs). Uses `normalize/textnorm.py` for simple text normalization.

- **`egra_eval/data/textgrid_io.py`**  
  Finds the requested tier case-insensitively (default `child`), gathers labeled intervals, strips filler tags (`<unk>`, `<noise>`, etc.), and concatenates labels to form **REF** per item while searching recursively across annotator folders.

- **`egra_eval/data/linking.py`**  
  Builds join keys from the EGRA CSV (`audio_name`, `audio_stem`) and attaches ASR HYPs by the chosen key (`stem` by default).

- **`egra_eval/data/nemo_manifest.py`**  
  Loads one or many NeMo manifests (JSONL), extracting `audio_path`, `audio_name`, `audio_stem` and `hyp_text`.

- **`egra_eval/data/dataset_layout.py`**  
  Utility helpers that discover dataset packages containing `0_Audio/`, `2_TextGrid/` and the `Student_*` CSVs.

- **`egra_eval/data/passage_merge.py`**  
  Parses the passages CSV (various encodings handled), extracts `passage_num` and fills missing `canonical_text` for `passage_numX` rows.

- **`egra_eval/report/summarize.py`**  
  Builds micro-averaged summaries for each alignment pair:
  - `summary_for_pair(df, prefix, by=None)` — aggregates metrics for one of `can_ref`, `can_hyp`, or `ref_hyp` (optionally grouped by columns).
  - `summary_per_speaker(df, prefix)` — per learner.
  - `summary_per_speaker_macro(df, prefix)` — per learner × macro category.
  - `summary_per_speaker_subcategory(df, prefix)` — per learner × macro category × subcategory.

- **`docker/Dockerfile`**  
  Debian 12 base with PyTorch (CPU or CUDA), NeMo ASR 2.4.1 and all Python dependencies pinned for reproducibility.

- **`docker-compose.yml`**  
  Two services:
  - `nemo-asr`: run inference (`infer.py`).
  - `egra-eval`: run evaluation (`evaluation.py`).
  Mounts repo as `/work`, data as `/io`, models as `/models`, temp segments as `/tmp_segments`.

- **`run_inference.sh` / `run_eval.sh`**  
  Thin wrappers to run the right compose service with the right command (evaluation now requires `--passages_csv`).
- **`run_nemo_offline_eval.sh`**  
  Generates normalized REF/CAN manifests and runs NVIDIA NeMo’s own `speech_to_text_eval.py` script for REF↔HYP and CAN↔HYP scoring. Handy for cross-checking the internal metrics against the official NeMo implementation.
