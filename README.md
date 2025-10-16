# EGRA ASR Evaluation Pipeline


1) **Transcribe** raw EGRA audio with a NeMo ASR model  
2) **Evaluate** the results against human references using EGRA-style metrics and standard ASR metrics.

Everything runs in Docker setup (CPU-only or GPU-enabled).

---

## Contents

- [Project structure](#project-structure)  
- [What the pipeline does](#what-the-pipeline-does)  
- [Input data format](#input-data-format)  
- [How to run (Docker)](#how-to-run-docker)  
  - [1) Build the image](#1-build-the-image)  
  - [2) Run inference (ASR)](#2-run-inference-asr)  
  - [3) Run evaluation](#3-run-evaluation)  
- [Outputs & how to interpret them](#outputs--how-to-interpret-them)  
- [Metrics & definitions](#metrics--definitions)  
- [Configuration knobs](#configuration-knobs)  
- [Troubleshooting](#troubleshooting)  
- [Source files](#source-files)
- [Straight forward steps](#straight-forward-steps)

---

## Project structure

```
.
├── docker/
│   └── Dockerfile                # Base image with PyTorch, NeMo, audio libs, pandas, jiwer, praatio, librosa, etc.
├── docker-compose.yml            # Compose with two services: nemo-asr (inference), egra-eval (evaluation)
├── evaluation.py                 # Main entrypoint for evaluation & summaries
├── infer.py                      # Main entrypoint for NeMo-based transcription
├── run_eval.sh                   # Wrapper script for evaluation
├── run_inference.sh              # Wrapper script for inference
├── egra_eval/
│   ├── data/
│   │   ├── linking.py            # Build keys, attach HYPs to EGRA rows
│   │   ├── nemo_manifest.py      # Load NeMo manifests (JSONL)
│   │   ├── passage_merge.py      # Fill missing canonical passages from CSV
│   │   └── textgrid_io.py        # Read REF text from TextGrid tiers
│   ├── eval/
│   │   └── run_eval.py           # Core scoring module (CAN/REF/HYP)
│   ├── metrics/
│   │   └── scoring.py            # Normalization + WER counts + ACC, P/R/F1
│   ├── normalize/
│   │   └── textnorm.py           # Simple text normalization (lowercase, remove punctuation, collapse spaces)
│   └── report/
│       └── summarize.py          # Summaries: macro, per-learner, overall, etc.
├── input_output_data/
│   ├── input/
│   │   ├── audio_and_texgrid/    # Audio data organized by learner_id (subfolders)
│   │   ├── egradata/             # EGRA CSV + learner META CSV
│   │   ├── nemo_asr_output/      # ASR output manifest (transcriptions.jsonl)
│   │   └── passages/             # Passage mapping CSV (passage number -> canonical text)
│   └── output/                   # Evaluation outputs (detailed + summaries)
└── nemo_inference/
    ├── models/                   # NeMo .nemo models (mounted read-only in container)
    └── tmp/                      # Temporary 16kHz segments dumped during inference (passage slicing). Use "debug" argument for inference (infer.py) in order to keep them for inspection.
```

---

## What the pipeline does

**Inference (`infer.py`)**
- Recursively scans `input_output_data/input/audio_and_texgrid/**` for `.wav` files.
- For each audio:
  - Loads audio, **resamples to 16 kHz** if needed (model was trained at 16 kHz).
  - If a matching **TextGrid** exists (exact stem or any `*passage*.TextGrid` in the same learner folder) and the target tier exists (default `child`), the audio is sliced by the tier’s intervals and each slice is transcribed; the slices are then concatenated into one `pred_text`.
  - If no TextGrid match or if no intervals in TextGrid, transcribes the full file.
- Writes a **NeMo-style JSONL manifest**: one line per audio with `audio_filepath`, `duration` and `pred_text`.

**Evaluation (`evaluation.py`)**
- Loads:
  - **EGRA CSV** (per item rows with at least: `learner_id`, `audio_file`, `audio_type`, `textgrid`).
  - **META CSV** (per-learner attributes, e.g., `gender`, `age`, etc.).
  - **Passage CSV** to fill missing `canonical_text` for `passage_numX` rows.
  - The **ASR manifest** produced above (`transcriptions.jsonl`) to attach `hyp_text`.
- Reads **reference text** (`ref_text`) from each row’s TextGrid (the `textgrid` column points to a filename inside the learner’s subfolder).  
- Computes metrics for:
  - **CAN vs REF** (annotator-based EGRA).
  - **CAN vs HYP** (ASR-based EGRA).
  - **REF vs HYP** (ASR quality vs human).
- Produces a **detailed CSV** and multiple **summaries**.

---

## Input data format

> All inputs live under `input_output_data/input/`. The docker compose mounts this at `/io/input` inside the container.

### 1) Audio & TextGrid layout

```
input_output_data/input/audio_and_texgrid/
├── <learner_id_1>/
│   ├── <audio>.wav
│   ├── <audio>.TextGrid
│   └── ...
└── <learner_id_2>/
    ├── <audio>.wav
    ├── <audio>.TextGrid
    └── ...
```

- Each **learner** has its own folder named by **`learner_id`**.
- **TextGrid tier**: by default the tier named **`child`** (case-insensitive) is used.
- For **passages**, any `*passage*.TextGrid` name is accepted (case-insensitive), or an exact stem match `<audio-stem>.TextGrid`.

### 2) EGRA CSV (per-item rows)

Minimum columns expected:
- `learner_id` — matches subfolder name in `audio_and_texgrid/`.
- `audio_file` — original audio file path or name; used to derive matching keys.
- `audio_type` — e.g., `passage_num1`, `iso_syllable1_1`, etc.
- `textgrid` — the **TextGrid filename** for this item; it is looked up as `<base>/audio_and_texgrid/<learner_id>/<textgrid>`.

Optional:
- `canonical_text` — if present it will be used; if **missing and `audio_type` is `passage_numX`**, we can fill from the **passages CSV** below.

Place this file in `input_output_data/input/egradata/`.

### 3) META CSV (per-learner attributes)

- A CSV with at least `learner_id` and any attributes you want in summaries (e.g., `gender`, `age`).
- Used to enhance the detailed results via a left join on `learner_id`.
- Place in `input_output_data/input/egradata/`.

### 4) Passages CSV (optional)

- Two columns: **Column A** “passage number” (mixed text like `Passage 1`) and **Column B** “passage text”.
- Used to fill missing `canonical_text` when `audio_type` looks like `passage_numX`.
- Place in `input_output_data/input/passages/`.

### 5) NeMo ASR model

- Put your `.nemo` model in `nemo_inference/models/`.
- The default scripts assume `nemo_inference/models/Swahili_exp1_100epochs.nemo` Download link: https://drive.google.com/file/d/1NQTC8532QluX7KXQNGcebKj9FseUzrO-/view?usp=sharing.

---

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
- Scan all `.wav` under `input_output_data/input/audio_and_texgrid/`.
- Write `input_output_data/input/nemo_asr_output/transcriptions.jsonl`.
- Store temp 16kHz segments under `nemo_inference/tmp/`.

Usage:
```bash
./run_inference.sh
```

**What it runs under the hood:**
```bash
docker compose run --rm nemo-asr bash -lc '
  python3 /work/infer.py \
    --model /models/Swahili_exp1_100epochs.nemo \
    --root_audio_dir /io/input/audio_and_texgrid \
    --output_manifest /io/input/nemo_asr_output/transcriptions.jsonl \
    --tmp_dir /work/nemo_inference/tmp \
    --tier_name child \
    --textgrid_keyword passage
'
```

> To use GPU at run time: add `--gpus all` after `docker compose run` or enable `gpus: "all"` in the compose file.

### 3) Run evaluation

We provide `run_eval.sh`. It will:
- Read EGRA CSV, META CSV, Passages CSV (defaults are set for your current tree).
- Attach ASR HYP from `transcriptions.jsonl`.
- Read REF from TextGrids.
- Write detailed and summary CSVs under `input_output_data/output/`.

Usage:
```bash
./run_eval.sh
```

**What it runs under the hood:**
```bash
docker compose run --rm egra-eval bash -lc '
  python3 /work/evaluation.py
'
```

The defaults are configured in `evaluation.py` (see **Configuration knobs** below).

---

## Outputs & how to interpret them

All evaluation outputs land in `input_output_data/output/`:

1. **`egra_eval_detailed.csv`** — One row per EGRA item with:
   - Keys: `learner_id`, `audio_type`, `audio_file`.
   - Texts: `CAN` (canonical), `REF` (annotator), `HYP` (ASR).
   - **CAN vs REF** metrics: `WER_can_ref`, `ACC_can_ref (EGRA_ACC)` plus counts `S_can_ref`, `D_can_ref`, `I_can_ref`, `C_can_ref (EGRA_COR)`, `N_can_ref`.
   - **CAN vs HYP** metrics: `WER_can_hyp`, `ACC_can_hyp (ASR_EGRA_ACC)` plus counts `S_can_hyp`, `D_can_hyp`, `I_can_hyp`, `C_can_hyp (ASR_EGRA_COR)`, `N_can_hyp`.
   - **REF vs HYP** metrics: `WER_asr`, `ASR_precision`, `ASR_recall`, `ASR_f1` plus counts `S_ref_hyp`, `D_ref_hyp`, `I_ref_hyp`, `C_ref_hyp`, `N_ref_hyp`.
   - **Agreement**: `MAE_COR = |EGRA_COR − ASR_EGRA_COR|` which represents the absolute difference in number of correct tokens between annotator-based and ASR-based evaluations.
   - All learner metadata merged in (e.g., `gender`, `age`).

2. **`egra_eval_summary.csv`** — Macro summary **by gender × audio_type** with mean values of core metrics.

3. **`egra_eval_summary_per_learner.csv`** — **Per-learner** mean of all metrics.

4. **`egra_eval_summary_per_learner_audio.csv`** — **Per-learner × audio_type** mean of all metrics.

5. **`egra_eval_summary_overall.csv`** — **Single-row** overall mean across all items.

Use these to compare:
- Human annotator performance (CAN vs REF).
- Automated EGRA performance (CAN vs HYP).
- ASR quality vs human (REF vs HYP).
- Agreement between automated and human EGRA, how closely automated and human scores match (`MAE_COR` closer to 0 is better).

---

## Metrics & definitions

All metrics are computed after **text normalization** (`normalize/textnorm.py`): NFC Unicode, lowercase, punctuation removed, whitespace collapsed.

We compute standard ASR alignment counts via `jiwer`:
- **S** — substitutions  
- **D** — deletions  
- **I** — insertions  
- **C** — correct matches 
- **N** — number of total reference tokens (groundtruth)

From those:

- **WER** = (S + D + I) / N  
- **ACC** = C / N

We apply the same counts to derive **EGRA-style** KPIs:

- **EGRA (Annotator-based)** from **CAN vs REF**  
  - `C_can_ref (EGRA_COR) = number of correct tokens = N_ref − S_can_ref − D_can_ref` 
  - `ACC_can_ref (EGRA_ACC) = EGRA_COR / N_ref`

- **ASR-based EGRA** from **CAN vs HYP**  
  - `C_can_hyp (ASR_EGRA_COR) = N_can − S_can_hyp − D_can_hyp`
  - `ACC_can_hyp (ASR_EGRA_ACC) = ASR_EGRA_COR / N_can`

- **Agreement** between annotator- and ASR-based correctness  
  - `MAE_COR = |EGRA_COR − ASR_EGRA_COR|`

- **ASR quality vs human** from **REF vs HYP**  
  - `WER_asr`, **precision**, **recall**, **F1**  
  - Precision/Recall are computed from counts:  
    - `precision = C / (C + I)`  
    - `recall    = C / (C + D)`  
    - `F1        = 2PR / (P+R)`

---

### Metric ranges & units

Unless otherwise noted, all metrics are expressed as **ratios** between 0.0 and 1.0 (not multiplied by 100).  
If you prefer percentage-style reporting, multiply these values by 100 when displaying or plotting results.

| Metric | Description | Typical Range / Unit | Interpretation |
|:--|:--|:--|:--|
| **WER_can_ref**, **WER_can_hyp**, **WER_asr** | Word Error Rate (substitutions + deletions + insertions) / N | 0.0–1.0 (can exceed 1.0 with many insertions) | Lower is better |
| **ACC_can_ref (EGRA_ACC)**, **ACC_can_hyp (ASR_EGRA_ACC)** | Accuracy = C / N | 0.0–1.0 | Higher is better |
| **ASR_precision**, **ASR_recall**, **ASR_f1** | Precision/Recall/F1 between REF and HYP | 0.0–1.0 | Higher is better |
| **C_can_ref (EGRA_COR)**, **C_can_hyp (ASR_EGRA_COR)** | Correctness count = N − S − D | Integer ≥ 0 | Count of correct tokens |
| **S_\***, **D_\***, **I_\***, **C_\***, **N_\*** | Alignment counts (Substitutions, Deletions, Insertions, Correct, Total) | Integers ≥ 0 | Raw counts |
| **MAE_COR** | Mean Absolute Error between EGRA_COR and ASR_EGRA_COR | Integer ≥ 0 | Lower indicates better agreement |

**Note:**  
If the canonical or reference text has `N = 0`, ratio-based metrics (WER, ACC, precision, recall, F1) are undefined and will appear as `NaN` in the output CSVs.

---


## Configuration knobs

### Inference (`infer.py`)
- **Model path**: `--model /models/your_model.nemo`
- **Audio root**: `--root_audio_dir /io/input/audio_and_texgrid`
- **Output manifest**: `--output_manifest /io/input/nemo_asr_output/transcriptions.jsonl`
- **Temp segments**: `--tmp_dir /work/nemo_inference/tmp`
- **TextGrid tier**: `--tier_name child` (case-insensitive)
- **Passage match**: `--textgrid_keyword passage`

> `run_inference.sh` already passes these. If you need to tweak, edit that script.

### Evaluation (`evaluation.py`)
Defaults are defined near the top:

```python
IO_ROOT = Path(os.getenv("IO_ROOT", "/io"))
DEFAULTS = {
  "egra_csv":      IO_ROOT / "input" / "egradata" / "Student_Dummy_Canonical_EGRA_030925.csv",
  "meta_csv":      IO_ROOT / "input" / "egradata" / "Student_Dummy_MetaData_EGRA_030925.csv",
  "passages_csv":  IO_ROOT / "input" / "passages" / "oral_passages.csv",
  "nemo_manifest": IO_ROOT / "input" / "nemo_asr_output" / "transcriptions.jsonl",
  "textgrids_dir": IO_ROOT / "input" / "audio_and_texgrid",
  "out_csv":       IO_ROOT / "output" / "egra_eval_detailed.csv",
  "summary_csv":   IO_ROOT / "output" / "egra_eval_summary.csv",
}
```

You can run `evaluation.py` without arguments (inside the container) and it will use those paths.  
If you need to override any input/output path, pass the CLI flags (see `--help`).

---

## Troubleshooting

- **No GPU used**: Ensure the image was built with `--build-arg TORCH_CUDA=cu121` **and** you run with `--gpus all` or `gpus: "all"` in compose.
- **Empty or short `pred_text`**: Check that the model matches the language/domain. Also verify sample rate conversion (the script resamples to 16 kHz automatically).
- **Missing REF text**: Ensure the EGRA CSV `textgrid` column points to a valid TextGrid filename inside `audio_and_texgrid/<learner_id>/`.
- **Passage segmentation not applied**: Confirm there’s a `*passage*.TextGrid` (case-insensitive) or an exact `<audio-stem>.TextGrid` next to the WAV; confirm the tier name (default `child`) exists in the TextGrid.
- **Manifests don’t match**: Matching joins rely on file **stem** by default. If your naming is unusual, switch `match_on` to `name` or `path` in `evaluation.py` or rename files consistently.
- **Permissions**: The repo root and `input_output_data` are mounted read-write. Models are mounted read-only from `nemo_inference/models`.

---

## Source files

- **`infer.py`**  
  Scans audio, resamples to 16k, optionally slices by TextGrid intervals, transcribes via NeMo (`ASRModel.restore_from(...)`), writes a NeMo-style JSONL manifest with `pred_text`.

- **`evaluation.py`**  
  Orchestrates the evaluation pipeline. Loads EGRA & META tables, attaches HYP from manifest, reads REF from TextGrids, fills missing canonical passages, computes metrics and writes detailed + summary CSVs.

- **`egra_eval/metrics/scoring.py`**  
  Wraps `jiwer` to produce counts (**S, D, I, C, N**), **WER**, **ACC** and **precision/recall/F1**. Uses `normalize/textnorm.py` for simple text normalization.

- **`egra_eval/data/textgrid_io.py`**  
  Finds the requested tier case-insensitively (default `child`), gathers labeled intervals, concatenates labels to form **REF** per item.

- **`egra_eval/data/linking.py`**  
  Builds join keys from the EGRA CSV (`audio_name`, `audio_stem`) and attaches ASR HYPs by the chosen key (`stem` by default).

- **`egra_eval/data/nemo_manifest.py`**  
  Loads one or many NeMo manifests (JSONL), extracting `audio_path`, `audio_name`, `audio_stem` and `hyp_text`.

- **`egra_eval/data/passage_merge.py`**  
  Parses the passages CSV (various encodings handled), extracts `passage_num` and fills missing `canonical_text` for `passage_numX` rows.

- **`egra_eval/report/summarize.py`**  
  - `macro_summary(df, by=...)` — means of key metrics, grouped by the provided columns.  
  - `per_learner_full(df)` — mean of all metrics per learner.  
  - `overall_full(df)` — overall mean (single row).  
  - `mean_by(df, by=[...])` — generic group mean for all metrics.

- **`docker/Dockerfile`**  
  Debian 12 base with PyTorch (CPU or CUDA), NeMo ASR 2.4.1 and all Python dependencies pinned for reproducibility.

- **`docker-compose.yml`**  
  Two services:
  - `nemo-asr`: run inference (`infer.py`).
  - `egra-eval`: run evaluation (`evaluation.py`).
  Mounts repo as `/work`, data as `/io`, models as `/models`, temp segments as `/tmp_segments`.

- **`run_inference.sh` / `run_eval.sh`**  
  Thin wrappers to run the right compose service with the right command.

---

### Straight forward steps

1. Put your `.nemo` model into `nemo_inference/models/`.  
2. Place your audio + TextGrids into `input_output_data/input/audio_and_texgrid/<learner_id>/`.  
3. Provide EGRA CSV + META CSV under `input_output_data/input/egradata/`.  
4. Provide passages CSV file under `input_output_data/input/passages/`
5. Run:
   ```bash
   ./run_inference.sh
   ./run_eval.sh
   ```
6. Read `input_output_data/output/*.csv` for item-level details and summaries.
