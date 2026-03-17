#!/usr/bin/env python3
"""Command-line entry point for running the simplified EGRA evaluation pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from datetime import datetime

# Import the local dp_align module provided in the context
try:
    import dp_align
except ImportError:
    raise ImportError("Could not import 'dp_align'. Ensure dp_align.py is in the script directory.")

from egra_eval.data.dataset_layout import DatasetLayoutError, resolve_dataset_paths
from egra_eval.data.linking import add_audio_keys, attach_hypotheses
from egra_eval.data.nemo_manifest import load_many_manifests
from egra_eval.data.passage_merge import attach_passage_texts
from egra_eval.data.textgrid_io import add_refs_from_textgrid
from egra_eval.eval.run_eval import evaluate
from egra_eval.metrics.phonological import compute_phonological_metrics_row
from egra_eval.pipeline.manifest_builder import (
    build_reference_manifest_dataframe,
    write_manifest_jsonl,
)
from egra_eval.pipeline.manifest_cleaner import clean_manifest_jsonl
from egra_eval.report.summarize import (
    aggregate_phonological_metrics,
    summary_for_pair,
    summary_per_speaker,
    summary_per_speaker_macro,
    summary_per_speaker_subcategory,
    summary_phonological_by_category,
    _annotate_audio_categories,
)

CONSONANTS = set("bcdfghjklmnpqrstvwxyz")


def _append_a_to_consonant_letters(text: str) -> str:
    if text is None or pd.isna(text):
        return text
    parts = re.split(r"(\s+)", str(text))

    def transform_token(token: str) -> str:
        cleaned = re.sub(r"[^a-z]", "", token.lower())
        if not cleaned:
            return token
        first_char = cleaned[0]
        if first_char in CONSONANTS:
            if token.lower().endswith("a"):
                return token
            return f"{token}a"
        return token

    return "".join(
        transform_token(part) if idx % 2 == 0 else part
        for idx, part in enumerate(parts)
    )


def adjust_letter_canonical_text(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    if "audio_type" not in df.columns or "canonical_text" not in df.columns:
        return df
    mask = df["audio_type"].astype(str).str.contains("letter", case=False, na=False)
    if not mask.any():
        return df
    logger.info("Adjusting canonical texts for %d letter row(s).", int(mask.sum()))
    out = df.copy()
    out.loc[mask, "canonical_text"] = out.loc[mask, "canonical_text"].apply(_append_a_to_consonant_letters)
    return out

# ---------------------------------------------------------------------------
# Advanced Metrics & Alignment Logic (dp_align Integration)
# ---------------------------------------------------------------------------

def get_csid_sequence(canonical_text: str, other_text: str) -> List[str]:
    """
    Uses dp_align to generate the sequence of 'c', 's', 'i', 'd' tags.
    
    Args:
        canonical_text: The "Ground Truth" / Reference list for the alignment.
        other_text: The "Test" / Hypothesis list for the alignment.
        
    Returns:
        A list of strings ['c', 's', 'i', 'c', 'd', ...] representing the alignment path.
    """
    # Handle NaNs or non-strings
    if pd.isna(canonical_text): canonical_text = ""
    if pd.isna(other_text): other_text = ""

    # Normalize transcripts (lowercase, remove punctuation, collapse whitespace)
    def normalize_transcript(text: str) -> str:
        if text is None:
            return ""
        s = str(text).lower()
        # Replace punctuation with space
        s = re.sub(r"[{}]".format(re.escape(string.punctuation)), " ", s)
        # Collapse multiple whitespace and strip
        s = re.sub(r"\s+", " ", s).strip()
        return s

    can_norm = normalize_transcript(canonical_text)
    other_norm = normalize_transcript(other_text)

    # dp_align works on lists of tokens (words), so we split the normalized text.
    can_list = can_norm.split()
    other_list = other_norm.split()
    
    # dp_align(ref_list, test_list) -> We treat Canonical as Ref, Other as Test.
    # Returns (errors, alignment). Alignment is list of tuples: (test_token, ref_token, tag)
    _, alignment = dp_align.dp_align(can_list, other_list, output_align=True)
    
    # Extract the tag (3rd element) from the alignment tuple
    # alignment structure: [(test_token, ref_token, 'c'), ...]
    return [item[2] for item in alignment]


def compute_fine_grained_metrics(row) -> pd.Series:
    """
    Computes MER, P/R/F1 for substitutions, insertions, deletions, and all mistakes.
    Uses Alignment of Alignments (Meta-Alignment).
    """
    # Handle both naming conventions: lowercase with _text suffix or uppercase
    can = row.get('canonical_text', row.get('CAN', ''))
    ref = row.get('reference_text', row.get('REF', ''))
    hyp = row.get('hypothesis_text', row.get('HYP', ''))

    # 1. Generate Sequence A (Reference vs Canonical) and B (Hypothesis vs Canonical)
    # These are lists of 'c', 's', 'i', 'd'
    seq_a = get_csid_sequence(can, ref)
    seq_b = get_csid_sequence(can, hyp)
    
    # 2. Meta-Alignment: Align Sequence B (Pred) to Sequence A (Truth)
    # We treat seq_a as the "Ref List" and seq_b as the "Test List" in dp_align terms
    mer_errors, mer_alignment = dp_align.dp_align(seq_a, seq_b, output_align=True)
    
    # 3. Calculate Mistake Error Rate (MER)
    # MER is essentially the WER of the two mistake sequences
    mer_val = mer_errors.get_wer()

    # 4. Prepare data for Scikit-Learn P/R/F1
    # mer_alignment structure: [(seq_b_token, seq_a_token, align_type), ...]
    # y_pred comes from seq_b (Hypothesis mistake types)
    # y_true comes from seq_a (Reference mistake types)
    
    y_pred = [x[0] for x in mer_alignment]
    y_true = [x[1] for x in mer_alignment]
    
    # Note: dp_align uses '-' for gaps. 
    # e.g. If Reference had 's' and Hyp had nothing (deletion of mistake), y_true='s', y_pred='-'
    
    metrics_res = {}
    labels = ['s', 'i', 'd']
    
    # Calculate P/R/F1 for specific mistake types
    # We use `labels` to force sklearn to only look at s, i, d.
    # '-' or 'c' are treated as "not the class of interest"
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    
    for i, label in enumerate(labels):
        metrics_res[f'{label.upper()}_Precision'] = p[i]
        metrics_res[f'{label.upper()}_Recall'] = r[i]
        metrics_res[f'{label.upper()}_F1'] = f1[i]

    # 5. "All Mistakes" (Binary: Mistake vs Correct)
    # Convert s, i, d to 'x'. Convert 'c' and '-' to 'c' (or ignore).
    # We want to know: If it WAS a mistake (x), did we predict a mistake (x)?
    
    def binarize(token):
        if token in ['s', 'i', 'd']:
            return 'x'
        return 'c' # Treat 'c' and '-' as correct/neutral

    y_true_bin = [binarize(t) for t in y_true]
    y_pred_bin = [binarize(t) for t in y_pred]
    
    p_x, r_x, f1_x, _ = precision_recall_fscore_support(y_true_bin, y_pred_bin, labels=['x'], zero_division=0)
    
    metrics_res['Mistakes_Precision'] = p_x[0]
    metrics_res['Mistakes_Recall'] = r_x[0]
    metrics_res['Mistakes_F1'] = f1_x[0]
    metrics_res['MER'] = mer_val

    return pd.Series(metrics_res)


def calculate_advanced_metrics(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Enriches the dataframe with:
    1. MAE_EGRA_ACC: |ACC_can_ref - ACC_can_hyp| (Absolute Percentage Difference)
    2. Bias Baseline: MAE between actual ACC and the dataset average ACC.
    3. MER and Fine-grained P/R/F1 metrics using dp_align.
    """
    logger.info("Calculating advanced metrics (MAE, Bias, MER, P/R/F1) using dp_align...")
    
    # 1. MAE_EGRA_ACC
    # Ensure both ACC columns are on the same scale (0..1). Some codepaths return percentages (0..100).
    if "ACC_can_ref" in df.columns and "ACC_can_hyp" in df.columns:
        # Detect percent-scale (values > 1.0) and convert to 0..1
        ref_max = df["ACC_can_ref"].dropna().max() if not df["ACC_can_ref"].dropna().empty else 0
        hyp_max = df["ACC_can_hyp"].dropna().max() if not df["ACC_can_hyp"].dropna().empty else 0
        if ref_max > 1.0 or hyp_max > 1.0:
            logger.info("Detected ACC in 0-100 scale; converting ACC_can_ref/ACC_can_hyp to 0..1 scale for MAE calculation.")
            df["ACC_can_ref_norm_for_mae"] = df["ACC_can_ref"] / 100.0
            df["ACC_can_hyp_norm_for_mae"] = df["ACC_can_hyp"] / 100.0
        else:
            df["ACC_can_ref_norm_for_mae"] = df["ACC_can_ref"]
            df["ACC_can_hyp_norm_for_mae"] = df["ACC_can_hyp"]

        # Compute per-row MAE in the normalized (0..1) space
        df["MAE_EGRA_ACC"] = (df["ACC_can_ref_norm_for_mae"] - df["ACC_can_hyp_norm_for_mae"]).abs()
        # Optionally keep MAE in percent points for reporting by multiplying to 100 later in summary writer.
    else:
        df["MAE_EGRA_ACC"] = np.nan

    # Also compute MAE on correct counts (|EGRA_COR - ASR_EGRA_COR|) when counts are available
    if "C_can_ref" in df.columns and "C_can_hyp" in df.columns:
        df["MAE_EGRA_COR"] = (df["C_can_ref"] - df["C_can_hyp"]).abs()
    else:
        df["MAE_EGRA_COR"] = np.nan

    # 2. Bias Baseline
    # Predict the average ACC_can_ref for everyone
    if "ACC_can_ref" in df.columns:
        mean_acc = df["ACC_can_ref"].mean()
        # MAE between actual and mean
        df["Bias_Baseline_MAE"] = (df["ACC_can_ref"] - mean_acc).abs()
        logger.info(f"Bias Baseline (Mean Accuracy across dataset): {mean_acc:.4f}")
    else:
        df["Bias_Baseline_MAE"] = np.nan

    # 3. Fine-grained Metrics (MER, P, R, F1)
    # Check for either naming convention (CAN/REF/HYP or canonical_text/reference_text/hypothesis_text)
    has_lowercase = all(c in df.columns for c in ["canonical_text", "reference_text", "hypothesis_text"])
    has_uppercase = all(c in df.columns for c in ["CAN", "REF", "HYP"])
    if has_lowercase or has_uppercase:
        # Filter to ensure we don't crash on empty rows, though get_csid handles it
        logger.info("Running alignment-of-alignments (MER/Fine-grained)...")
        fine_grained = df.apply(compute_fine_grained_metrics, axis=1)
        df = pd.concat([df, fine_grained], axis=1)
    
    return df

# ---------------------------------------------------------------------------
# Logging / CLI
# ---------------------------------------------------------------------------

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("egra_eval")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
    return logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run EGRA evaluation using canonical/ASR data.")
    p.add_argument("--dataset_root", required=True, help="Root folder containing 0_Audio/2_TextGrid and Student_* CSVs.")
    p.add_argument("--output_root", default=None, help="Directory where outputs will be written (detailed CSV + summary subfolders). If omitted, defaults to input_output_data/output/experiments/exp_{YYYY_MM_DD_hh_mm_ss}.")

    p.add_argument("--egra_csv", default=None)
    p.add_argument("--meta_csv", default=None)
    p.add_argument(
        "--passages_csv",
        required=True,
        help="CSV mapping passage numbers to canonical text (e.g. oral_passages.csv).",
    )
    p.add_argument("--nemo_manifest", action="append", default=None, help="Path(s) to NeMo JSON manifests with ASR hypotheses (can be supplied multiple times).")
    p.add_argument("--manifest_audio_key", default="audio_filepath")
    p.add_argument("--manifest_hyp_key", default="pred_text")
    p.add_argument("--manifest_can_key", default=None)
    p.add_argument("--match_on", choices=["stem", "name", "path"], default="stem", help="How to join manifest HYPs to EGRA rows.")

    p.add_argument("--out_csv", default=None)
    p.add_argument("--summary_can_ref_dir", default=None)
    p.add_argument("--summary_can_hyp_dir", default=None)
    p.add_argument("--summary_ref_hyp_dir", default=None)

    # Optional manifest-oriented workflow (PR2 style, opt-in).
    p.add_argument(
        "--build_manifest_first",
        action="store_true",
        help="Build and clean a reference manifest before scoring, then load it for REF/CAN texts.",
    )
    p.add_argument(
        "--manifest_only",
        action="store_true",
        help="Build/clean manifest and exit without running evaluation.",
    )
    p.add_argument(
        "--manifest_in",
        default=None,
        help="Path to an existing cleaned manifest JSONL to load for REF/CAN attachment.",
    )
    p.add_argument(
        "--manifest_raw_out",
        default=None,
        help="Path for the raw built reference manifest JSONL. Defaults to <output_root>/manifests/ref_manifest.raw.jsonl.",
    )
    p.add_argument(
        "--manifest_clean_out",
        default=None,
        help="Path for the cleaned manifest JSONL. Defaults to <output_root>/manifests/ref_manifest.clean.jsonl.",
    )
    p.add_argument(
        "--manifest_clean_drop_empty",
        action="store_true",
        help="Drop rows with empty cleaned text from the generated manifest.",
    )
    p.add_argument(
        "--manifest_path_prefix",
        default=None,
        help="Optional path prefix to rewrite audio paths in built manifest (e.g., /io/input/<dataset>).",
    )
    p.add_argument(
        "--manifest_text_key",
        default="ref_text",
        help="Text key to load from cleaned manifest into REF (default: ref_text).",
    )
    return p.parse_args()

# ---------------------------------------------------------------------------
# IO resolution / outputs
# ---------------------------------------------------------------------------

def resolve_inputs(args: argparse.Namespace, logger: logging.Logger):
    """Resolve dataset layout and output dirs; mutate args with defaults."""
    try:
        layout = resolve_dataset_paths(args.dataset_root)
    except DatasetLayoutError as exc:
        raise SystemExit(str(exc)) from exc

    # If output_root not provided, default under repository input_output_data/output/experiments
    if not args.output_root:
        default_dir = Path("input_output_data") / "output" / "experiments" / f"exp_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        logger.info("No --output_root supplied; using default: %s", default_dir)
        args.output_root = str(default_dir)

    # Fill inferred CSVs
    args.egra_csv = args.egra_csv or str(layout.canonical_csv)
    args.meta_csv = args.meta_csv or str(layout.metadata_csv)
    passages_path = Path(args.passages_csv)
    if not passages_path.exists():
        raise SystemExit(f"--passages_csv file not found: {passages_path}")

    # Resolve manifest(s)
    if not args.nemo_manifest:
        candidate = layout.root / "nemo_asr_output" / "transcriptions.jsonl"
        if candidate.exists():
            args.nemo_manifest = [str(candidate)]
        else:
            raise SystemExit(
                "No --nemo_manifest supplied and no default manifest found at "
                f"{candidate}. Run inference first or provide the path explicitly."
            )

    logger.info(
        "Dataset root resolved: audio=%s | textgrids=%s | canonical=%s | metadata=%s",
        layout.audio_root, layout.textgrid_root, layout.canonical_csv, layout.metadata_csv,
    )

    # Prepare outputs
    base = Path(args.output_root)
    base.mkdir(parents=True, exist_ok=True)

    args.out_csv = args.out_csv or str(base / "egra_eval_detailed.csv")
    args.summary_can_ref_dir = args.summary_can_ref_dir or str(base / "can_ref")
    args.summary_can_hyp_dir = args.summary_can_hyp_dir or str(base / "can_hyp")
    args.summary_ref_hyp_dir = args.summary_ref_hyp_dir or str(base / "ref_hyp")

    summary_dirs: Dict[str, Path] = {
        "can_ref": Path(args.summary_can_ref_dir),
        "can_hyp": Path(args.summary_can_hyp_dir),
        "ref_hyp": Path(args.summary_ref_hyp_dir),
    }
    return layout, summary_dirs


def ensure_output_dirs(out_csv: str, summary_dirs: Dict[str, Path]) -> None:
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    for d in summary_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data loading / writing
# ---------------------------------------------------------------------------

def load_manifest(args: argparse.Namespace, logger: logging.Logger) -> pd.DataFrame:
    if not args.nemo_manifest:
        return pd.DataFrame()
    logger.info("Loading %d NeMo manifest(s)...", len(args.nemo_manifest))
    df = load_many_manifests(
        args.nemo_manifest,
        audio_key=args.manifest_audio_key,
        hyp_key=args.manifest_hyp_key,
        can_key=args.manifest_can_key,
    )
    logger.info("Loaded manifest rows: %d", len(df))
    return df


def load_text_manifest(
    path: str,
    *,
    audio_key: str = "audio_filepath",
    text_key: str = "ref_text",
    can_key: str = "can_text",
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    logger = logger or logging.getLogger("egra_eval")
    rows: List[Dict[str, Any]] = []
    bad = 0
    total = 0

    in_path = Path(path)
    if not in_path.exists():
        logger.warning("Text manifest does not exist: %s", in_path)
        return pd.DataFrame()

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            total += 1
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                bad += 1
                continue
            if not isinstance(obj, dict):
                bad += 1
                continue
            audio = str(obj.get(audio_key, "") or "").strip()
            if not audio:
                bad += 1
                continue
            p = Path(audio)
            rows.append(
                {
                    "audio_path": audio,
                    "audio_name": p.name,
                    "audio_stem": p.stem,
                    "manifest_text": str(obj.get(text_key, "") or ""),
                    "manifest_can_text": str(obj.get(can_key, "") or ""),
                }
            )

    out = pd.DataFrame(rows)
    logger.info(
        "Loaded text manifest %s | rows=%d | total_lines=%d | skipped=%d",
        in_path,
        len(out),
        total,
        bad,
    )
    return out


def attach_texts_from_manifest(
    df_egra: pd.DataFrame,
    text_manifest_df: pd.DataFrame,
    *,
    match_on: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    if text_manifest_df is None or text_manifest_df.empty:
        logger.warning("Text manifest is empty; keeping existing canonical/ref columns.")
        return df_egra

    left = df_egra.copy()
    right = text_manifest_df.copy()

    if match_on == "stem":
        key = "audio_stem"
    elif match_on == "name":
        key = "audio_name"
    elif match_on == "path":
        key = "audio_path_csv"
        right = right.rename(columns={"audio_path": "audio_path_csv"})
    else:
        raise ValueError("match_on must be one of: 'stem' | 'name' | 'path'")

    merged = left.merge(
        right[[key, "manifest_text", "manifest_can_text"]].drop_duplicates(subset=[key], keep="last"),
        on=key,
        how="left",
    )

    if "ref_text" not in merged.columns:
        merged["ref_text"] = ""
    if "canonical_text" not in merged.columns:
        merged["canonical_text"] = ""

    merged["ref_text"] = merged["manifest_text"].where(
        merged["manifest_text"].astype(str).str.strip() != "",
        merged["ref_text"],
    )
    merged["canonical_text"] = merged["manifest_can_text"].where(
        merged["manifest_can_text"].astype(str).str.strip() != "",
        merged["canonical_text"],
    )

    attached_ref = (merged["manifest_text"].astype(str).str.strip() != "").sum()
    attached_can = (merged["manifest_can_text"].astype(str).str.strip() != "").sum()
    logger.info(
        "Attached texts from cleaned manifest (key=%s): ref_rows=%d, can_rows=%d",
        key,
        int(attached_ref),
        int(attached_can),
    )

    merged.drop(columns=["manifest_text", "manifest_can_text"], inplace=True)
    return merged


def write_detailed_csv(df: pd.DataFrame, path: str, logger: logging.Logger) -> None:
    rename_map = {
        "learner_id": "learner_id",
        "audio_type": "audio_type",
        "audio_file": "audio_file",
        "CAN": "CAN",
        "REF": "REF",
        "HYP": "HYP",
        "WER_can_ref": "WER_can_ref",
        "ACC_can_ref": "ACC_can_ref (EGRA_ACC)",
        "S_can_ref": "S_can_ref",
        "D_can_ref": "D_can_ref",
        "I_can_ref": "I_can_ref",
        "C_can_ref": "C_can_ref (EGRA_COR)",
        "N_can_ref": "N_can_ref",
        "WER_can_hyp": "WER_can_hyp",
        "ACC_can_hyp": "ACC_can_hyp (ASR_EGRA_ACC)",
        "S_can_hyp": "S_can_hyp",
        "D_can_hyp": "D_can_hyp",
        "I_can_hyp": "I_can_hyp",
        "C_can_hyp": "C_can_hyp (ASR_EGRA_COR)",
        "N_can_hyp": "N_can_hyp",
        "WER_ref_hyp": "WER_ref_hyp",
        "ACC_ref_hyp": "ACC_ref_hyp",
        "S_ref_hyp": "S_ref_hyp",
        "D_ref_hyp": "D_ref_hyp",
        "I_ref_hyp": "I_ref_hyp",
        "C_ref_hyp": "C_ref_hyp",
        "N_ref_hyp": "N_ref_hyp",
        "EGRA_COR": "EGRA-COR",
        "EGRA_ACC": "EGRA-ACC",
        "ASR_EGRA_COR": "ASR-EGRA-COR",
        "ASR_EGRA_ACC": "ASR-EGRA-ACC",
        "MAE_EGRA_COR": "MAE_EGRA_COR",
        "ASR_WER": "ASR_WER",
        "has_hyp": "has_hyp",
        "gender": "gender",
        "child_grade": "child_grade",
        "child_age": "child_age",
        "region": "region",
        "council": "council",
        "ward": "ward",
        "Kiswahili_lang1": "Kiswahili_lang1",
        "Kigogo_lang2": "Kigogo_lang2",
        "Kirangi_lang3": "Kirangi_lang3",
        "Kihaya_lang4": "Kihaya_lang4",
        "Runyambo_lang5": "Runyambo_lang5",
        "Kihangaza_lang6": "Kihangaza_lang6",
        "English_lang7": "English_lang7",
        "ACC_can_ref_norm_for_mae": "ACC_can_ref_norm_for_mae",
        "ACC_can_hyp_norm_for_mae": "ACC_can_hyp_norm_for_mae",
        "MAE_EGRA_ACC": "MAE_EGRA_ACC",
        "Bias_Baseline_MAE": "Bias_Baseline_MAE",
        "S_Precision": "S_Precision",
        "S_Recall": "S_Recall",
        "S_F1": "S_F1",
        "I_Precision": "I_Precision",
        "I_Recall": "I_Recall",
        "I_F1": "I_F1",
        "D_Precision": "D_Precision",
        "D_Recall": "D_Recall",
        "D_F1": "D_F1",
        "Mistakes_Precision": "Mistakes_Precision",
        "Mistakes_Recall": "Mistakes_Recall",
        "Mistakes_F1": "Mistakes_F1",
        "Mistake_Error_Rate (MER)": "Mistake_Error_Rate (MER)",
        "S_TP": "S_TP",
        "S_FP": "S_FP",
        "S_FN": "S_FN",
        "S_Precision": "S_Precision",
        "S_Recall": "S_Recall",
        "S_F1": "S_F1",
        "D_TP": "D_TP",
        "D_FP": "D_FP",
        "D_FN": "D_FN",
        "D_Precision": "D_Precision",
        "D_Recall": "D_Recall",
        "D_F1": "D_F1",
        "I_TP": "I_TP",
        "I_FP": "I_FP",
        "I_FN": "I_FN",
        "I_Precision": "I_Precision",
        "I_Recall": "I_Recall",
        "I_F1": "I_F1",
    }
    # Defensive deduplication: some upstream joins can produce identical duplicate
    # rows (observed for `full_syllable1_grid` entries). Prefer dropping duplicates
    # based on the identifying keys if available, otherwise drop exact duplicates.
    initial_rows = len(df)
    # Prefer deduplication that also checks the canonical/reference/hypothesis
    # text columns when they exist so we don't drop rows that only differ
    # in those fields. This helps verify whether rows are truly identical.
    base_keys = ["learner_id", "audio_type", "audio_file"]
    text_keys = ["CAN", "REF", "HYP", "canonical_text", "reference_text", "hypothesis_text"]
    subset_keys = [c for c in base_keys + text_keys if c in df.columns]

    if subset_keys:
        df = df.drop_duplicates(subset=subset_keys, keep="first")
        removed = initial_rows - len(df)
        if removed > 0:
            logger.info("Removed %d duplicate row(s) based on keys %s before writing detailed CSV.", removed, subset_keys)
    else:
        df = df.drop_duplicates(keep="first")
        removed = initial_rows - len(df)
        if removed > 0:
            logger.info("Removed %d exact duplicate row(s) before writing detailed CSV.", removed)

    df.rename(columns=rename_map).to_csv(path, index=False)
    logger.info("Wrote detailed results -> %s (rows: %d)", path, len(df))


def write_summary_csvs(df: pd.DataFrame, summary_dirs: Dict[str, Path], logger: logging.Logger) -> List[Path]:
    written: List[Path] = []

    def _write_one(pair: str, directory: Path) -> None:
        per_speaker = summary_per_speaker(df, pair)
        global_row = summary_for_pair(df, pair, by=None)
        global_row.insert(0, "learner_id", "__GLOBAL__")

        combined = (
            global_row
            if per_speaker.empty
            else pd.concat(
                [global_row[per_speaker.columns], per_speaker],
                ignore_index=True,
            )
        )
        per_speaker_path = directory / "egra_eval_summary_per_speaker_global.csv"
        combined.to_csv(per_speaker_path, index=False)
        logger.info("[%s] Wrote per-speaker summary -> %s", pair, per_speaker_path)
        written.append(per_speaker_path)

        macro_path = directory / "egra_eval_summary_per_speaker_macro.csv"
        summary_per_speaker_macro(df, pair).to_csv(macro_path, index=False)
        logger.info("[%s] Wrote per-speaker × macro-category summary -> %s", pair, macro_path)
        written.append(macro_path)

        subcat_path = directory / "egra_eval_summary_per_speaker_subcat.csv"
        summary_per_speaker_subcategory(df, pair).to_csv(subcat_path, index=False)
        logger.info("[%s] Wrote per-speaker × subcategory summary -> %s", pair, subcat_path)
        written.append(subcat_path)

    for pair, directory in summary_dirs.items():
        _write_one(pair, directory)

    return written


def write_text_summary(df: pd.DataFrame, out_csv: str, logger: logging.Logger) -> Path:
    summary_path = Path(out_csv).parent / "egra_eval_summary.txt"

    # --- 1. Gather Global Metrics ---
    metrics = {}
    
    # Global EGRA counts
    can_ref = summary_for_pair(df, "can_ref")
    if not can_ref.empty:
        metrics["EGRA-COR"] = can_ref["C_can_ref"].iloc[0]
        metrics["EGRA-ACC"] = can_ref["ACC_can_ref"].iloc[0]

    # Global ASR counts
    can_hyp = summary_for_pair(df, "can_hyp")
    if not can_hyp.empty:
        metrics["ASR-EGRA-COR"] = can_hyp["C_can_hyp"].iloc[0]
        metrics["ASR-EGRA-ACC"] = can_hyp["ACC_can_hyp"].iloc[0]

    # Global WER
    ref_hyp = summary_for_pair(df, "ref_hyp")
    if not ref_hyp.empty:
        metrics["ASR_WER"] = ref_hyp["WER_ref_hyp"].iloc[0]

    # Global r_correlation (EGRA-COR vs ASR-EGRA-COR, all rows)
    r_correlation_global = "Not Available"
    scatter_path_global = None
    if "C_can_ref" in df.columns and "C_can_hyp" in df.columns:
        common = df[["C_can_ref", "C_can_hyp"]].dropna()
        if len(common) >= 2:
            arr_x = common["C_can_ref"].to_numpy()
            arr_y = common["C_can_hyp"].to_numpy()
            if arr_x.std() > 0 and arr_y.std() > 0:
                r_correlation_global = float(np.corrcoef(arr_x, arr_y)[0, 1])
                # Save scatter plot
                try:
                    scatter_dir = summary_path.parent / "scatter_plots"
                    scatter_dir.mkdir(parents=True, exist_ok=True)
                    plt.figure(figsize=(6, 4))
                    plt.scatter(arr_x, arr_y, alpha=0.6, s=30)
                    mn = min(arr_x.min(), arr_y.min())
                    mx = max(arr_x.max(), arr_y.max())
                    plt.plot([mn, mx], [mn, mx], color="gray", linestyle="--", linewidth=0.8)
                    plt.xlabel("EGRA-COR (C_can_ref)")
                    plt.ylabel("ASR-EGRA-COR (C_can_hyp)")
                    plt.title(f"Global: EGRA-COR vs ASR-EGRA-COR (r={r_correlation_global:.3f})")
                    fname = f"scatter_GLOBAL_C_can_ref_vs_C_can_hyp_r{r_correlation_global:.3f}.png"
                    safe_fname = fname.replace(" ", "_")
                    scatter_path_global = scatter_dir / safe_fname
                    plt.tight_layout()
                    plt.savefig(scatter_path_global)
                    plt.close()
                    logger.info("Wrote global scatter plot -> %s", scatter_path_global)
                except Exception as exc:
                    logger.warning("Could not write global scatter plot: %s", exc)
    metrics["r_correlation"] = r_correlation_global

    # Averages for row-based metrics
    if "MAE_EGRA_ACC" in df.columns:
        metrics["MAE_EGRA_ACC"] = df["MAE_EGRA_ACC"].dropna().mean() * 100.0
    if "Bias_Baseline_MAE" in df.columns:
        metrics["Bias_Baseline_MAE"] = df["Bias_Baseline_MAE"].dropna().mean()
    if "MER" in df.columns:
        metrics["MER"] = df["MER"].dropna().mean()

    # --- 2. Gather Category Data ---
    # This returns a dict: {'passage_passage': {'S_TP': ...}, 'letters_grid': {...}}
    phono_by_cat = summary_phonological_by_category(df)

    # --- 3. Define Task Mapping & Groups ---
    task_map = {
        'T1': 'passage_passage',
        'T2': 'syllables_grid',
        'T3': 'syllables_isolated', 
        'T4': 'nonwords_grid',
        'T5': 'nonwords_isolated',
        'T6': 'letters_grid',
        'T7': 'letters_isolated'
    }
    
    group_A_tasks = ['T1', 'T2', 'T4', 'T6'] # Passage & Grid
    group_B_tasks = ['T3', 'T5', 'T7']       # Isolated

    # Helper to calculate P/R/F1 from TP/FP/FN
    def calc_prf1(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1

    # Prepare a few additional overall metrics to write at the very top
    # MAE on correct counts (mean of per-row MAE_EGRA_COR) if available
    if "MAE_EGRA_COR" in df.columns:
        metrics["MAE_EGRA_COR"] = df["MAE_EGRA_COR"].dropna().mean()
    else:
        metrics["MAE_EGRA_COR"] = float("nan")

    # Mistakes F1 (mean across rows) if available
    if "Mistakes_F1" in df.columns:
        metrics["Mistakes_F1"] = df["Mistakes_F1"].dropna().mean()
    else:
        metrics["Mistakes_F1"] = float("nan")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        # Helper to format numeric values to 4 decimal places for per-task output
        def _fmt4(val) -> str:
            if val is None:
                return "NaN"
            if isinstance(val, str) and val in ("N/A", "Not Available"):
                return val
            try:
                if pd.isna(val):
                    return "NaN"
            except Exception:
                pass
            try:
                return f"{float(val):.4f}"
            except Exception:
                return str(val)

        # Write a compact overall metrics table first
        f.write("Metric\tValue\n")
        # Desired ordering for the top table
        top_keys = [
            "EGRA-COR",
            "EGRA-ACC",
            "ASR-EGRA-COR",
            "ASR-EGRA-ACC",
            "MAE_EGRA_COR",
            "MAE_EGRA_ACC",
            "Bias_Baseline_MAE",
            "MER",
            "Mistakes_F1",
            "ASR_WER",
            "r_correlation",
        ]
        for k in top_keys:
            v = metrics.get(k, float("nan"))
            if pd.isna(v):
                out_val = "NaN"
            elif isinstance(v, (int, np.integer)):
                out_val = f"{int(v)}"
            elif isinstance(v, float) or isinstance(v, np.floating):
                out_val = f"{v:.2f}"
            else:
                out_val = str(v)
            f.write(f"{k}\t{out_val}\n")
        f.write("\n")

        f.write("=== MAPPED METRICS ===\n\n")

        for task_id, cat_name in task_map.items():
            f.write(f"--- Task {task_id} ({cat_name}) ---\n")
            
            # Get specific category data (default to empty dict if missing)
            cat_data = phono_by_cat.get(cat_name, {})
            # Annotate dataframe with macro/sub categories and extract rows for this category
            df_cat = _annotate_audio_categories(df)
            try:
                macro, sub = cat_name.split("_", 1)
            except Exception:
                macro, sub = (cat_name, None)
            sel = df_cat[
                (df_cat["macro_category"] == macro) & (df_cat["sub_category"] == sub)
            ] if macro is not None and sub is not None else df_cat[df_cat["macro_category"] == macro]

            # Per-category MER (mean of per-row MER column) if available, otherwise fall back to global
            if not sel.empty and "MER" in sel.columns:
                cat_mer = float(sel["MER"].dropna().mean())
            else:
                cat_mer = metrics.get("MER", float("nan"))
            
            # Calculate aggregate "Mistakes" (S+D+I) for this category
            m_tp = sum(cat_data.get(f'{k}_TP', 0) for k in ['S', 'D', 'I'])
            m_fp = sum(cat_data.get(f'{k}_FP', 0) for k in ['S', 'D', 'I'])
            m_fn = sum(cat_data.get(f'{k}_FN', 0) for k in ['S', 'D', 'I'])
            m_p, m_r, m_f1 = calc_prf1(m_tp, m_fp, m_fn)

            # --- Formatting for Group A (Passage & Grid) ---
            if task_id in group_A_tasks:
                # Compute per-task mean WER_ref_hyp for this category
                if not sel.empty and "WER_ref_hyp" in sel.columns:
                    wer_ref_hyp_task = sel["WER_ref_hyp"].dropna().mean()
                else:
                    wer_ref_hyp_task = metrics.get('ASR_WER', 'N/A')
                f.write(f"wer_ref_hyp: {_fmt4(wer_ref_hyp_task)}\n")

                # Compute Pearson r between per-utterance predicted correct (C_ref_hyp) and actual correct (C_can_ref)
                r_val = "Not Available"
                if not sel.empty and "C_ref_hyp" in sel.columns and "C_can_ref" in sel.columns:
                    x = sel["C_ref_hyp"].dropna().to_numpy()
                    y = sel["C_can_ref"].dropna().to_numpy()
                    # align lengths by index of sel where both present
                    common = sel[["C_ref_hyp", "C_can_ref"]].dropna()
                    if len(common) >= 2:
                        arr_x = common["C_ref_hyp"].to_numpy()
                        arr_y = common["C_can_ref"].to_numpy()
                        # require variance
                        if arr_x.std() > 0 and arr_y.std() > 0:
                            r_val = float(np.corrcoef(arr_x, arr_y)[0, 1])
                        else:
                            r_val = "Not Available"
                f.write(f"r_correlation: {_fmt4(r_val)}\n")
                # Save scatter plot for this category (if we have numeric r and data)
                try:
                    scatter_dir = summary_path.parent / "scatter_plots"
                    scatter_dir.mkdir(parents=True, exist_ok=True)
                    if isinstance(r_val, float):
                        common = sel[["C_ref_hyp", "C_can_ref"]].dropna()
                        if not common.empty:
                            xs = common["C_can_ref"].to_numpy()
                            ys = common["C_ref_hyp"].to_numpy()
                            plt.figure(figsize=(6, 4))
                            plt.scatter(xs, ys, alpha=0.6, s=20)
                            # identity line for reference
                            mn = min(xs.min(), ys.min())
                            mx = max(xs.max(), ys.max())
                            plt.plot([mn, mx], [mn, mx], color="gray", linestyle="--", linewidth=0.8)
                            plt.xlabel("Actual correct count (C_can_ref)")
                            plt.ylabel("Predicted correct count (C_ref_hyp)")
                            plt.title(f"{task_id} — {cat_name} — scatter C_can_ref vs C_ref_hyp (r={r_val:.3f})")
                            fname = f"scatter_{task_id}_{cat_name}_C_can_ref_vs_C_ref_hyp_r{r_val:.3f}.png"
                            safe_fname = fname.replace(" ", "_")
                            outpath = scatter_dir / safe_fname
                            plt.tight_layout()
                            plt.savefig(outpath)
                            plt.close()
                            logger.info("Wrote scatter plot -> %s", outpath)
                except Exception as exc:  # pragma: no cover - non-fatal plotting
                    logger.warning("Could not write scatter plot for %s (%s): %s", task_id, cat_name, exc)
                # MAE on correct counts (mean absolute difference)
                mae_counts = float(sel["MAE_EGRA_COR"].dropna().mean()) if ("MAE_EGRA_COR" in sel.columns and not sel["MAE_EGRA_COR"].dropna().empty) else metrics.get("MAE_EGRA_ACC", float("nan"))
                f.write(f"MAE_correct_counts: {mae_counts:.2f}\n")
                # MER kept as fraction (not percent)
                f.write(f"MER: {cat_mer:.2f} (fraction)\n")

                # S, D, I specific metrics
                for proc_code, proc_name in [('S', 'subs'), ('D', 'del'), ('I', 'insert')]:
                    p = cat_data.get(f'{proc_code}_Precision', 0.0)
                    r = cat_data.get(f'{proc_code}_Recall', 0.0)
                    f1 = cat_data.get(f'{proc_code}_F1', 0.0)
                    f.write(f"{proc_name}_prec: {p:.4f}\n")
                    f.write(f"{proc_name}_r: {r:.4f}\n")
                    f.write(f"{proc_name}_f1: {f1:.4f}\n")

                # Aggregated Mistakes metrics
                f.write(f"mistakes_prec: {m_p:.4f}\n")
                f.write(f"mistakes_r: {m_r:.4f}\n")
                f.write(f"mistakes_f1: {m_f1:.4f}\n")

            # --- Formatting for Group B (Isolated) ---
            elif task_id in group_B_tasks:
                f.write(f"wer_ref_hyp: {_fmt4(metrics.get('ASR_WER', 'N/A'))}\n")
                f.write(f"egra_acc: {_fmt4(metrics.get('EGRA-ACC', 'N/A'))}\n")
                
                # Map 'Mistakes' metrics to 'corr_mistake_pred'
                f.write(f"corr_mistake_pred_prec: {m_p:.4f}\n")
                f.write(f"corr_mistake_pred_r: {m_r:.4f}\n")
                f.write(f"corr_mistake_pred_f1: {m_f1:.4f}\n")
                # Two simple baselines for isolated tasks:
                #  - no-mistakes: always predict "no mistake" (never predict a mistake)
                #  - all-mistakes: always predict "mistake" for every item
                if not sel.empty and all(c in sel.columns for c in ["S_can_ref", "D_can_ref", "I_can_ref"]):
                    # Ground-truth binary: 1 = any phonological mistake (S/D/I), 0 = correct
                    gt = ((sel.get("S_can_ref", 0).fillna(0) + sel.get("D_can_ref", 0).fillna(0) + sel.get("I_can_ref", 0).fillna(0)) > 0).astype(int)
                    y_true = gt.values
                    from sklearn.metrics import precision_recall_fscore_support as prfs

                    # No-mistakes baseline: predict all zeros
                    y_pred_no = np.zeros_like(y_true)
                    p_no, r_no, f1_no, _ = prfs(y_true, y_pred_no, labels=[1], zero_division=0)

                    # All-mistakes baseline: predict all ones
                    y_pred_all = np.ones_like(y_true)
                    p_all, r_all, f1_all, _ = prfs(y_true, y_pred_all, labels=[1], zero_division=0)

                    f.write(f"baseline_no_mistakes_prec: {float(p_no[0]):.4f}\n")
                    f.write(f"baseline_no_mistakes_r: {float(r_no[0]):.4f}\n")
                    f.write(f"baseline_no_mistakes_f1: {float(f1_no[0]):.4f}\n")

                    f.write(f"baseline_all_mistakes_prec: {float(p_all[0]):.4f}\n")
                    f.write(f"baseline_all_mistakes_r: {float(r_all[0]):.4f}\n")
                    f.write(f"baseline_all_mistakes_f1: {float(f1_all[0]):.4f}\n")
                else:
                    f.write(f"baseline_no_mistakes_prec: Not Available\n")
                    f.write(f"baseline_no_mistakes_r: Not Available\n")
                    f.write(f"baseline_no_mistakes_f1: Not Available\n")
                    f.write(f"baseline_all_mistakes_prec: Not Available\n")
                    f.write(f"baseline_all_mistakes_r: Not Available\n")
                    f.write(f"baseline_all_mistakes_f1: Not Available\n")
            
            f.write("\n")

    logger.info("Wrote text summary -> %s", summary_path)
    return summary_path


def write_t1_example_walkthrough(df: pd.DataFrame, out_csv: str, logger: logging.Logger) -> Path:
    """
    Produce a step-by-step walkthrough for Task T1 (passage_passage) using a real row.
    The output file illustrates how the metrics that appear in the T1 block of
    egra_eval_summary.txt are computed from canonical/reference/hypothesis text.
    """

    out_path = Path(out_csv).parent / "t1_metric_example.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_cat = _annotate_audio_categories(df)
    sel = df_cat[(df_cat["macro_category"] == "passage") & (df_cat["sub_category"] == "passage")]

    if sel.empty:
        out_path.write_text("No passage_passage rows were found; cannot build the T1 example.\n", encoding="utf-8")
        logger.info("No T1 rows found; wrote placeholder at %s", out_path)
        return out_path

    # Prefer rows that actually have hypotheses attached
    candidates = sel
    if "has_hyp" in candidates.columns:
        candidates = candidates[candidates["has_hyp"] == True]
    if "HYP" in candidates.columns:
        candidates = candidates[candidates["HYP"].notna() & (candidates["HYP"].astype(str) != "")]
    if candidates.empty:
        candidates = sel

    row = candidates.iloc[0]

    can = row.get("canonical_text", row.get("CAN", ""))
    ref = row.get("reference_text", row.get("REF", ""))
    hyp = row.get("hypothesis_text", row.get("HYP", ""))

    # Use same normalization as get_csid_sequence
    def normalize_transcript(text: str) -> str:
        if pd.isna(text) or text is None:
            return ""
        s = str(text).lower()
        s = re.sub(r"[{}]".format(re.escape(string.punctuation)), " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    can_norm = normalize_transcript(can)
    ref_norm = normalize_transcript(ref)
    hyp_norm = normalize_transcript(hyp)

    can_tokens = can_norm.split()
    ref_tokens = ref_norm.split()
    hyp_tokens = hyp_norm.split()

    # Alignment using the same dp_align calls used elsewhere
    _, align_can_ref = dp_align.dp_align(can_tokens, ref_tokens, output_align=True)
    _, align_can_hyp = dp_align.dp_align(can_tokens, hyp_tokens, output_align=True)

    seq_ref = [a[2] for a in align_can_ref]
    seq_hyp = [a[2] for a in align_can_hyp]
    mer_errors, mer_alignment = dp_align.dp_align(seq_ref, seq_hyp, output_align=True)

    fine_metrics = compute_fine_grained_metrics(row)
    phono_by_cat = summary_phonological_by_category(df)
    cat_data = phono_by_cat.get("passage_passage", {})

    # Correlation (r) mirrors the logic in write_text_summary for Group A tasks
    r_val = "Not Available"
    sel_counts = sel[["C_ref_hyp", "C_can_ref"]].dropna() if all(c in sel.columns for c in ["C_ref_hyp", "C_can_ref"]) else pd.DataFrame()
    if not sel_counts.empty and sel_counts["C_ref_hyp"].std() > 0 and sel_counts["C_can_ref"].std() > 0:
        arr_x = sel_counts["C_ref_hyp"].to_numpy()
        arr_y = sel_counts["C_can_ref"].to_numpy()
        r_val = float(np.corrcoef(arr_x, arr_y)[0, 1])

    mae_counts = float(sel["MAE_EGRA_COR"].dropna().mean()) if ("MAE_EGRA_COR" in sel.columns and not sel["MAE_EGRA_COR"].dropna().empty) else float("nan")
    cat_mer = float(sel["MER"].dropna().mean()) if ("MER" in sel.columns and not sel["MER"].dropna().empty) else float("nan")

    m_tp = sum(cat_data.get(f"{k}_TP", 0) for k in ["S", "D", "I"])
    m_fp = sum(cat_data.get(f"{k}_FP", 0) for k in ["S", "D", "I"])
    m_fn = sum(cat_data.get(f"{k}_FN", 0) for k in ["S", "D", "I"])
    m_p = m_tp / (m_tp + m_fp) if (m_tp + m_fp) > 0 else 0.0
    m_r = m_tp / (m_tp + m_fn) if (m_tp + m_fn) > 0 else 0.0
    m_f1 = 2 * m_p * m_r / (m_p + m_r) if (m_p + m_r) > 0 else 0.0

    def _summarize_alignment(alignment, label: str, max_rows: int = 30) -> list[str]:
        lines = [f"Alignment {label} (showing up to {max_rows} rows):"]
        for idx, (test_tok, ref_tok, tag) in enumerate(alignment):
            if idx >= max_rows:
                lines.append(f"... ({len(alignment) - max_rows} more rows omitted)")
                break
            lines.append(f"{idx+1:03d}: CAN='{ref_tok}' | OTHER='{test_tok}' | tag={tag}")
        return lines

    lines: list[str] = []
    lines.append("T1 Metric Walkthrough (passage_passage)\n")
    lines.append("Selected example row (first available with a hypothesis):")
    lines.append(f"- learner_id: {row.get('learner_id', 'N/A')}")
    lines.append(f"- audio_file: {row.get('audio_file', 'N/A')}")
    lines.append(f"- audio_type: {row.get('audio_type', 'N/A')}")
    lines.append(f"- C_can_ref (actual correct count): {row.get('C_can_ref', 'N/A')}")
    lines.append(f"- C_ref_hyp (predicted correct count): {row.get('C_ref_hyp', 'N/A')}")
    lines.append("")

    lines.append("Step 1: Canonical vs Reference alignment (dp_align)")
    lines.append(f"- Canonical token count: {len(can_tokens)} | Reference token count: {len(ref_tokens)}")
    lines.append(f"- csid sequence length: {len(seq_ref)} | counts: {dict(Counter(seq_ref))}")
    lines.extend(_summarize_alignment(align_can_ref, "(CAN vs REF)") )
    lines.append("")

    lines.append("Step 2: Canonical vs Hypothesis alignment (dp_align)")
    lines.append(f"- Hypothesis token count: {len(hyp_tokens)}")
    lines.append(f"- csid sequence length: {len(seq_hyp)} | counts: {dict(Counter(seq_hyp))}")
    lines.extend(_summarize_alignment(align_can_hyp, "(CAN vs HYP)") )
    lines.append("")

    lines.append("Step 3: Meta-alignment of mistake sequences (MER)")
    lines.append(f"- MER WER (sequence-level): {mer_errors.get_wer():.4f}")
    lines.append(f"- MER alignment length: {len(mer_alignment)}")
    lines.append(f"- MER alignment sample (seq_b_token | seq_a_token | align_type):")
    for idx, (pred_tok, true_tok, align_type) in enumerate(mer_alignment[:30]):
        lines.append(f"  {idx+1:03d}: pred='{pred_tok}' | true='{true_tok}' | align={align_type}")
    if len(mer_alignment) > 30:
        lines.append(f"  ... ({len(mer_alignment) - 30} more rows omitted)")
    lines.append("")

    lines.append("Step 4: Row-level fine-grained metrics (from compute_fine_grained_metrics)")
    lines.append(f"- MER (row): {fine_metrics.get('MER', float('nan')):.4f}")
    lines.append(f"- Substitution P/R/F1: {fine_metrics.get('S_Precision', float('nan')):.4f} / {fine_metrics.get('S_Recall', float('nan')):.4f} / {fine_metrics.get('S_F1', float('nan')):.4f}")
    lines.append(f"- Insertion P/R/F1: {fine_metrics.get('I_Precision', float('nan')):.4f} / {fine_metrics.get('I_Recall', float('nan')):.4f} / {fine_metrics.get('I_F1', float('nan')):.4f}")
    lines.append(f"- Deletion P/R/F1: {fine_metrics.get('D_Precision', float('nan')):.4f} / {fine_metrics.get('D_Recall', float('nan')):.4f} / {fine_metrics.get('D_F1', float('nan')):.4f}")
    lines.append(f"- Mistakes (binary) P/R/F1: {fine_metrics.get('Mistakes_Precision', float('nan')):.4f} / {fine_metrics.get('Mistakes_Recall', float('nan')):.4f} / {fine_metrics.get('Mistakes_F1', float('nan')):.4f}")
    lines.append("")

    lines.append("Step 5: How the T1 summary values are formed")
    def _fmt4_local(v):
        try:
            if pd.isna(v):
                return "NaN"
        except Exception:
            pass
        try:
            return f"{float(v):.4f}"
        except Exception:
            return str(v)
    
    lines.append(f"- WER_ref_hyp (row): {_fmt4_local(row.get('WER_ref_hyp', float('nan')))}")
    lines.append(f"- Correlation r_correlation uses all T1 rows' (C_can_ref, C_ref_hyp) pairs; this row contributes ({row.get('C_can_ref', 'N/A')}, {row.get('C_ref_hyp', 'N/A')}) -> r_correlation = {_fmt4_local(r_val)}")
    lines.append(f"- MAE_correct_counts for this row: {abs(row.get('C_can_ref', 0) - row.get('C_ref_hyp', 0)) if pd.notna(row.get('C_can_ref', np.nan)) and pd.notna(row.get('C_ref_hyp', np.nan)) else 'N/A'}; category mean = {mae_counts:.4f}")
    lines.append(f"- Category MER (mean of row MER): {cat_mer:.4f}")
    lines.append(f"- Category substitution P/R/F1: {cat_data.get('S_Precision', 0.0):.4f} / {cat_data.get('S_Recall', 0.0):.4f} / {cat_data.get('S_F1', 0.0):.4f}")
    lines.append(f"- Category deletion P/R/F1: {cat_data.get('D_Precision', 0.0):.4f} / {cat_data.get('D_Recall', 0.0):.4f} / {cat_data.get('D_F1', 0.0):.4f}")
    lines.append(f"- Category insertion P/R/F1: {cat_data.get('I_Precision', 0.0):.4f} / {cat_data.get('I_Recall', 0.0):.4f} / {cat_data.get('I_F1', 0.0):.4f}")
    lines.append(f"- Category mistakes P/R/F1 (aggregated S+D+I): {m_p:.4f} / {m_r:.4f} / {m_f1:.4f}")
    lines.append("")

    lines.append("Step 6: Text snippets (original and normalized)")
    lines.append(f"CAN (original): {can}")
    lines.append(f"CAN (normalized): {can_norm}")
    lines.append(f"REF (original): {ref}")
    lines.append(f"REF (normalized): {ref_norm}")
    lines.append(f"HYP (original): {hyp}")
    lines.append(f"HYP (normalized): {hyp_norm}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote T1 walkthrough -> %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger = setup_logger()
    args = parse_args()

    layout, summary_dirs = resolve_inputs(args, logger)
    ensure_output_dirs(args.out_csv, summary_dirs)

    logger.info("Starting EGRA evaluation pipeline")
    logger.info(
        "EGRA CSV: %s | META CSV: %s | Passages CSV: %s | TextGrid root: %s | Manifest(s): %s",
        args.egra_csv, args.meta_csv, args.passages_csv, layout.textgrid_root, args.nemo_manifest,
    )

    df_egra = pd.read_csv(args.egra_csv)
    df_meta = pd.read_csv(args.meta_csv)
    logger.info("Loaded EGRA rows: %d | META rows: %d", len(df_egra), len(df_meta))

    # Base enrichment shared by both default (PR1) and optional manifest-oriented (PR2) workflows.
    df_egra = adjust_letter_canonical_text(df_egra, logger)
    df_egra = add_audio_keys(df_egra, audio_col="audio_file")

    manifest_df = load_manifest(args, logger)
    df_egra = attach_hypotheses(df_egra, manifest_df, match_on=args.match_on)
    df_egra = add_refs_from_textgrid(
        df_egra,
        base_dir=str(layout.textgrid_root),
        textgrid_col="textgrid",
        tier_name="child",
        logger=logger,
    )
    df_egra = attach_passage_texts(df_egra, args.passages_csv, logger=logger)

    manifest_to_load: Path | None = Path(args.manifest_in) if args.manifest_in else None

    # Optional PR2 workflow: build manifest -> clean manifest.
    if args.build_manifest_first:
        manifests_dir = Path(args.output_root) / "manifests"
        manifests_dir.mkdir(parents=True, exist_ok=True)
        raw_manifest_path = Path(args.manifest_raw_out) if args.manifest_raw_out else manifests_dir / "ref_manifest.raw.jsonl"
        clean_manifest_path = Path(args.manifest_clean_out) if args.manifest_clean_out else manifests_dir / "ref_manifest.clean.jsonl"

        logger.info("Stage A: building reference manifest -> %s", raw_manifest_path)
        built_manifest_df = build_reference_manifest_dataframe(
            df_egra,
            dataset_root=layout.root,
            audio_root=layout.audio_root,
            path_prefix=args.manifest_path_prefix,
            logger=logger,
        )
        write_manifest_jsonl(built_manifest_df, raw_manifest_path)

        logger.info(
            "Stage B: cleaning manifest (train_aggressive, drop_empty=%s) -> %s",
            args.manifest_clean_drop_empty,
            clean_manifest_path,
        )
        clean_stats = clean_manifest_jsonl(
            input_path=raw_manifest_path,
            output_path=clean_manifest_path,
            drop_empty_text=args.manifest_clean_drop_empty,
            text_input_key="ref_text",
            text_output_key=args.manifest_text_key,
        )
        logger.info(
            "Manifest cleaning stats: total=%d, bad_json=%d, written=%d, dropped_empty=%d",
            clean_stats["total_lines"],
            clean_stats["bad_json_skipped"],
            clean_stats["written_lines"],
            clean_stats["dropped_empty_text"],
        )

        manifest_to_load = clean_manifest_path

    if args.manifest_only:
        if not args.build_manifest_first:
            raise SystemExit("--manifest_only requires --build_manifest_first.")
        logger.info("Manifest-only mode enabled; skipping evaluation stage.")
        print("Raw manifest:", raw_manifest_path)
        print("Clean manifest:", clean_manifest_path)
        return

    if manifest_to_load is not None:
        if not manifest_to_load.exists():
            raise SystemExit(f"--manifest_in file not found: {manifest_to_load}")
        logger.info("Stage C: loading cleaned manifest for REF/CAN attachment -> %s", manifest_to_load)
        text_manifest_df = load_text_manifest(
            str(manifest_to_load),
            text_key=args.manifest_text_key,
            can_key="can_text",
            logger=logger,
        )
        df_egra = attach_texts_from_manifest(
            df_egra,
            text_manifest_df,
            match_on=args.match_on,
            logger=logger,
        )

    # Run Basic Evaluation (Jiwer based counts for standard WER/ACC)
    df_results = evaluate(df_egra, df_meta)

    # Ensure per-row WER_ref_hyp is calculated (not just global)
    if not "WER_ref_hyp" in df_results.columns:
        # Calculate WER_ref_hyp per row if missing
        def calc_wer_ref_hyp(row):
            try:
                s = float(row.get("S_ref_hyp", 0))
                d = float(row.get("D_ref_hyp", 0))
                i = float(row.get("I_ref_hyp", 0))
                n = float(row.get("N_ref_hyp", 0))
                return (s + d + i) / n if n > 0 else float("nan")
            except Exception:
                return float("nan")
        df_results["WER_ref_hyp"] = df_results.apply(calc_wer_ref_hyp, axis=1)

    # --- NEW: Run Advanced Metrics using dp_align ---
    df_results = calculate_advanced_metrics(df_results, logger)
    # ------------------------------------------------

    # --- NEW: Compute Phonological Metrics (TP/FP/FN for S/D/I) ---
    logger.info("Computing phonological metrics (TP/FP/FN for substitutions, deletions, insertions)...")
    phonological_metrics = df_results.apply(compute_phonological_metrics_row, axis=1)
    phonological_df = pd.DataFrame(list(phonological_metrics))
    df_results = pd.concat([df_results, phonological_df], axis=1)
    # ----------------------------------------------------------------

    # Ensure all relevant columns are present for output
    required_cols = [
        "learner_id","audio_type","audio_file","CAN","REF","HYP","WER_can_ref","ACC_can_ref","S_can_ref","D_can_ref","I_can_ref","C_can_ref","N_can_ref",
        "WER_can_hyp","ACC_can_hyp","S_can_hyp","D_can_hyp","I_can_hyp","C_can_hyp","N_can_hyp",
        "WER_ref_hyp","ACC_ref_hyp","S_ref_hyp","D_ref_hyp","I_ref_hyp","C_ref_hyp","N_ref_hyp",
        "EGRA-COR","EGRA-ACC","ASR-EGRA-COR","ASR-EGRA-ACC","MAE_EGRA_COR","ASR_WER","has_hyp","gender","child_grade","child_age","region","council","ward",
        "Kiswahili_lang1","Kigogo_lang2","Kirangi_lang3","Kihaya_lang4","Runyambo_lang5","Kihangaza_lang6","English_lang7",
        "ACC_can_ref_norm_for_mae","ACC_can_hyp_norm_for_mae","MAE_EGRA_ACC","Bias_Baseline_MAE",
        "S_Precision","S_Recall","S_F1","I_Precision","I_Recall","I_F1","D_Precision","D_Recall","D_F1",
        "Mistakes_Precision","Mistakes_Recall","Mistakes_F1","Mistake_Error_Rate (MER)",
        "S_TP","S_FP","S_FN","S_Precision","S_Recall","S_F1","D_TP","D_FP","D_FN","D_Precision","D_Recall","D_F1",
        "I_TP","I_FP","I_FN","I_Precision","I_Recall","I_F1"
    ]
    for col in required_cols:
        if col not in df_results.columns:
            df_results[col] = float("nan")

    write_detailed_csv(df_results, args.out_csv, logger)
    summary_txt_path = write_text_summary(df_results, args.out_csv, logger)
    t1_walkthrough_path = write_t1_example_walkthrough(df_results, args.out_csv, logger)
    summary_paths = write_summary_csvs(df_results, summary_dirs, logger)

    print("Detailed results:", args.out_csv)
    print("Summary text:", summary_txt_path)
    print("T1 walkthrough:", t1_walkthrough_path)
    print("Summary directories/files:")
    for path in summary_paths:
        print("  ", path)


if __name__ == "__main__":
    main()
