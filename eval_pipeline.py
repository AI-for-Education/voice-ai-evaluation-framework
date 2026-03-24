#!/usr/bin/env python3
"""Run EGRA evaluation from a prebuilt cleaned manifest."""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from egra_eval.data.dataset_layout import DatasetLayoutError, resolve_dataset_paths
from egra_eval.eval.run_eval import evaluate
from egra_eval.metrics.phonological import compute_phonological_metrics_row
from egra_eval.pipeline.eval_utils import (
    adjust_letter_canonical_text,
    calculate_advanced_metrics,
)
from egra_eval.report.writers import (
    write_detailed_csv,
    write_summary_csvs,
    write_t1_example_walkthrough,
    write_text_summary,
)


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("egra_eval")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    return logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run EGRA evaluation from cleaned manifest.")
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--manifest_in", required=True, help="Cleaned manifest JSONL.")
    p.add_argument("--output_root", default=None)
    p.add_argument("--egra_csv", default=None)
    p.add_argument("--meta_csv", default=None)
    p.add_argument("--out_csv", default=None)
    p.add_argument("--summary_can_ref_dir", default=None)
    p.add_argument("--summary_can_hyp_dir", default=None)
    p.add_argument("--summary_ref_hyp_dir", default=None)
    p.add_argument("--manifest_audio_key", default="audio_filepath")
    p.add_argument("--manifest_ref_key", default="ref_text")
    p.add_argument("--manifest_can_key", default="can_text")
    p.add_argument("--manifest_hyp_key", default="pred_text")
    return p.parse_args()


def resolve_outputs(args: argparse.Namespace, logger: logging.Logger) -> dict[str, Path]:
    if not args.output_root:
        args.output_root = str(
            Path("input_output_data")
            / "output"
            / "experiments"
            / f"exp_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        )
        logger.info("No --output_root supplied; using default: %s", args.output_root)

    base = Path(args.output_root)
    args.out_csv = args.out_csv or str(base / "egra_eval_detailed.csv")
    args.summary_can_ref_dir = args.summary_can_ref_dir or str(base / "can_ref")
    args.summary_can_hyp_dir = args.summary_can_hyp_dir or str(base / "can_hyp")
    args.summary_ref_hyp_dir = args.summary_ref_hyp_dir or str(base / "ref_hyp")

    summary_dirs = {
        "can_ref": Path(args.summary_can_ref_dir),
        "can_hyp": Path(args.summary_can_hyp_dir),
        "ref_hyp": Path(args.summary_ref_hyp_dir),
    }
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    for d in summary_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return summary_dirs


def load_eval_manifest(
    path: str,
    *,
    audio_key: str,
    ref_key: str,
    can_key: str,
    hyp_key: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    rows = []
    total = 0
    bad = 0
    in_path = Path(path)
    if not in_path.exists():
        raise SystemExit(f"--manifest_in file not found: {in_path}")

    text = in_path.read_text(encoding="utf-8")
    objects: list[dict[str, Any]] = []

    # 1) Try strict JSON first (single object or array of objects).
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            objects = [parsed]
        elif isinstance(parsed, list):
            objects = [obj for obj in parsed if isinstance(obj, dict)]
            bad += sum(1 for obj in parsed if not isinstance(obj, dict))
        else:
            bad += 1
    except json.JSONDecodeError:
        # 2) Fallback: concatenated JSON objects (object after object, possibly pretty-printed).
        decoder = json.JSONDecoder()
        idx = 0
        n = len(text)
        while idx < n:
            while idx < n and text[idx].isspace():
                idx += 1
            if idx >= n:
                break
            try:
                obj, next_idx = decoder.raw_decode(text, idx)
            except json.JSONDecodeError:
                bad += 1
                # Move forward to avoid infinite loop on malformed content.
                idx += 1
                continue
            if isinstance(obj, dict):
                objects.append(obj)
            else:
                bad += 1
            idx = next_idx

    for obj in objects:
        total += 1
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
                "manifest_ref_text": obj.get(ref_key, ""),
                "manifest_can_text": obj.get(can_key, ""),
                "manifest_hyp_text": obj.get(hyp_key, ""),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        # Keep expected columns to avoid downstream KeyError and emit a clear failure.
        out = pd.DataFrame(
            columns=[
                "audio_path",
                "audio_name",
                "audio_stem",
                "manifest_ref_text",
                "manifest_can_text",
                "manifest_hyp_text",
            ]
        )
    logger.info(
        "Loaded eval manifest %s | rows=%d | total_lines=%d | skipped=%d",
        in_path,
        len(out),
        total,
        bad,
    )
    return out


def _valid_text_mask(series: pd.Series) -> pd.Series:
    return series.notna() & (series.astype(str).str.strip() != "")


def _extract_learner_id(audio_stem: str) -> str:
    base = audio_stem.split("-audio", 1)[0]
    m = re.match(r"^(\d{6}_\d{6}_[A-Za-z])$", base)
    if m:
        return m.group(1)
    parts = base.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:3])
    return base


def _infer_audio_type(audio_stem: str) -> str:
    source = audio_stem.split("-audio_", 1)[1] if "-audio_" in audio_stem else audio_stem
    source = re.sub(r"_segment\d+$", "", source)
    s = source.lower()

    if "letters_grids_letters_grid" in s or "letters_grid" in s:
        return "full_letter_grid"
    if "non_words_grid_non_words_grid" in s or "non_words_grid" in s or "nonword" in s:
        return "full_nonword_grid"
    if "grid_syllables_1_syllables_grid_1" in s or "syllables_grid_1" in s or "syllables_grid" in s:
        return "full_syllable1_grid"

    m = re.search(r"iso_letter_(\d+)", s)
    if m:
        return f"iso_letter_{m.group(1)}"
    m = re.search(r"iso_non_word_?(\d+)", s)
    if m:
        return f"iso_non_word{m.group(1)}"
    m = re.search(r"iso_syllable_?(\d+)", s)
    if m:
        return f"iso_syllable{m.group(1)}"
    m = re.search(r"random_syl_?(\d+)", s)
    if m:
        return f"random_syl{m.group(1)}"
    m = re.search(r"passage_?(\d+)", s)
    if m:
        return f"passage_num{m.group(1)}"

    return "unknown"


def build_eval_rows_from_manifest(manifest_df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    if manifest_df is None or manifest_df.empty:
        raise SystemExit(
            "Manifest loaded with 0 valid rows. Ensure --manifest_in is JSONL, JSON array, or concatenated JSON objects "
            "and that each record has a non-empty audio filepath key."
        )

    out = manifest_df.copy()
    out["audio_file"] = out["audio_name"].astype(str)

    seg_mask = out["audio_name"].astype(str).str.contains(r"_segment\d+\.wav$", regex=True, na=False)
    if seg_mask.any() and (~seg_mask).any():
        dropped = int((~seg_mask).sum())
        logger.warning(
            "Manifest contains mixed granularity; dropping %d non-segment row(s) and evaluating only segment rows.",
            dropped,
        )
        out = out.loc[seg_mask].copy()

    if out.empty:
        raise SystemExit("No segment rows found in manifest after filtering.")

    out["learner_id"] = out["audio_stem"].astype(str).apply(_extract_learner_id)
    out["audio_type"] = out["audio_stem"].astype(str).apply(_infer_audio_type)
    out["ref_text"] = out["manifest_ref_text"].fillna("").astype(str)
    out["canonical_text"] = out["manifest_can_text"].fillna("").astype(str)
    out["hyp_text"] = out["manifest_hyp_text"].fillna("").astype(str)

    # Compatibility aliases expected by some metrics helpers.
    out["reference_text"] = out["ref_text"]
    out["hypothesis_text"] = out["hyp_text"]

    logger.info(
        "Prepared segment-level evaluation rows from manifest: rows=%d | non-empty REF=%d | non-empty CAN=%d | non-empty HYP=%d",
        len(out),
        int(_valid_text_mask(out["ref_text"]).sum()),
        int(_valid_text_mask(out["canonical_text"]).sum()),
        int(_valid_text_mask(out["hyp_text"]).sum()),
    )
    return out


def main() -> None:
    logger = setup_logger()
    args = parse_args()

    try:
        layout = resolve_dataset_paths(args.dataset_root)
    except DatasetLayoutError as exc:
        raise SystemExit(str(exc)) from exc

    summary_dirs = resolve_outputs(args, logger)
    args.meta_csv = args.meta_csv or str(layout.metadata_csv)
    df_meta = pd.read_csv(args.meta_csv)
    logger.info("Loaded META rows: %d", len(df_meta))

    manifest_df = load_eval_manifest(
        args.manifest_in,
        audio_key=args.manifest_audio_key,
        ref_key=args.manifest_ref_key,
        can_key=args.manifest_can_key,
        hyp_key=args.manifest_hyp_key,
        logger=logger,
    )
    df_eval = build_eval_rows_from_manifest(manifest_df, logger)
    df_eval = adjust_letter_canonical_text(df_eval, logger)

    df_results = evaluate(df_eval, df_meta)

    if "WER_ref_hyp" not in df_results.columns:
        def calc_wer_ref_hyp(row: pd.Series) -> float:
            try:
                s = float(row.get("S_ref_hyp", 0))
                d = float(row.get("D_ref_hyp", 0))
                i = float(row.get("I_ref_hyp", 0))
                n = float(row.get("N_ref_hyp", 0))
                return (s + d + i) / n if n > 0 else float("nan")
            except Exception:
                return float("nan")

        df_results["WER_ref_hyp"] = df_results.apply(calc_wer_ref_hyp, axis=1)

    df_results = calculate_advanced_metrics(df_results, logger)
    logger.info("Computing phonological metrics (TP/FP/FN for substitutions, deletions, insertions)...")
    phonological_metrics = df_results.apply(compute_phonological_metrics_row, axis=1)
    phonological_df = pd.DataFrame(list(phonological_metrics))
    df_results = pd.concat([df_results, phonological_df], axis=1)

    required_cols = [
        "learner_id", "audio_type", "audio_file", "CAN", "REF", "HYP",
        "WER_can_ref", "ACC_can_ref", "S_can_ref", "D_can_ref", "I_can_ref", "C_can_ref", "N_can_ref",
        "WER_can_hyp", "ACC_can_hyp", "S_can_hyp", "D_can_hyp", "I_can_hyp", "C_can_hyp", "N_can_hyp",
        "WER_ref_hyp", "ACC_ref_hyp", "S_ref_hyp", "D_ref_hyp", "I_ref_hyp", "C_ref_hyp", "N_ref_hyp",
        "EGRA-COR", "EGRA-ACC", "ASR-EGRA-COR", "ASR-EGRA-ACC", "MAE_EGRA_COR", "ASR_WER", "has_hyp",
        "gender", "child_grade", "child_age", "region", "council", "ward",
        "Kiswahili_lang1", "Kigogo_lang2", "Kirangi_lang3", "Kihaya_lang4", "Runyambo_lang5",
        "Kihangaza_lang6", "English_lang7",
        "ACC_can_ref_norm_for_mae", "ACC_can_hyp_norm_for_mae", "MAE_EGRA_ACC", "Bias_Baseline_MAE",
        "S_Precision", "S_Recall", "S_F1", "I_Precision", "I_Recall", "I_F1", "D_Precision", "D_Recall",
        "D_F1", "Mistakes_Precision", "Mistakes_Recall", "Mistakes_F1", "Mistake_Error_Rate (MER)",
        "S_TP", "S_FP", "S_FN", "D_TP", "D_FP", "D_FN", "I_TP", "I_FP", "I_FN",
    ]
    for col in required_cols:
        if col not in df_results.columns:
            df_results[col] = np.nan

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
