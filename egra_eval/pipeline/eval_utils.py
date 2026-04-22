from __future__ import annotations

import logging
import re
import string
from typing import List

import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

import dp_align


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


def get_csid_sequence(canonical_text: str, other_text: str) -> List[str]:
    if pd.isna(canonical_text):
        canonical_text = ""
    if pd.isna(other_text):
        other_text = ""

    def normalize_transcript(text: str) -> str:
        if text is None:
            return ""
        s = str(text).lower()
        s = re.sub(r"[{}]".format(re.escape(string.punctuation)), " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    can_norm = normalize_transcript(canonical_text)
    other_norm = normalize_transcript(other_text)

    can_list = can_norm.split()
    other_list = other_norm.split()

    _, alignment = dp_align.dp_align(can_list, other_list, output_align=True)
    return [item[2] for item in alignment]


def compute_fine_grained_metrics(row) -> pd.Series:
    can = row.get("CAN_FG", row.get("canonical_text", row.get("CAN", "")))
    ref = row.get("REF_FG", row.get("reference_text", row.get("REF", "")))
    hyp = row.get("HYP_FG", row.get("hypothesis_text", row.get("HYP", "")))

    def _to_text(v) -> str:
        if v is None:
            return ""
        try:
            if pd.isna(v):
                return ""
        except Exception:
            pass
        return str(v)

    can = _to_text(can)
    ref = _to_text(ref)
    hyp = _to_text(hyp)

    empty_metrics = {
        "S_Precision": np.nan,
        "S_Recall": np.nan,
        "S_F1": np.nan,
        "I_Precision": np.nan,
        "I_Recall": np.nan,
        "I_F1": np.nan,
        "D_Precision": np.nan,
        "D_Recall": np.nan,
        "D_F1": np.nan,
        "Mistakes_Precision": np.nan,
        "Mistakes_Recall": np.nan,
        "Mistakes_F1": np.nan,
        "MER": np.nan,
    }

    # MER/fine-grained alignment is undefined without a canonical sequence.
    if not can.strip():
        return pd.Series(empty_metrics)

    seq_a = get_csid_sequence(can, ref)
    seq_b = get_csid_sequence(can, hyp)

    mer_errors, mer_alignment = dp_align.dp_align(seq_a, seq_b, output_align=True)
    mer_val = mer_errors.get_wer() if mer_errors.n_total > 0 else np.nan

    y_pred = [x[0] for x in mer_alignment]
    y_true = [x[1] for x in mer_alignment]

    metrics_res = {}
    labels = ["s", "i", "d"]
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)

    for i, label in enumerate(labels):
        metrics_res[f"{label.upper()}_Precision"] = p[i]
        metrics_res[f"{label.upper()}_Recall"] = r[i]
        metrics_res[f"{label.upper()}_F1"] = f1[i]

    def binarize(token):
        if token in ["s", "i", "d"]:
            return "x"
        return "c"

    y_true_bin = [binarize(t) for t in y_true]
    y_pred_bin = [binarize(t) for t in y_pred]
    p_x, r_x, f1_x, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, labels=["x"], zero_division=0
    )

    metrics_res["Mistakes_Precision"] = p_x[0]
    metrics_res["Mistakes_Recall"] = r_x[0]
    metrics_res["Mistakes_F1"] = f1_x[0]
    metrics_res["MER"] = mer_val
    return pd.Series(metrics_res)


def calculate_advanced_metrics(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    logger.info("Calculating advanced metrics (MAE, Bias, MER, P/R/F1) using dp_align...")

    if "ACC_can_ref" in df.columns and "ACC_can_hyp" in df.columns:
        ref_max = df["ACC_can_ref"].dropna().max() if not df["ACC_can_ref"].dropna().empty else 0
        hyp_max = df["ACC_can_hyp"].dropna().max() if not df["ACC_can_hyp"].dropna().empty else 0
        if ref_max > 1.0 or hyp_max > 1.0:
            logger.info("Detected ACC in 0-100 scale; converting ACC_can_ref/ACC_can_hyp to 0..1 scale for MAE calculation.")
            df["ACC_can_ref_norm_for_mae"] = df["ACC_can_ref"] / 100.0
            df["ACC_can_hyp_norm_for_mae"] = df["ACC_can_hyp"] / 100.0
        else:
            df["ACC_can_ref_norm_for_mae"] = df["ACC_can_ref"]
            df["ACC_can_hyp_norm_for_mae"] = df["ACC_can_hyp"]

        df["MAE_EGRA_ACC"] = (df["ACC_can_ref_norm_for_mae"] - df["ACC_can_hyp_norm_for_mae"]).abs()
    else:
        df["MAE_EGRA_ACC"] = np.nan

    if "C_can_ref" in df.columns and "C_can_hyp" in df.columns:
        df["MAE_EGRA_COR"] = (df["C_can_ref"] - df["C_can_hyp"]).abs()
    else:
        df["MAE_EGRA_COR"] = np.nan

    if "ACC_can_ref" in df.columns:
        mean_acc = df["ACC_can_ref"].mean()
        df["Bias_Baseline_MAE"] = (df["ACC_can_ref"] - mean_acc).abs()
        logger.info("Bias Baseline (Mean Accuracy across dataset): %.4f", mean_acc)
    else:
        df["Bias_Baseline_MAE"] = np.nan

    has_lowercase = all(c in df.columns for c in ["canonical_text", "reference_text", "hypothesis_text"])
    has_uppercase = all(c in df.columns for c in ["CAN", "REF", "HYP"])
    if has_lowercase or has_uppercase:
        logger.info("Running alignment-of-alignments (MER/Fine-grained)...")
        fine_grained = df.apply(compute_fine_grained_metrics, axis=1)
        df = pd.concat([df, fine_grained], axis=1)

    return df
