# egra_eval/eval/run_eval.py
from __future__ import annotations
import logging
import math
import re
import pandas as pd
from egra_eval2.scoring import score


_SEGMENT_RE = re.compile(r"_segment(\d+)\.wav$", re.IGNORECASE)


def _group_key_and_order(audio_file: str) -> tuple[str, int]:
    """
    Build a stable group key for files that belong to the same original audio item
    and a sortable segment order.
    """
    af = str(audio_file or "")
    m = _SEGMENT_RE.search(af)
    if m:
        return _SEGMENT_RE.sub(".wav", af), int(m.group(1))
    return af, 0


def _safe_join_text(series: pd.Series) -> str:
    parts = []
    for val in series:
        if not isinstance(val, str):
            continue
        s = val.strip()
        if s:
            parts.append(s)
    return " ".join(parts)


def _first_non_empty_text(series: pd.Series) -> str:
    for val in series:
        if not isinstance(val, str):
            continue
        s = val.strip()
        if s:
            return s
    return ""


def evaluate(df_egra: pd.DataFrame, df_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Compute row-level scores:
      - CAN vs REF  -> WER_can_ref, ACC_can_ref, S/D/I/C/N (C_can_ref = EGRA_COR)
      - CAN vs HYP  -> WER_can_hyp, ACC_can_hyp, S/D/I/C/N (C_can_hyp = ASR_EGRA_COR)
      - REF vs HYP  -> WER_ref_hyp, precision/recall/F1, S/D/I/C/N
      - Agreement   -> MAE_COR = |C_can_ref - C_can_hyp|
    """
    logger = logging.getLogger("egra_eval")
    rows = []
    missing_hyp_count = 0

    work = df_egra.copy()
    work["_group_key"], work["_seg_order"] = zip(
        *work.get("audio_file", pd.Series([""] * len(work))).astype(str).apply(_group_key_and_order)
    )

    # Compute CAN-vs-(concat REF/HYP) once per original file group.
    # These values are attached only on one representative row per group to avoid
    # duplicating CAN metrics for each segment in summary aggregations.
    group_scores: dict[int, dict[str, float | int]] = {}
    for _, g in work.groupby("_group_key", dropna=False):
        g_sorted = g.sort_values(by="_seg_order", kind="stable")
        rep_idx = int(g_sorted.index[0])
        # Canonical belongs to the original item, not each segment. Concatenating
        # it across segments duplicates CAN and inflates CAN-side WER.
        can = _first_non_empty_text(g_sorted["canonical_text"]) if "canonical_text" in g_sorted.columns else ""
        if not can and "canonical_text" in g_sorted.columns:
            # Keep legacy behavior: if canonical is empty after join, use first row raw value.
            first_can = g_sorted["canonical_text"].iloc[0]
            can = first_can if isinstance(first_can, str) else ""
        ref_concat = _safe_join_text(g_sorted["ref_text"]) if "ref_text" in g_sorted.columns else ""
        hyp_concat = _safe_join_text(g_sorted["hyp_text"]) if "hyp_text" in g_sorted.columns else ""

        # For EGRA (annotation-based), use ANN/REF as truth and CAN as hypothesis:
        #   EGRA-COR = N_ann - S - D
        #   EGRA-ACC = EGRA-COR / N_ann
        # Keep variable name for compatibility with downstream columns.
        s_can_ref = score(ref_concat, can)
        s_can_hyp = score(can, hyp_concat)
        egra_cor = s_can_ref.COR
        acc_can_ref = s_can_ref.ACC_COR * 100.0 if not math.isnan(s_can_ref.ACC_COR) else math.nan
        acc_can_hyp = s_can_hyp.ACC * 100.0 if not math.isnan(s_can_hyp.ACC) else math.nan
        asr_egra_cor = s_can_hyp.COR
        group_scores[rep_idx] = {
            "WER_can_ref": s_can_ref.WER,
            "ACC_can_ref": acc_can_ref,
            "S_can_ref": s_can_ref.S,
            "D_can_ref": s_can_ref.D,
            "I_can_ref": s_can_ref.I,
            "C_can_ref": egra_cor,
            "N_can_ref": s_can_ref.N,
            "WER_can_hyp": s_can_hyp.WER,
            "ACC_can_hyp": acc_can_hyp,
            "S_can_hyp": s_can_hyp.S,
            "D_can_hyp": s_can_hyp.D,
            "I_can_hyp": s_can_hyp.I,
            "C_can_hyp": asr_egra_cor,
            "N_can_hyp": s_can_hyp.N,
            "EGRA_COR": egra_cor,
            "EGRA_ACC": acc_can_ref,
            "ASR_EGRA_COR": asr_egra_cor,
            "ASR_EGRA_ACC": acc_can_hyp,
            "MAE_EGRA_COR": abs(egra_cor - asr_egra_cor),
            # Text triplet used by advanced/fine-grained metrics.
            "CAN_FG": can,
            "REF_FG": ref_concat,
            "HYP_FG": hyp_concat,
        }

    for idx, r in work.iterrows():
        can = r.get("canonical_text", "")  # CAN
        ref = r.get("ref_text", "")        # REF (human)
        hyp = r.get("hyp_text", "")        # HYP (ASR)
        has_hyp = isinstance(hyp, str) and hyp.strip() != ""
        if not has_hyp:
            missing_hyp_count += 1

        s_ref_hyp = score(ref, hyp)

        acc_ref_hyp = s_ref_hyp.ACC * 100.0 if not math.isnan(s_ref_hyp.ACC) else math.nan

        can_block = {
            "WER_can_ref": math.nan,
            "ACC_can_ref": math.nan,
            "S_can_ref": math.nan,
            "D_can_ref": math.nan,
            "I_can_ref": math.nan,
            "C_can_ref": math.nan,
            "N_can_ref": math.nan,
            "WER_can_hyp": math.nan,
            "ACC_can_hyp": math.nan,
            "S_can_hyp": math.nan,
            "D_can_hyp": math.nan,
            "I_can_hyp": math.nan,
            "C_can_hyp": math.nan,
            "N_can_hyp": math.nan,
            "EGRA_COR": math.nan,
            "EGRA_ACC": math.nan,
            "ASR_EGRA_COR": math.nan,
            "ASR_EGRA_ACC": math.nan,
            "MAE_EGRA_COR": math.nan,
            "CAN_FG": None,
            "REF_FG": None,
            "HYP_FG": None,
        }
        if int(idx) in group_scores:
            can_block.update(group_scores[int(idx)])

        row = {
            "learner_id": r.get("learner_id"),
            "audio_type": r.get("audio_type"),
            "audio_file": r.get("audio_file"),

            # Texte brute
            "CAN": can, "REF": ref, "HYP": hyp,
            "CAN_FG": can_block["CAN_FG"],
            "REF_FG": can_block["REF_FG"],
            "HYP_FG": can_block["HYP_FG"],

            # --- CAN vs REF
            "WER_can_ref": can_block["WER_can_ref"],   # already in percentage units
            "ACC_can_ref": can_block["ACC_can_ref"],
            "S_can_ref": can_block["S_can_ref"],
            "D_can_ref": can_block["D_can_ref"],
            "I_can_ref": can_block["I_can_ref"],
            "C_can_ref": can_block["C_can_ref"],               # necesar pentru summary_*
            "N_can_ref": can_block["N_can_ref"],

            # --- CAN vs HYP
            "WER_can_hyp": can_block["WER_can_hyp"],
            "ACC_can_hyp": can_block["ACC_can_hyp"],
            "S_can_hyp": can_block["S_can_hyp"],
            "D_can_hyp": can_block["D_can_hyp"],
            "I_can_hyp": can_block["I_can_hyp"],
            "C_can_hyp": can_block["C_can_hyp"],               # required for summary_*
            "N_can_hyp": can_block["N_can_hyp"],

            # --- REF vs HYP (ASR quality against human reference)
            "WER_ref_hyp": s_ref_hyp.WER,    # percent
            "ACC_ref_hyp": acc_ref_hyp,
            "S_ref_hyp": s_ref_hyp.S,
            "D_ref_hyp": s_ref_hyp.D,
            "I_ref_hyp": s_ref_hyp.I,
            "C_ref_hyp": s_ref_hyp.C,
            "N_ref_hyp": s_ref_hyp.N,

            # --- Agreement between human EGRA and ASR EGRA
            "has_hyp": has_hyp,
        }

        row["EGRA_COR"] = can_block["EGRA_COR"]
        row["EGRA_ACC"] = can_block["EGRA_ACC"]
        row["ASR_EGRA_COR"] = can_block["ASR_EGRA_COR"]
        row["ASR_EGRA_ACC"] = can_block["ASR_EGRA_ACC"]
        row["MAE_EGRA_COR"] = can_block["MAE_EGRA_COR"]
        row["ASR_WER"] = s_ref_hyp.WER

        rows.append(row)

    df_scores = pd.DataFrame(rows)
    logger.info(f"Scored {len(df_scores):,} rows. Merging metadata...")
    if missing_hyp_count:
        logger.info(f"ASR hypothesis is empty for {missing_hyp_count} row(s); scored using empty-hypothesis behavior.")

    # Attach metadata by learner_id (left join).
    out = df_scores.merge(df_meta, on="learner_id", how="left")

    metric_cols_order = [
        "WER_can_ref", "ACC_can_ref",
        "S_can_ref", "D_can_ref", "I_can_ref", "C_can_ref", "N_can_ref",
        "WER_can_hyp", "ACC_can_hyp",
        "S_can_hyp", "D_can_hyp", "I_can_hyp", "C_can_hyp", "N_can_hyp",
        "WER_ref_hyp", "ACC_ref_hyp",
        "S_ref_hyp", "D_ref_hyp", "I_ref_hyp", "C_ref_hyp", "N_ref_hyp",
    ]
    extra_cols_order = [
        "EGRA_COR",
        "EGRA_ACC",
        "ASR_EGRA_COR",
        "ASR_EGRA_ACC",
        "MAE_EGRA_COR",
        "ASR_WER",
    ]
    base_cols = [
        "learner_id", "audio_type", "audio_file",
        "CAN", "REF", "HYP",
    ] + metric_cols_order + extra_cols_order + ["has_hyp"]
    existing_base = [c for c in base_cols if c in out.columns]
    other_cols = [c for c in out.columns if c not in existing_base]
    out = out[existing_base + other_cols]
    missing = out["learner_id"].isna().sum()
    if missing:
        logger.warning(f"Metadata merge left {missing} rows without learner_id match.")
    return out
