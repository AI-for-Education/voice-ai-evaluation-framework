# egra_eval/eval/run_eval.py
from __future__ import annotations
import logging
import math
import pandas as pd
from egra_eval.metrics.scoring import score

def evaluate(df_egra: pd.DataFrame, df_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Calculează scoruri pe fiecare rând:
      - CAN vs REF  -> WER_can_ref, ACC_can_ref, S/D/I/C/N (C_can_ref = EGRA_COR)
      - CAN vs HYP  -> WER_can_hyp, ACC_can_hyp, S/D/I/C/N (C_can_hyp = ASR_EGRA_COR)
      - REF vs HYP  -> WER_ref_hyp, precision/recall/F1, S/D/I/C/N
      - Acord       -> MAE_COR = |C_can_ref - C_can_hyp|
    """
    logger = logging.getLogger("egra_eval")
    rows = []
    missing_hyp_count = 0

    for _, r in df_egra.iterrows():
        can = r.get("canonical_text", "")  # CAN
        ref = r.get("ref_text", "")        # REF (uman)
        hyp = r.get("hyp_text", "")        # HYP (ASR)
        has_hyp = isinstance(hyp, str) and hyp.strip() != ""
        if not has_hyp:
            missing_hyp_count += 1

        s_can_ref = score(can, ref)
        s_can_hyp = score(can, hyp) if has_hyp else None
        s_ref_hyp = score(ref, hyp) if has_hyp else None

        egra_cor = s_can_ref.C  # = C_can_ref
        asr_egra_cor = s_can_hyp.C if s_can_hyp else math.nan
        mae_cor = abs(egra_cor - asr_egra_cor) if s_can_hyp else math.nan

        acc_can_ref = s_can_ref.ACC * 100.0 if not math.isnan(s_can_ref.ACC) else math.nan
        acc_can_hyp = (
            s_can_hyp.ACC * 100.0 if (s_can_hyp and not math.isnan(s_can_hyp.ACC)) else math.nan
        )

        acc_ref_hyp = (
            s_ref_hyp.ACC * 100.0
            if (s_ref_hyp and not math.isnan(s_ref_hyp.ACC)) else math.nan
        )

        row = {
            "learner_id": r.get("learner_id"),
            "audio_type": r.get("audio_type"),
            "audio_file": r.get("audio_file"),

            # Texte brute
            "CAN": can, "REF": ref, "HYP": hyp,

            # --- CAN vs REF
            "WER_can_ref": s_can_ref.WER,   # deja în procente
            "ACC_can_ref": acc_can_ref,
            "S_can_ref": s_can_ref.S,
            "D_can_ref": s_can_ref.D,
            "I_can_ref": s_can_ref.I,
            "C_can_ref": s_can_ref.C,               # necesar pentru summary_*
            "N_can_ref": s_can_ref.N,

            # --- CAN vs HYP
            "WER_can_hyp": s_can_hyp.WER if s_can_hyp else math.nan,
            "ACC_can_hyp": acc_can_hyp,
            "S_can_hyp": s_can_hyp.S if s_can_hyp else 0,
            "D_can_hyp": s_can_hyp.D if s_can_hyp else 0,
            "I_can_hyp": s_can_hyp.I if s_can_hyp else 0,
            "C_can_hyp": s_can_hyp.C if s_can_hyp else math.nan,               # necesar pentru summary_*
            "N_can_hyp": s_can_hyp.N if s_can_hyp else 0,

            # --- REF vs HYP (calitatea ASR față de uman)
            "WER_ref_hyp": s_ref_hyp.WER if s_ref_hyp else math.nan,    # procente
            "ACC_ref_hyp": acc_ref_hyp,
            "S_ref_hyp": s_ref_hyp.S if s_ref_hyp else 0,
            "D_ref_hyp": s_ref_hyp.D if s_ref_hyp else 0,
            "I_ref_hyp": s_ref_hyp.I if s_ref_hyp else 0,
            "C_ref_hyp": s_ref_hyp.C if s_ref_hyp else math.nan,
            "N_ref_hyp": s_ref_hyp.N if s_ref_hyp else 0,

            # --- Acord între EGRA uman și EGRA-ASR
            "has_hyp": has_hyp,
        }

        row["EGRA_COR"] = egra_cor
        row["EGRA_ACC"] = acc_can_ref
        row["ASR_EGRA_COR"] = asr_egra_cor
        row["ASR_EGRA_ACC"] = acc_can_hyp
        row["MAE_EGRA_COR"] = mae_cor
        row["ASR_WER"] = s_ref_hyp.WER if s_ref_hyp else math.nan

        rows.append(row)

    df_scores = pd.DataFrame(rows)
    logger.info(f"Scored {len(df_scores):,} rows. Merging metadata...")
    if missing_hyp_count:
        logger.info(f"No ASR hypothesis for {missing_hyp_count} row(s); CAN/HYP and REF/HYP metrics set to NaN.")

    # atașează metadata pe learner_id (left join)
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
