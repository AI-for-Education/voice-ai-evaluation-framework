from __future__ import annotations

import logging
import pandas as pd
from egra_eval.metrics.scoring import score


def evaluate(df_egra: pd.DataFrame, df_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Row-wise evaluation:
      - Annotator-based EGRA (CAN vs REF): EGRA_COR, EGRA_ACC
      - ASR-based   EGRA (CAN vs HYP): ASR_EGRA_COR, ASR_EGRA_ACC
      - ASR quality vs human (REF vs HYP): WER + P/R/F1
      - Agreement: MAE between the two COR measures
    """
    logger = logging.getLogger("egra_eval")

    rows = []
    for idx, r in df_egra.iterrows():
        can = r.get("canonical_text", "")  # CAN
        ref = r.get("ref_text", "")        # REF (annotator)
        hyp = r.get("hyp_text", "")        # HYP (ASR)

        # 1) CAN vs REF -> annotator-based EGRA
        s_can_ref = score(can, ref)

        # 2) CAN vs HYP -> ASR-based EGRA
        s_can_hyp = score(can, hyp)

        # 3) REF vs HYP -> ASR quality vs human
        s_ref_hyp = score(ref, hyp)

        egra_cor     = s_can_ref.C
        egra_acc     = s_can_ref.ACC
        asr_egra_cor = s_can_hyp.C
        asr_egra_acc = s_can_hyp.ACC
        mae_cor      = abs(egra_cor - asr_egra_cor)

        rows.append({
            "learner_id": r["learner_id"],
            "audio_type": r["audio_type"],
            "audio_file": r["audio_file"],

            # Raw texts
            "CAN": can, "REF": ref, "HYP": hyp,

            # Annotator-based EGRA
            "WER_can_ref": s_can_ref.WER,
            "ACC_can_ref": s_can_ref.ACC,
            "EGRA_COR": egra_cor,
            "EGRA_ACC": egra_acc,
            "S_can_ref": s_can_ref.S,
            "D_can_ref": s_can_ref.D,
            "I_can_ref": s_can_ref.I,
            "C_can_ref": s_can_ref.C,
            "N_can_ref": s_can_ref.N,

            # ASR-based EGRA
            "WER_can_hyp": s_can_hyp.WER,
            "ACC_can_hyp": s_can_hyp.ACC,
            "ASR_EGRA_COR": asr_egra_cor,
            "ASR_EGRA_ACC": asr_egra_acc,
            "S_can_hyp": s_can_hyp.S,
            "D_can_hyp": s_can_hyp.D,
            "I_can_hyp": s_can_hyp.I,
            "C_can_hyp": s_can_hyp.C,
            "N_can_hyp": s_can_hyp.N,

            # ASR quality vs human (REF–HYP)
            "WER_asr": s_ref_hyp.WER,
            "ASR_precision": s_ref_hyp.precision,
            "ASR_recall": s_ref_hyp.recall,
            "ASR_f1": s_ref_hyp.f1,
            "S_ref_hyp": s_ref_hyp.S,
            "D_ref_hyp": s_ref_hyp.D,
            "I_ref_hyp": s_ref_hyp.I,
            "C_ref_hyp": s_ref_hyp.C,
            "N_ref_hyp": s_ref_hyp.N,

            # Agreement
            "MAE_COR": mae_cor,
        })

    df_scores = pd.DataFrame(rows)
    logger.info(f"Scored {len(df_scores):,} rows. Merging metadata...")

    # Attach learner metadata (e.g., gender, age)
    out = df_scores.merge(df_meta, on="learner_id", how="left")
    missing_meta = out["learner_id"].isna().sum()
    if missing_meta:
        logger.warning(f"Metadata merge left {missing_meta} rows without a match on learner_id.")
    return out

