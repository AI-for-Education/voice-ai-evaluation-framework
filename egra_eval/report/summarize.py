from __future__ import annotations

import pandas as pd


def macro_summary(df: pd.DataFrame, by=None) -> pd.DataFrame:
    """
    Macro averages of key metrics, optionally grouped by columns.
    Adds a 'samples_per_group' column.
    """
    metrics = ["WER_can_ref", "ACC_can_ref", "WER_can_hyp", "ACC_can_hyp"]
    existing = [m for m in metrics if m in df.columns]
    if by is None:
        out = df[existing].mean(numeric_only=True).to_frame("mean").T
        out["samples_per_group"] = len(df)
        return out

    grp = df.groupby(by, dropna=False)
    out = grp[existing].mean(numeric_only=True).reset_index()
    out["samples_per_group"] = grp.size().values
    return out


def _all_metrics() -> list[str]:
    return [
        # Annotator-based EGRA (CAN vs REF)
        "WER_can_ref", "ACC_can_ref", "EGRA_COR", "EGRA_ACC",
        "S_can_ref", "D_can_ref", "I_can_ref", "C_can_ref", "N_can_ref",

        # ASR-based EGRA (CAN vs HYP)
        "WER_can_hyp", "ACC_can_hyp", "ASR_EGRA_COR", "ASR_EGRA_ACC",
        "S_can_hyp", "D_can_hyp", "I_can_hyp", "C_can_hyp", "N_can_hyp",

        # ASR quality vs human (REF vs HYP)
        "WER_asr", "ASR_precision", "ASR_recall", "ASR_f1",
        "S_ref_hyp", "D_ref_hyp", "I_ref_hyp", "C_ref_hyp", "N_ref_hyp",

        # Agreement between annotator-based and ASR-based EGRA
        "MAE_COR",
    ]


def mean_by(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Generic helper: mean of the full metric set grouped by `by`."""
    metrics = [m for m in _all_metrics() if m in df.columns]
    grp = df.groupby(by, dropna=False)
    out = grp[metrics].mean(numeric_only=True).reset_index()
    out["samples_per_group"] = grp.size().values
    return out


def per_learner_full(df: pd.DataFrame) -> pd.DataFrame:
    """Mean of all metrics per learner_id (one row per learner)."""
    return mean_by(df, by=["learner_id"])


def overall_full(df: pd.DataFrame) -> pd.DataFrame:
    """Single-row mean of all metrics across the entire dataset."""
    metrics = [m for m in _all_metrics() if m in df.columns]
    out = df[metrics].mean(numeric_only=True).to_frame("mean").T
    out.insert(0, "scope", "overall")
    out["samples_per_group"] = len(df)
    return out
