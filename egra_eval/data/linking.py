from __future__ import annotations

from pathlib import Path
import logging
import pandas as pd


def add_audio_keys(df: pd.DataFrame, audio_col: str = "audio_file") -> pd.DataFrame:
    """Create join keys (name/stem) from the EGRA CSV audio file column."""
    if audio_col not in df.columns:
        raise KeyError(f"Expected column '{audio_col}' not found in EGRA CSV.")
    out = df.copy()
    out["audio_path_csv"] = out[audio_col].astype(str)
    out["audio_name"] = out["audio_path_csv"].apply(lambda p: Path(p).name)
    out["audio_stem"] = out["audio_name"].apply(lambda n: Path(n).stem)
    return out


def attach_hypotheses(
    df_egra: pd.DataFrame,
    df_manifest: pd.DataFrame,
    match_on: str = "stem",  # "stem" | "name" | "path"
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Left-join NeMo HYPs onto EGRA rows.

    match_on:
      - "stem": join on audio_stem (most robust)
      - "name": join on audio_name
      - "path": join on full path
    """
    logger = logger or logging.getLogger("egra_eval")

    if df_manifest is None or df_manifest.empty:
        out = df_egra.copy()
        if "hyp_text" not in out.columns:
            out["hyp_text"] = ""
        logger.info("No ASR hypotheses to attach (empty manifest).")
        return out

    left = df_egra.copy()
    right = df_manifest.copy()

    if match_on == "stem":
        key = "audio_stem"
    elif match_on == "name":
        key = "audio_name"
    elif match_on == "path":
        key = "audio_path_csv"
        right = right.rename(columns={"audio_path": "audio_path_csv"})
    else:
        raise ValueError("match_on must be one of: 'stem' | 'name' | 'path'")

    if key not in left.columns:
        raise KeyError(f"Left frame missing join key '{key}'. Did you call add_audio_keys()?")

    if key not in right.columns:
        raise KeyError(f"Right frame (manifest) missing join key '{key}'.")

    before = left["learner_id"].nunique() if "learner_id" in left else len(left)
    out = left.merge(
        right[[key, "hyp_text"]],
        on=key,
        how="left",
        suffixes=("", "_from_manifest"),
    )

    # Consolidate hyp_text columns if both exist
    if "hyp_text_x" in out.columns and "hyp_text_y" in out.columns:
        out["hyp_text"] = out["hyp_text_x"].fillna(out["hyp_text_y"]).fillna("")
        out.drop(columns=["hyp_text_x", "hyp_text_y"], inplace=True)
    elif "hyp_text" not in out.columns:
        out["hyp_text"] = out.get("hyp_text_from_manifest", "")

    if "hyp_text_from_manifest" in out.columns:
        out.drop(columns=["hyp_text_from_manifest"], inplace=True)

    attached = out["hyp_text"].notna().sum()
    logger.info(f"Attached HYP for ~{attached:,} rows (join key: {key}; EGRA entities before merge: {before:,}).")
    return out

