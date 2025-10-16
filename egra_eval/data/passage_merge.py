from __future__ import annotations

from typing import Optional
import logging
import pandas as pd
import re


def _read_passages_csv(path: str, logger: logging.Logger | None = None) -> pd.DataFrame:
    logger = logger or logging.getLogger("egra_eval")
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, header=None, names=["raw_num", "passage_text"],
                             encoding=enc, engine="python", on_bad_lines="skip")
            logger.info(f"Loaded passages CSV using encoding={enc}")
            return df
        except Exception as e:
            last_err = e
    raise last_err


def attach_passage_texts(
    df_egra: pd.DataFrame,
    passages_csv: str,
    text_col: str = "canonical_text",
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Fill missing canonical_text for rows where audio_type contains 'passage_numX',
    using a two-column CSV: column A = passage number, column B = text.
    """
    logger = logger or logging.getLogger("egra_eval")
    df_pass = _read_passages_csv(passages_csv, logger=logger)

    # Clean & extract passage numbers
    df_pass["raw_num"] = df_pass["raw_num"].astype(str).str.strip()
    df_pass["passage_text"] = df_pass["passage_text"].astype(str).str.strip()
    df_pass = df_pass[df_pass["raw_num"].str.contains(r"\d", regex=True, na=False)].copy()
    df_pass["passage_num"] = df_pass["raw_num"].str.extract(r"(\d+)").astype(int)
    df_pass = df_pass[["passage_num", "passage_text"]]

    df = df_egra.copy()
    df["passage_num"] = df["audio_type"].astype(str).str.extract(r"passage_num(\d+)").astype(float).astype("Int64")

    before_missing = df[text_col].isna().sum() if text_col in df else len(df)
    df = df.merge(df_pass, on="passage_num", how="left")
    df[text_col] = df.get(text_col, pd.Series([None] * len(df))).fillna(df["passage_text"])
    df.drop(columns=["passage_text"], inplace=True)

    after_missing = df[text_col].isna().sum()
    logger.info(f"Passage texts merged. Missing {text_col}: before={before_missing}, after={after_missing}")
    return df

