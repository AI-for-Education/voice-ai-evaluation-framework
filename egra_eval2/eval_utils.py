import logging
import math
import pandas as pd
import re
import unicodedata

CONSONANTS = set("bcdfghjklmnpqrstvwxyz")
_punct = re.compile(r"[^\w\s'\u2019\u00C0-\u024F\u1E00-\u1EFF]")
_ws = re.compile(r"\s+")


def precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(tp, fn):
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def balanced_accuracy(tp, tn, fp, fn):
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
    return (sensitivity + specificity) / 2


def accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0


def _safe_div(num: float, den: float) -> float:
    return (num / den) if den else math.nan


def aggregate_error_rate(group: pd.DataFrame, prefix: str) -> float:
    S = group[f"S_{prefix}"].sum()
    D = group[f"D_{prefix}"].sum()
    I = group[f"I_{prefix}"].sum()
    C = group[f"C_{prefix}"].sum()
    N = group[f"N_{prefix}"].sum()

    error_rate = _safe_div((S + D + I) * 100.0, N)
    # acc = _safe_div(C * 100.0, N)

    return error_rate


def text_normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFC", text)
    t = t.lower()
    t = _punct.sub(" ", t)
    t = _ws.sub(" ", t).strip()
    return t


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


def adjust_letter_canonical_text(
    df: pd.DataFrame, logger: logging.Logger
) -> pd.DataFrame:
    if "audio_type" not in df.columns or "canonical_text" not in df.columns:
        return df
    mask = df["audio_type"].astype(str).str.contains("letter", case=False, na=False)
    if not mask.any():
        return df
    logger.info("Adjusting canonical texts for %d letter row(s).", int(mask.sum()))
    out = df.copy()
    out.loc[mask, "canonical_text"] = out.loc[mask, "canonical_text"].apply(
        _append_a_to_consonant_letters
    )
    return out
