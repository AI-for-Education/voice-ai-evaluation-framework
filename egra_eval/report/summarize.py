from __future__ import annotations

import math
import pandas as pd

_PAIR_PREFIXES = ("can_ref", "can_hyp", "ref_hyp")


def _safe_div(num: float, den: float) -> float:
    return (num / den) if den else math.nan


def _check_required(df: pd.DataFrame, prefix: str) -> None:
    missing = [f"{stat}_{prefix}" for stat in ("S", "D", "I", "C", "N") if f"{stat}_{prefix}" not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for prefix '{prefix}': {missing}")


def _metric_columns(prefix: str) -> list[str]:
    return [
        f"WER_{prefix}",
        f"ACC_{prefix}",
        f"S_{prefix}",
        f"D_{prefix}",
        f"I_{prefix}",
        f"C_{prefix}",
        f"N_{prefix}",
    ]


def _aggregate_pair(group: pd.DataFrame, prefix: str) -> dict[str, float]:
    S = group[f"S_{prefix}"].sum()
    D = group[f"D_{prefix}"].sum()
    I = group[f"I_{prefix}"].sum()
    C = group[f"C_{prefix}"].sum()
    N = group[f"N_{prefix}"].sum()

    wer = _safe_div((S + D + I) * 100.0, N)
    acc = _safe_div(C * 100.0, N)

    return {
        f"WER_{prefix}": wer,
        f"ACC_{prefix}": acc,
        f"S_{prefix}": S,
        f"D_{prefix}": D,
        f"I_{prefix}": I,
        f"C_{prefix}": C,
        f"N_{prefix}": N,
    }


def summary_for_pair(df: pd.DataFrame, prefix: str, by: list[str] | None = None) -> pd.DataFrame:

    if prefix not in _PAIR_PREFIXES:
        raise ValueError(f"Unsupported prefix '{prefix}'. Expected one of: {_PAIR_PREFIXES}")

    _check_required(df, prefix)
    metric_cols = _metric_columns(prefix)

    if df.empty:
        if by:
            return pd.DataFrame(columns=list(by) + metric_cols)
        return pd.DataFrame(columns=metric_cols)

    if by is None:
        rec = _aggregate_pair(df, prefix)
        return pd.DataFrame([rec])

    rows = []
    grp = df.groupby(by, dropna=False)
    for keys, sub in grp:
        rec = _aggregate_pair(sub, prefix)
        if not isinstance(keys, tuple):
            keys = (keys,)
        for col, val in zip(by, keys):
            rec[col] = val
        rows.append(rec)

    out = pd.DataFrame(rows)
    if not out.empty:
        group_cols = list(by)
        # Ensure consistent column order: group columns first
        other_cols = [col for col in metric_cols if col in out.columns]
        out = out[group_cols + other_cols]
    return out


def _categorize_audio_type(audio_type: str | float) -> tuple[str | None, str | None]:
    if not isinstance(audio_type, str):
        return (None, None)

    at = audio_type.lower()

    if at == "full_letter_grid":
        return ("letters", "grid")
    if at.startswith("iso_letter_"):
        return ("letters", "isolated")
    if at.startswith("iso_random_letters"):
        return ("letters", "random")

    if at.startswith("full_syllable"):
        return ("syllables", "grid")
    if at.startswith("iso_syllable"):
        return ("syllables", "isolated")
    if at.startswith("rand_syllable") or at.startswith("random_syl"):
        return ("syllables", "random")

    if at.startswith("full_nonword"):
        return ("nonwords", "grid")
    if at.startswith("iso_non_word"):
        return ("nonwords", "isolated")
    if at.startswith("iso_random_nw"):
        return ("nonwords", "random")

    if at.startswith("passage_num"):
        return ("passage", "passage")

    return (None, None)


def _annotate_audio_categories(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cats = out.get("audio_type", pd.Series([None] * len(out))).apply(_categorize_audio_type)
    out["macro_category"] = [c[0] for c in cats]
    out["sub_category"] = [c[1] for c in cats]
    return out


def summary_per_speaker(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return summary_for_pair(df, prefix, by=["learner_id"])


def summary_per_speaker_macro(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df_cat = _annotate_audio_categories(df)
    df_cat = df_cat[df_cat["macro_category"].notna()].copy()
    if df_cat.empty:
        cols = ["learner_id", "macro_category"] + _metric_columns(prefix)
        return pd.DataFrame(columns=cols)
    return summary_for_pair(df_cat, prefix, by=["learner_id", "macro_category"])


def summary_per_speaker_subcategory(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df_cat = _annotate_audio_categories(df)
    df_cat = df_cat[df_cat["macro_category"].notna()].copy()
    if df_cat.empty:
        cols = ["learner_id", "macro_category", "sub_category"] + _metric_columns(prefix)
        return pd.DataFrame(columns=cols)
    return summary_for_pair(df_cat, prefix, by=["learner_id", "macro_category", "sub_category"])
