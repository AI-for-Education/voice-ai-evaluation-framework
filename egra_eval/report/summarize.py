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


def aggregate_phonological_metrics(df: pd.DataFrame, by: list[str] | None = None) -> dict[str, float]:
    """
    Aggregate phonological metrics (TP/FP/FN, Precision/Recall/F1) for substitutions, deletions, insertions.
    
    Args:
        df: DataFrame with phonological metric columns (S_TP, S_FP, S_FN, etc.)
        by: Optional list of columns to group by. If None, aggregates across all rows.
    
    Returns:
        Dictionary with aggregated metrics
    """
    if df.empty:
        return {}
    
    # Columns for each error type
    error_types = {
        'S': ['S_TP', 'S_FP', 'S_FN', 'S_Precision', 'S_Recall', 'S_F1'],
        'D': ['D_TP', 'D_FP', 'D_FN', 'D_Precision', 'D_Recall', 'D_F1'],
        'I': ['I_TP', 'I_FP', 'I_FN', 'I_Precision', 'I_Recall', 'I_F1'],
    }
    
    # Check which columns exist
    available_cols = set(df.columns)
    
    result = {}
    
    if by is None:
        # Aggregate across all rows
        for prefix, cols in error_types.items():
            existing_cols = [c for c in cols if c in available_cols]
            if not existing_cols:
                continue
            
            # Sum TP, FP, FN
            if f'{prefix}_TP' in existing_cols:
                tp = df[f'{prefix}_TP'].sum()
                fp = df[f'{prefix}_FP'].sum() if f'{prefix}_FP' in existing_cols else 0
                fn = df[f'{prefix}_FN'].sum() if f'{prefix}_FN' in existing_cols else 0
                
                result[f'{prefix}_TP'] = float(tp)
                result[f'{prefix}_FP'] = float(fp)
                result[f'{prefix}_FN'] = float(fn)
                
                # Compute aggregated Precision, Recall, F1
                precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
                recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
                f1 = 2 * (precision * recall) / (precision + recall) if not (precision + recall == 0 or (precision != precision) or (recall != recall)) else float('nan')
                
                result[f'{prefix}_Precision'] = precision
                result[f'{prefix}_Recall'] = recall
                result[f'{prefix}_F1'] = f1
    else:
        # Group by specified columns
        grouped = df.groupby(by, dropna=False)
        for keys, sub_df in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            
            for prefix, cols in error_types.items():
                existing_cols = [c for c in cols if c in available_cols]
                if not existing_cols:
                    continue
                
                tp = sub_df[f'{prefix}_TP'].sum() if f'{prefix}_TP' in existing_cols else 0
                fp = sub_df[f'{prefix}_FP'].sum() if f'{prefix}_FP' in existing_cols else 0
                fn = sub_df[f'{prefix}_FN'].sum() if f'{prefix}_FN' in existing_cols else 0
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
                recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
                f1 = 2 * (precision * recall) / (precision + recall) if not (precision + recall == 0 or (precision != precision) or (recall != recall)) else float('nan')
                
                # Store with group keys as prefix
                key_str = '_'.join(str(k) for k in keys)
                result[f'{key_str}_{prefix}_TP'] = float(tp)
                result[f'{key_str}_{prefix}_FP'] = float(fp)
                result[f'{key_str}_{prefix}_FN'] = float(fn)
                result[f'{key_str}_{prefix}_Precision'] = precision
                result[f'{key_str}_{prefix}_Recall'] = recall
                result[f'{key_str}_{prefix}_F1'] = f1
    
    return result


def summary_phonological_by_category(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    Compute phonological metrics aggregated by category.
    
    Args:
        df: DataFrame with phonological metrics and category columns
    
    Returns:
        Dictionary mapping category names to aggregated metrics
    """
    df_cat = _annotate_audio_categories(df)
    df_cat = df_cat[df_cat["macro_category"].notna()].copy()
    
    if df_cat.empty:
        return {}
    
    results = {}
    
    # Overall aggregate
    results['__OVERALL__'] = aggregate_phonological_metrics(df_cat, by=None)
    
    # By macro category
    for macro_cat in df_cat["macro_category"].unique():
        if pd.isna(macro_cat):
            continue
        sub_df = df_cat[df_cat["macro_category"] == macro_cat]
        results[f'macro_{macro_cat}'] = aggregate_phonological_metrics(sub_df, by=None)
    
    # By macro + sub category
    for (macro_cat, sub_cat), sub_df in df_cat.groupby(["macro_category", "sub_category"], dropna=False):
        if pd.isna(macro_cat) or pd.isna(sub_cat):
            continue
        results[f'{macro_cat}_{sub_cat}'] = aggregate_phonological_metrics(sub_df, by=None)
    
    return results
