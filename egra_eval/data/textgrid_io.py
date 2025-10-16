from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import logging
import pandas as pd
from praatio import textgrid


def _find_tier_case_insensitive(tg: textgrid.Textgrid, tier_name: str):
    for name in tg.tierNames:
        if name.lower() == tier_name.lower():
            return tg.getTier(name)
    return None


def _entries_from_tier(tier) -> List[Tuple[float, float, str]]:
    """
    Normalize Praatio 5.x / 6.x tier entries into a list of (start, end, label).
    """
    entries = getattr(tier, "entries", None)
    if entries is None:
        entries = getattr(tier, "entryList", [])

    out = []
    for e in entries:
        if isinstance(e, (tuple, list)) and len(e) >= 3:
            start, end, lab = e[0], e[1], e[2]
        else:
            start = getattr(e, "start", None)
            end = getattr(e, "end", None)
            lab = getattr(e, "label", "")
        if start is None or end is None:
            continue
        out.append((float(start), float(end), (lab or "").strip()))
    return out


def _labels_from_tier(tier) -> list[str]:
    return [lab for _s, _e, lab in _entries_from_tier(tier) if lab]


def read_ref_from_textgrid(path: str, tier_name: str = "child") -> str:
    if not Path(path).exists():
        return ""
    tg = textgrid.openTextgrid(
        path,
        includeEmptyIntervals=False,
        reportingMode="silence",
    )
    tier = _find_tier_case_insensitive(tg, tier_name)
    if tier is None:
        return ""
    labels = _labels_from_tier(tier)
    return " ".join(labels).strip()


def add_refs_from_textgrid(
    df: pd.DataFrame,
    base_dir: str,
    textgrid_col: str = "textgrid",
    tier_name: str = "child",
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Adds a 'ref_text' column to df by reading TextGrid files for each learner.

    Each learner has a folder named <learner_id> inside base_dir.
    The TextGrid filenames are listed in df[textgrid_col].
    """
    logger = logger or logging.getLogger("egra_eval")

    if "learner_id" not in df.columns or textgrid_col not in df.columns:
        logger.warning("DataFrame missing 'learner_id' or TextGrid filename column; REF text will be empty.")
        out = df.copy()
        out["ref_text"] = ""
        return out

    ref_texts = []
    missing = 0
    failed = 0

    for _, row in df.iterrows():
        learner_id = str(row.get("learner_id", "")).strip()
        tg_name = str(row.get(textgrid_col, "")).strip()

        if not learner_id or not tg_name:
            ref_texts.append("")
            missing += 1
            continue

        tg_path = Path(base_dir) / learner_id / tg_name
        if not tg_path.exists():
            logger.warning(f"Missing TextGrid: {tg_path}")
            ref_texts.append("")
            missing += 1
            continue

        try:
            ref = read_ref_from_textgrid(str(tg_path), tier_name=tier_name)
        except Exception as e:
            logger.warning(f"Failed to read {tg_path}: {e}")
            ref = ""
            failed += 1

        ref_texts.append(ref)

    out = df.copy()
    out["ref_text"] = ref_texts
    logger.info(f"Attached REF text for {len(out):,} rows (missing={missing}, failed={failed}).")
    return out

