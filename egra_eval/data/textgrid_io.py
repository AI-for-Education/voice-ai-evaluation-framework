from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

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


EXCLUDED_TAGS = {
    "<uu>",
    "<unk>",
    "<um>",
    "<noise>",
    "<mm>",
    "<eh>",
    "<ee>",
    "<ah>",
}


def _labels_from_tier(tier) -> list[str]:
    """
    Return the text labels for a tier, ignoring the <enumerator> tag.
    """
    labels = []
    for _s, _e, lab in _entries_from_tier(tier):
        if not lab:
            continue
        normalized = lab.strip().lower()
        if normalized == "<enumerator>":
            continue
        tokens = lab.split()
        filtered_tokens = [tok for tok in tokens if tok.strip().lower() not in EXCLUDED_TAGS]
        filtered = " ".join(filtered_tokens).strip()
        if not filtered:
            continue
        labels.append(filtered)
    return labels


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

    TextGrid files are searched recursively under base_dir (typically the 2_TextGrid/
    folder). The function first tries to match on the audio stem (requires
    `add_audio_keys` to have been called). If that fails, it attempts a match using
    the TextGrid filename provided in `textgrid_col`. When multiple matches exist for
    a stem, the first one encountered is used.
    """
    logger = logger or logging.getLogger("egra_eval")

    if "learner_id" not in df.columns or textgrid_col not in df.columns:
        logger.warning("DataFrame missing 'learner_id' or TextGrid filename column; REF text will be empty.")
        out = df.copy()
        out["ref_text"] = ""
        return out

    textgrid_root = Path(base_dir)
    if not textgrid_root.exists():
        logger.warning("TextGrid root %s does not exist; REF text will be empty.", textgrid_root)
        out = df.copy()
        out["ref_text"] = ""
        return out

    index: Dict[str, List[Path]] = {}
    for tg_path in sorted(textgrid_root.rglob("*.TextGrid")):
        index.setdefault(tg_path.stem.lower(), []).append(tg_path)

    logger.info("Indexed %d unique TextGrid stems from %s", len(index), textgrid_root)

    ref_texts = []
    missing = 0
    failed = 0
    matched_via_audio = 0
    matched_via_column = 0

    for _, row in df.iterrows():
        learner_id = str(row.get("learner_id", "")).strip()
        tg_name = str(row.get(textgrid_col, "")).strip()
        audio_stem = str(row.get("audio_stem", "")).strip().lower()

        tg_path: Path | None = None

        if audio_stem:
            candidates = index.get(audio_stem)
            if candidates:
                tg_path = candidates[0]
                matched_via_audio += 1

        if tg_path is None and tg_name:
            tg_stem = Path(tg_name).stem.lower()
            candidates = index.get(tg_stem)
            if candidates:
                tg_path = candidates[0]
                matched_via_column += 1
            else:
                candidate_path = textgrid_root / learner_id / tg_name
                if candidate_path.exists():
                    tg_path = candidate_path
                    matched_via_column += 1

        if not tg_path or not tg_path.exists():
            logger.warning(
                "Missing TextGrid: expected stem '%s' or file '%s' under %s",
                audio_stem,
                tg_name,
                textgrid_root,
            )
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
    logger.info(
        "Attached REF text for %d rows (missing=%d, failed=%d, matched_via_audio=%d, matched_via_column=%d).",
        len(out),
        missing,
        failed,
        matched_via_audio,
        matched_via_column,
    )
    return out
