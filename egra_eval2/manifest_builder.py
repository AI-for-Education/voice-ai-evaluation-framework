from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import soundfile as sf


def resolve_audio_path(row: pd.Series, audio_root: Path) -> Path:
    candidates: list[Path] = []
    raw = Path(str(row["audio_file"]))
    learner = str(row.get("learner_id", "")).strip()

    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(audio_root / raw)
        if learner:
            candidates.append(audio_root / learner / raw.name)

    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"Could not locate audio file for row: learner_id={learner}, audio_file={raw}"
    )


def format_output_path(actual_path: Path, dataset_root: Path, prefix: Optional[str]) -> str:
    if prefix:
        try:
            rel = actual_path.relative_to(dataset_root)
            return str(Path(prefix) / rel)
        except ValueError:
            pass
    return str(actual_path)


def compute_duration(path: Path) -> float:
    with sf.SoundFile(str(path)) as f:
        return float(len(f) / f.samplerate)


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    s = str(value)
    return "" if s.lower() == "nan" else s


def build_reference_manifest_dataframe(
    df: pd.DataFrame,
    *,
    dataset_root: Path,
    audio_root: Path,
    path_prefix: Optional[str] = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Build a NeMo-style manifest DataFrame from EGRA rows.

    Required input columns:
      - audio_file
      - canonical_text
      - ref_text
    """
    logger = logger or logging.getLogger("egra_eval")

    records = []
    missing_audio = 0
    for _, row in df.iterrows():
        try:
            audio_path = resolve_audio_path(row, audio_root)
        except FileNotFoundError as exc:
            logger.warning("%s", exc)
            missing_audio += 1
            continue

        record = {
            "audio_filepath": format_output_path(audio_path, dataset_root, path_prefix),
            "duration": compute_duration(audio_path),
            "pred_text": _safe_text(row.get("hyp_text", "")),
            "ref_text": _safe_text(row.get("ref_text", "")),
            "can_text": _safe_text(row.get("canonical_text", "")),
        }
        records.append(record)

    out = pd.DataFrame(records)
    if out.empty:
        logger.info(
            "Built reference manifest DataFrame: rows=%d (missing_audio=%d).",
            len(out),
            missing_audio,
        )
        return out

    # Defensive deduplication: the input canonical CSV can contain repeated rows
    # (same audio file repeated twice). Keep one row per audio file to prevent
    # duplicate downstream segments/results.
    before_exact = len(out)
    out = out.drop_duplicates(keep="first")
    removed_exact = before_exact - len(out)
    if removed_exact:
        logger.warning(
            "Removed %d exact duplicate manifest row(s) before audio_filepath deduplication.",
            removed_exact,
        )

    before_audio = len(out)
    out = out.drop_duplicates(subset=["audio_filepath"], keep="first")
    removed_audio = before_audio - len(out)
    if removed_audio:
        logger.warning(
            "Removed %d duplicate row(s) by audio_filepath while building reference manifest.",
            removed_audio,
        )

    logger.info(
        "Built reference manifest DataFrame: rows=%d (missing_audio=%d).",
        len(out),
        missing_audio,
    )
    return out


def write_manifest_jsonl(df_manifest: pd.DataFrame, output_path: str | Path) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for _, row in df_manifest.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
