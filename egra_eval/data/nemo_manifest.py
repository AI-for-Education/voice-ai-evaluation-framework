# egra_eval/data/nemo_manifest.py
from __future__ import annotations
from pathlib import Path
import json
import gzip
from typing import Iterable, Dict, Any, Optional, List
import pandas as pd

def _open_text(path: str):
    p = Path(path)
    if p.suffix == ".gz":
        return gzip.open(p, "rt", encoding="utf-8")
    return p.open("r", encoding="utf-8")

def load_nemo_manifest(
    manifest_path: str,
    audio_key: str = "audio_filepath",
    hyp_key: str = "pred_text",
    can_key: str | None = None,
    logger=None,
) -> pd.DataFrame:
    """
    Reads a line-delimited NeMo manifest and returns a DataFrame with:
      - audio_path (full path string from manifest)
      - audio_name (filename)
      - audio_stem (filename without extension)
      - hyp_text (predicted transcription)
      - can_from_manifest (optional canonical text if present)
    Skips blank lines and lines missing the audio key.
    """
    rows: List[Dict[str, Any]] = []
    total = 0
    bad = 0

    try:
        with _open_text(manifest_path) as f:
            for line in f:
                total += 1
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    bad += 1
                    continue

                audio = str(obj.get(audio_key, "") or "").strip()
                if not audio:
                    bad += 1
                    continue

                hyp = obj.get(hyp_key, "")
                can = obj.get(can_key, "") if can_key else None

                name = Path(audio).name
                stem = Path(audio).stem

                rows.append(
                    {
                        "audio_path": audio,
                        "audio_name": name,
                        "audio_stem": stem,
                        "hyp_text": hyp,
                        "can_from_manifest": can,
                    }
                )
    except FileNotFoundError:
        if logger:
            logger.error(f"Manifest not found: {manifest_path}")
        return pd.DataFrame(
            columns=["audio_path", "audio_name", "audio_stem", "hyp_text", "can_from_manifest"]
        )

    df = pd.DataFrame(rows)
    if logger:
        logger.info(
            f"Loaded manifest {manifest_path} | rows={len(df)} | total_lines={total} | skipped={bad}"
        )
    return df

def load_many_manifests(
    manifests: Iterable[str],
    audio_key: str = "audio_filepath",
    hyp_key: str = "pred_text",
    can_key: str | None = None,
    logger=None,
) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    for m in manifests:
        df = load_nemo_manifest(m, audio_key=audio_key, hyp_key=hyp_key, can_key=can_key, logger=logger)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        if logger:
            logger.warning("No manifests produced any rows.")
        return pd.DataFrame(columns=["audio_path","audio_name","audio_stem","hyp_text","can_from_manifest"])

    df = pd.concat(dfs, ignore_index=True)

    # Drop duplicates by both stem and name (keep last, assuming newest wins)
    before = len(df)
    df = df.drop_duplicates(subset=["audio_stem", "audio_name"], keep="last")
    if logger and len(df) != before:
        logger.info(f"Deduplicated manifests: {before} -> {len(df)} rows")

    return df

