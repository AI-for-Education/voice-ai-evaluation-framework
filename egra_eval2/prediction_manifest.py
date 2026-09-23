"""Load one or more inference transcript manifests for evaluation."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd


def _open_text(path: str):
    source = Path(path)
    if source.suffix == ".gz":
        return gzip.open(source, "rt", encoding="utf-8")
    return source.open("r", encoding="utf-8")


def load_prediction_manifest(
    manifest_path: str,
    audio_key: str = "audio_filepath",
    hyp_key: str = "pred_text",
    can_key: str | None = None,
    logger=None,
) -> pd.DataFrame:
    """Load a line-delimited transcript manifest into evaluation columns."""
    rows: list[dict[str, Any]] = []
    total = 0
    bad = 0

    try:
        with _open_text(manifest_path) as stream:
            for line in stream:
                total += 1
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue

                audio = str(item.get(audio_key, "") or "").strip()
                if not audio:
                    bad += 1
                    continue
                name = Path(audio).name
                rows.append(
                    {
                        "audio_path": audio,
                        "audio_name": name,
                        "audio_stem": Path(audio).stem,
                        "hyp_text": item.get(hyp_key, ""),
                        "can_from_manifest": (
                            item.get(can_key, "") if can_key else None
                        ),
                    }
                )
    except FileNotFoundError:
        if logger:
            logger.error("Manifest not found: %s", manifest_path)
        return pd.DataFrame(
            columns=[
                "audio_path",
                "audio_name",
                "audio_stem",
                "hyp_text",
                "can_from_manifest",
            ]
        )

    frame = pd.DataFrame(rows)
    if logger:
        logger.info(
            "Loaded manifest %s | rows=%d | total_lines=%d | skipped=%d",
            manifest_path,
            len(frame),
            total,
            bad,
        )
    return frame


def load_prediction_manifests(
    manifests: Iterable[str],
    audio_key: str = "audio_filepath",
    hyp_key: str = "pred_text",
    can_key: str | None = None,
    logger=None,
) -> pd.DataFrame:
    """Load and combine transcript manifests, keeping the last duplicate."""
    frames: list[pd.DataFrame] = []
    for manifest in manifests:
        frame = load_prediction_manifest(
            manifest,
            audio_key=audio_key,
            hyp_key=hyp_key,
            can_key=can_key,
            logger=logger,
        )
        if not frame.empty:
            frames.append(frame)
    if not frames:
        if logger:
            logger.warning("No prediction manifests produced any rows.")
        return pd.DataFrame(
            columns=[
                "audio_path",
                "audio_name",
                "audio_stem",
                "hyp_text",
                "can_from_manifest",
            ]
        )

    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(
        subset=["audio_stem", "audio_name"], keep="last"
    )
    if logger and len(combined) != before:
        logger.info("Deduplicated manifests: %d -> %d rows", before, len(combined))
    return combined
