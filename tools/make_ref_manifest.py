#!/usr/bin/env python3
"""
Utility script that builds a NeMo-style JSONL manifest containing:
  - audio_filepath
  - duration
  - pred_text (empty placeholder)
  - ref_text  (human REF from TextGrid)
  - can_text  (canonical text, including passages)

"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from egra_eval.data.dataset_layout import DatasetLayoutError, resolve_dataset_paths
from egra_eval.data.linking import add_audio_keys
from egra_eval.data.passage_merge import attach_passage_texts
from egra_eval.data.textgrid_io import add_refs_from_textgrid
from evaluation import adjust_letter_canonical_text


LOGGER = logging.getLogger("make_ref_manifest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a NeMo JSONL manifest with canonical and reference texts.")
    parser.add_argument("--dataset_root", required=True, help="Dataset root containing 0_Audio/, 2_TextGrid/, Student_* CSVs.")
    parser.add_argument("--output_jsonl", required=True, help="Destination JSONL file.")
    parser.add_argument("--passages_csv", required=True, help="CSV mapping passage numbers to canonical text (oral_passages.csv).")
    parser.add_argument("--dataset_annotator", default=None, help="Optional annotator folder under 2_TextGrid/ to restrict TextGrid search.")
    parser.add_argument("--tier_name", default="child", help="TextGrid tier that holds the child transcription.")
    parser.add_argument(
        "--path_prefix",
        default=None,
        help="Optional path prefix to replace the dataset root (e.g., /io/input/<dataset>).",
    )
    return parser.parse_args()


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
    raise FileNotFoundError(f"Could not locate audio file for row: learner_id={learner}, audio_file={raw}")


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


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    try:
        layout = resolve_dataset_paths(args.dataset_root, annotator=args.dataset_annotator)
    except DatasetLayoutError as exc:
        raise SystemExit(str(exc)) from exc

    passages_path = Path(args.passages_csv)
    if not passages_path.exists():
        raise SystemExit(f"--passages_csv file not found: {passages_path}")

    LOGGER.info("Loading EGRA canonical CSV: %s", layout.canonical_csv)
    df = pd.read_csv(layout.canonical_csv)
    df = attach_passage_texts(df, args.passages_csv, logger=LOGGER)
    df = adjust_letter_canonical_text(df, LOGGER)
    df = add_audio_keys(df, audio_col="audio_file")
    df = add_refs_from_textgrid(
        df,
        base_dir=str(layout.textgrid_root),
        textgrid_col="textgrid",
        tier_name=args.tier_name,
        logger=LOGGER,
    )

    dataset_root_path = layout.root
    audio_root = layout.audio_root

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Writing manifest: %s", out_path)
    with out_path.open("w", encoding="utf-8") as out_f:
        for _, row in df.iterrows():
            try:
                audio_path = resolve_audio_path(row, audio_root)
                duration = compute_duration(audio_path)
            except FileNotFoundError as err:
                LOGGER.warning(str(err))
                continue

            record = {
                "audio_filepath": format_output_path(audio_path, dataset_root_path, args.path_prefix),
                "duration": duration,
                "pred_text": "",
                "ref_text": str(row.get("ref_text", "") or ""),
                "can_text": str(row.get("canonical_text", "") or ""),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    LOGGER.info("Done. Wrote %d rows.", len(df))


if __name__ == "__main__":
    main()
