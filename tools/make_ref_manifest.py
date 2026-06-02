#!/usr/bin/env python3
"""Build a NeMo-style reference manifest (audio_filepath, duration, ref_text, can_text)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from egra_eval2.dataset_layout import DatasetLayoutError, resolve_dataset_paths
from egra_eval2.linking import add_audio_keys
from egra_eval2.passage_merge import attach_passage_texts
from egra_eval2.textgrid_io import add_refs_from_textgrid
from egra_eval2.manifest_builder import (
    build_reference_manifest_dataframe,
    write_manifest_jsonl,
)
from egra_eval2.eval_utils import adjust_letter_canonical_text


LOGGER = logging.getLogger("make_ref_manifest")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build reference manifest from dataset CSV + TextGrid.")
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--passages_csv", required=True)
    p.add_argument("--dataset_annotator", default=None)
    p.add_argument("--tier_name", default="child")
    p.add_argument("--path_prefix", default=None)
    return p.parse_args()


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

    df = pd.read_csv(layout.canonical_csv)
    df = adjust_letter_canonical_text(df, LOGGER)
    df = add_audio_keys(df, audio_col="audio_file")
    df = add_refs_from_textgrid(
        df,
        base_dir=str(layout.textgrid_root),
        textgrid_col="textgrid",
        tier_name=args.tier_name,
        logger=LOGGER,
    )
    df = attach_passage_texts(df, args.passages_csv, logger=LOGGER)

    manifest_df = build_reference_manifest_dataframe(
        df,
        dataset_root=layout.root,
        audio_root=layout.audio_root,
        path_prefix=args.path_prefix,
        logger=LOGGER,
    )
    write_manifest_jsonl(manifest_df, args.output_jsonl)
    LOGGER.info("Done. Wrote manifest: %s (rows=%d)", args.output_jsonl, len(manifest_df))


if __name__ == "__main__":
    main()

