#!/usr/bin/env python3
"""Command-line entry point for running the simplified EGRA evaluation pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from egra_eval.data.dataset_layout import DatasetLayoutError, resolve_dataset_paths
from egra_eval.data.linking import add_audio_keys, attach_hypotheses
from egra_eval.data.nemo_manifest import load_many_manifests
from egra_eval.data.passage_merge import attach_passage_texts
from egra_eval.data.textgrid_io import add_refs_from_textgrid
from egra_eval.eval.run_eval import evaluate
from egra_eval.report.summarize import (
    summary_for_pair,
    summary_per_speaker,
    summary_per_speaker_macro,
    summary_per_speaker_subcategory,
)

# ---------------------------------------------------------------------------
# Logging / CLI
# ---------------------------------------------------------------------------

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("egra_eval")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
    return logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run EGRA evaluation using canonical/ASR data.")
    p.add_argument("--dataset_root", required=True, help="Root folder containing 0_Audio/2_TextGrid and Student_* CSVs.")
    p.add_argument("--dataset_annotator", default=None, help="Annotator name under 2_TextGrid to use (defaults to first alphabetically).")
    p.add_argument("--output_root", required=True, help="Directory where outputs will be written (detailed CSV + summary subfolders).")

    p.add_argument("--egra_csv", default=None)
    p.add_argument("--meta_csv", default=None)
    p.add_argument("--passages_csv", default=None)
    p.add_argument("--nemo_manifest", action="append", default=None, help="Path(s) to NeMo JSON manifests with ASR hypotheses (can be supplied multiple times).")
    p.add_argument("--manifest_audio_key", default="audio_filepath")
    p.add_argument("--manifest_hyp_key", default="pred_text")
    p.add_argument("--manifest_can_key", default=None)
    p.add_argument("--match_on", choices=["stem", "name", "path"], default="stem", help="How to join manifest HYPs to EGRA rows.")

    p.add_argument("--out_csv", default=None)
    p.add_argument("--summary_can_ref_dir", default=None)
    p.add_argument("--summary_can_hyp_dir", default=None)
    p.add_argument("--summary_ref_hyp_dir", default=None)
    return p.parse_args()

# ---------------------------------------------------------------------------
# IO resolution / outputs
# ---------------------------------------------------------------------------

def resolve_inputs(args: argparse.Namespace, logger: logging.Logger) -> tuple[Path, Dict[str, Path]]:
    """Resolve dataset layout and output dirs; mutate args with defaults."""
    try:
        layout = resolve_dataset_paths(args.dataset_root, annotator=args.dataset_annotator)
    except DatasetLayoutError as exc:
        raise SystemExit(str(exc)) from exc

    # Fill inferred CSVs
    args.egra_csv = args.egra_csv or str(layout.canonical_csv)
    args.meta_csv = args.meta_csv or str(layout.metadata_csv)
    args.passages_csv = args.passages_csv or ""

    # Resolve manifest(s)
    if not args.nemo_manifest:
        candidate = layout.root / "nemo_asr_output" / "transcriptions.jsonl"
        if candidate.exists():
            args.nemo_manifest = [str(candidate)]
        else:
            raise SystemExit(
                "No --nemo_manifest supplied and no default manifest found at "
                f"{candidate}. Run inference first or provide the path explicitly."
            )

    logger.info(
        "Dataset root resolved: audio=%s | textgrids=%s (annotator=%s) | canonical=%s | metadata=%s",
        layout.audio_root, layout.textgrid_root, layout.annotator, layout.canonical_csv, layout.metadata_csv,
    )

    # Prepare outputs
    base = Path(args.output_root)
    base.mkdir(parents=True, exist_ok=True)

    args.out_csv = args.out_csv or str(base / "egra_eval_detailed.csv")
    args.summary_can_ref_dir = args.summary_can_ref_dir or str(base / "can_ref")
    args.summary_can_hyp_dir = args.summary_can_hyp_dir or str(base / "can_hyp")
    args.summary_ref_hyp_dir = args.summary_ref_hyp_dir or str(base / "ref_hyp")

    summary_dirs = {
        "can_ref": Path(args.summary_can_ref_dir),
        "can_hyp": Path(args.summary_can_hyp_dir),
        "ref_hyp": Path(args.summary_ref_hyp_dir),
    }
    return layout.textgrid_root, summary_dirs


def ensure_output_dirs(out_csv: str, summary_dirs: Dict[str, Path]) -> None:
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    for d in summary_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data loading / writing
# ---------------------------------------------------------------------------

def load_manifest(args: argparse.Namespace, logger: logging.Logger) -> pd.DataFrame:
    if not args.nemo_manifest:
        return pd.DataFrame()
    logger.info("Loading %d NeMo manifest(s)...", len(args.nemo_manifest))
    df = load_many_manifests(
        args.nemo_manifest,
        audio_key=args.manifest_audio_key,
        hyp_key=args.manifest_hyp_key,
        can_key=args.manifest_can_key,
    )
    logger.info("Loaded manifest rows: %d", len(df))
    return df


def write_detailed_csv(df: pd.DataFrame, path: str, logger: logging.Logger) -> None:
    df.rename(columns={
        "ACC_can_ref": "ACC_can_ref (EGRA_ACC)",
        "C_can_ref": "C_can_ref (EGRA_COR)",
        "ACC_can_hyp": "ACC_can_hyp (ASR_EGRA_ACC)",
        "C_can_hyp": "C_can_hyp (ASR_EGRA_COR)",
    }).to_csv(path, index=False)
    logger.info("Wrote detailed results -> %s (rows: %d)", path, len(df))


def write_summary_csvs(df: pd.DataFrame, summary_dirs: Dict[str, Path], logger: logging.Logger) -> List[Path]:
    written: List[Path] = []

    def _write_one(pair: str, directory: Path) -> None:
        per_speaker = summary_per_speaker(df, pair)
        global_row = summary_for_pair(df, pair, by=None)
        global_row.insert(0, "learner_id", "__GLOBAL__")

        combined = global_row if per_speaker.empty else pd.concat(
            [per_speaker, global_row[per_speaker.columns]], ignore_index=True
        )
        per_speaker_path = directory / "egra_eval_summary_per_speaker_global.csv"
        combined.to_csv(per_speaker_path, index=False)
        logger.info("[%s] Wrote per-speaker summary -> %s", pair, per_speaker_path)
        written.append(per_speaker_path)

        macro_path = directory / "egra_eval_summary_per_speaker_macro.csv"
        summary_per_speaker_macro(df, pair).to_csv(macro_path, index=False)
        logger.info("[%s] Wrote per-speaker × macro-category summary -> %s", pair, macro_path)
        written.append(macro_path)

        subcat_path = directory / "egra_eval_summary_per_speaker_subcat.csv"
        summary_per_speaker_subcategory(df, pair).to_csv(subcat_path, index=False)
        logger.info("[%s] Wrote per-speaker × subcategory summary -> %s", pair, subcat_path)
        written.append(subcat_path)

    for pair, directory in summary_dirs.items():
        _write_one(pair, directory)

    return written

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger = setup_logger()
    args = parse_args()

    textgrid_root, summary_dirs = resolve_inputs(args, logger)
    ensure_output_dirs(args.out_csv, summary_dirs)

    logger.info("Starting EGRA evaluation pipeline")
    logger.info(
        "EGRA CSV: %s | META CSV: %s | Passages CSV: %s | TextGrid root: %s | Manifest(s): %s",
        args.egra_csv, args.meta_csv, args.passages_csv, textgrid_root, args.nemo_manifest,
    )

    df_egra = pd.read_csv(args.egra_csv)
    df_meta = pd.read_csv(args.meta_csv)
    logger.info("Loaded EGRA rows: %d | META rows: %d", len(df_egra), len(df_meta))

    df_egra = add_audio_keys(df_egra, audio_col="audio_file")
    manifest_df = load_manifest(args, logger)
    df_egra = attach_hypotheses(df_egra, manifest_df, match_on=args.match_on)
    df_egra = add_refs_from_textgrid(
        df_egra,
        base_dir=str(textgrid_root),
        textgrid_col="textgrid",
        tier_name="child",
        logger=logger,
    )

    if args.passages_csv and Path(args.passages_csv).exists():
        logger.info("Attaching passage texts from %s", args.passages_csv)
        df_egra = attach_passage_texts(df_egra, args.passages_csv, logger=logger)
    else:
        logger.info("No passages CSV provided or file not found; skipping canonical pass-through.")

    df_results = evaluate(df_egra, df_meta)
    write_detailed_csv(df_results, args.out_csv, logger)
    summary_paths = write_summary_csvs(df_results, summary_dirs, logger)

    print("Detailed results:", args.out_csv)
    print("Summary directories/files:")
    for path in summary_paths:
        print("  ", path)


if __name__ == "__main__":
    main()
