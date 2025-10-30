#!/usr/bin/env python3
import argparse
import logging
import os
from pathlib import Path

import pandas as pd

from egra_eval.eval.run_eval import evaluate
from egra_eval.report.summarize import (
    summary_for_pair,
    summary_per_speaker,
    summary_per_speaker_macro,
    summary_per_speaker_subcategory,
)
from egra_eval.data.nemo_manifest import load_many_manifests
from egra_eval.data.linking import add_audio_keys, attach_hypotheses
from egra_eval.data.textgrid_io import add_refs_from_textgrid
from egra_eval.data.passage_merge import attach_passage_texts


# -------------------------
# Logging
# -------------------------
def setup_logger() -> logging.Logger:
    logger = logging.getLogger("egra_eval")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    if not logger.handlers:
        logger.addHandler(handler)
    return logger


# -------------------------
# Defaults (container layout)
# -------------------------
IO_ROOT = Path(os.getenv("IO_ROOT", "/io"))

DEFAULTS = {
    # Inputs
    "egra_csv": IO_ROOT / "input" / "egradata" / "egra_eval_2speakers.csv",
    "meta_csv": IO_ROOT / "input" / "egradata" / "Student_Dummy_MetaData_EGRA_030925.csv",
    "passages_csv": IO_ROOT / "input" / "passages" / "oral_passages.csv",
    "nemo_manifest": IO_ROOT / "input" / "nemo_asr_output" / "transcriptions.jsonl",

    # TextGrid root
    "textgrids_dir": IO_ROOT / "input" / "audio_and_texgrid",

    # Outputs
    "out_csv": IO_ROOT / "output" / "egra_eval_detailed.csv",
    "summary_can_ref_dir": IO_ROOT / "output" / "can_ref",
    "summary_can_hyp_dir": IO_ROOT / "output" / "can_hyp",
    "summary_ref_hyp_dir": IO_ROOT / "output" / "ref_hyp",
}


def main():
    logger = setup_logger()

    # -------------------------
    # CLI
    # -------------------------
    ap = argparse.ArgumentParser(description="EGRA evaluation pipeline (simplified outputs)")
    ap.add_argument("--egra_csv", default=str(DEFAULTS["egra_csv"]))
    ap.add_argument("--meta_csv", default=str(DEFAULTS["meta_csv"]))
    ap.add_argument("--passages_csv", default=str(DEFAULTS["passages_csv"]))
    ap.add_argument(
        "--nemo_manifest",
        action="append",
        default=None,
        help="Path(s) to NeMo JSON manifest(s). Can pass multiple times.",
    )
    ap.add_argument("--manifest_audio_key", default="audio_filepath")
    ap.add_argument("--manifest_hyp_key", default="pred_text")
    ap.add_argument("--manifest_can_key", default=None)
    ap.add_argument(
        "--match_on",
        choices=["stem", "name", "path"],
        default="stem",
        help="How to join manifest HYPs to EGRA rows (default: stem).",
    )

    ap.add_argument("--out_csv", default=str(DEFAULTS["out_csv"]))
    ap.add_argument(
        "--summary_can_ref_dir",
        default=str(DEFAULTS["summary_can_ref_dir"]),
        help="Directory where CAN-REF summaries will be written.",
    )
    ap.add_argument(
        "--summary_can_hyp_dir",
        default=str(DEFAULTS["summary_can_hyp_dir"]),
        help="Directory where CAN-HYP summaries will be written.",
    )
    ap.add_argument(
        "--summary_ref_hyp_dir",
        default=str(DEFAULTS["summary_ref_hyp_dir"]),
        help="Directory where REF-HYP summaries will be written.",
    )
    args = ap.parse_args()

    BASE_TG_DIR = str(DEFAULTS["textgrids_dir"])

    # Ensure output dirs exist
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    summary_dirs = {
        "can_ref": Path(args.summary_can_ref_dir),
        "can_hyp": Path(args.summary_can_hyp_dir),
        "ref_hyp": Path(args.summary_ref_hyp_dir),
    }
    for dir_path in summary_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Log config
    # -------------------------
    logger.info("Starting EGRA evaluation pipeline (simplified outputs)")
    logger.info(f"EGRA CSV:        {args.egra_csv}")
    logger.info(f"Meta CSV:        {args.meta_csv}")
    logger.info(f"Passages CSV:    {args.passages_csv}")
    logger.info(f"TextGrids root:  {BASE_TG_DIR}")
    if args.nemo_manifest is None:
        args.nemo_manifest = [str(DEFAULTS["nemo_manifest"])]
    logger.info(f"Manifest(s):     {args.nemo_manifest}")

    # -------------------------
    # Load inputs
    # -------------------------
    df_egra = pd.read_csv(args.egra_csv)
    df_meta = pd.read_csv(args.meta_csv)
    logger.info(f"Loaded EGRA rows: {len(df_egra)}")
    logger.info(f"Loaded META rows: {len(df_meta)}")

    # Build join keys
    df_egra = add_audio_keys(df_egra, audio_col="audio_file")

    # Attach HYP from NeMo manifest(s)
    df_manifest = pd.DataFrame()
    if args.nemo_manifest:
        logger.info(f"Loading {len(args.nemo_manifest)} NeMo manifest(s)...")
        df_manifest = load_many_manifests(
            args.nemo_manifest,
            audio_key=args.manifest_audio_key,
            hyp_key=args.manifest_hyp_key,
            can_key=args.manifest_can_key,
        )
        logger.info(f"Loaded manifest rows: {len(df_manifest)}")

    df_egra = attach_hypotheses(df_egra, df_manifest, match_on=args.match_on)

    # Attach REF from TextGrids (tier="child")
    df_egra = add_refs_from_textgrid(
        df_egra,
        base_dir=BASE_TG_DIR,
        textgrid_col="textgrid",
        tier_name="child",
        logger=logger,
    )

    # Fill canonical passages if needed
    if args.passages_csv and Path(args.passages_csv).exists():
        logger.info(f"Attaching passage texts from {args.passages_csv}")
        df_egra = attach_passage_texts(df_egra, args.passages_csv, logger=logger)
    else:
        logger.info("No passages CSV provided or not found; skipping.")

    # -------------------------
    # Evaluate row-wise and write detailed CSV
    # -------------------------
    df_results = evaluate(df_egra, df_meta)

    alias_map = {
        "ACC_can_ref": "ACC_can_ref (EGRA_ACC)",
        "C_can_ref": "C_can_ref (EGRA_COR)",
        "ACC_can_hyp": "ACC_can_hyp (ASR_EGRA_ACC)",
        "C_can_hyp": "C_can_hyp (ASR_EGRA_COR)",
    }
    df_results_for_csv = df_results.rename(columns=alias_map)
    df_results_for_csv.to_csv(args.out_csv, index=False)
    logger.info(f"Wrote detailed results -> {args.out_csv} (rows: {len(df_results_for_csv)})")

    # -------------------------
    # Summaries (micro-averages via totals)
    # -------------------------
    summary_logs = []
    for pair, dir_path in summary_dirs.items():
        df_per_speaker = summary_per_speaker(df_results, pair)
        df_global = summary_for_pair(df_results, pair, by=None)
        df_global.insert(0, "learner_id", "__GLOBAL__")

        if df_per_speaker.empty:
            df_per_speaker_with_global = df_global
        else:
            df_global_aligned = df_global[df_per_speaker.columns]
            df_per_speaker_with_global = pd.concat(
                [df_per_speaker, df_global_aligned],
                ignore_index=True,
            )

        per_speaker_path = dir_path / "egra_eval_summary_per_speaker_global.csv"
        df_per_speaker_with_global.to_csv(per_speaker_path, index=False)
        logger.info(f"[{pair}] Wrote per-speaker summary -> {per_speaker_path}")
        summary_logs.append(per_speaker_path)

        df_macro = summary_per_speaker_macro(df_results, pair)
        macro_path = dir_path / "egra_eval_summary_per_speaker_macro.csv"
        df_macro.to_csv(macro_path, index=False)
        logger.info(f"[{pair}] Wrote per-speaker × macro-category summary -> {macro_path}")
        summary_logs.append(macro_path)

        df_subcat = summary_per_speaker_subcategory(df_results, pair)
        subcat_path = dir_path / "egra_eval_summary_per_speaker_subcat.csv"
        df_subcat.to_csv(subcat_path, index=False)
        logger.info(f"[{pair}] Wrote per-speaker × subcategory summary -> {subcat_path}")
        summary_logs.append(subcat_path)

    # Console pointers
    print("Detailed results:", args.out_csv)
    print("Summary directories/files:")
    for path in summary_logs:
        print("  ", path)


if __name__ == "__main__":
    main()
