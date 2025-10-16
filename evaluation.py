#!/usr/bin/env python3
import argparse
import logging
import os
from pathlib import Path
import pandas as pd

from egra_eval.eval.run_eval import evaluate
from egra_eval.report.summarize import macro_summary, per_learner_full, overall_full, mean_by
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
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))

    if not logger.handlers:
        logger.addHandler(handler)

    return logger


# -------------------------
# Defaults for container layout
# -------------------------
IO_ROOT = Path(os.getenv("IO_ROOT", "/io"))

DEFAULTS = {
    # Inputs
    "egra_csv":      IO_ROOT / "input" / "egradata" / "Student_Dummy_Canonical_EGRA_030925.csv",
    "meta_csv":      IO_ROOT / "input" / "egradata" / "Student_Dummy_MetaData_EGRA_030925.csv",
    "passages_csv":  IO_ROOT / "input" / "passages" / "oral_passages.csv",
    "nemo_manifest": IO_ROOT / "input" / "nemo_asr_output" / "transcriptions.jsonl",

    # TextGrid root (contains a subfolder per learner_id with TextGrids and WAVs)
    "textgrids_dir": IO_ROOT / "input" / "audio_and_texgrid",

    # Outputs
    "out_csv":       IO_ROOT / "output" / "egra_eval_detailed.csv",
    "summary_csv":   IO_ROOT / "output" / "egra_eval_summary.csv",
}


def main():
    logger = setup_logger()

    # -------------------------
    # Parse command-line args
    # -------------------------
    ap = argparse.ArgumentParser(description="EGRA evaluation pipeline")
    ap.add_argument("--egra_csv", default=str(DEFAULTS["egra_csv"]))
    ap.add_argument("--meta_csv", default=str(DEFAULTS["meta_csv"]))
    ap.add_argument("--passages_csv", default=str(DEFAULTS["passages_csv"]),
                    help="Two-column CSV for oral passages: column A=passage number, column B=text.")
    ap.add_argument("--nemo_manifest", action="append", default=[str(DEFAULTS["nemo_manifest"])],
                    help="Path(s) to NeMo JSON manifest(s). Can pass multiple times.")
    ap.add_argument("--manifest_audio_key", default="audio_filepath")
    ap.add_argument("--manifest_hyp_key", default="pred_text")
    ap.add_argument("--manifest_can_key", default=None)
    ap.add_argument("--match_on", choices=["stem", "name", "path"], default="stem",
                    help="How to join manifest HYPs to EGRA rows (default: stem).")
    ap.add_argument("--out_csv", default=str(DEFAULTS["out_csv"]))
    ap.add_argument("--summary_csv", default=str(DEFAULTS["summary_csv"]))
    args = ap.parse_args()

    # TextGrid base root (per-learner subfolders live here)
    BASE_TG_DIR = str(DEFAULTS["textgrids_dir"])

    # Ensure output folders
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)

    # Log the run configuration
    logger.info("Starting EGRA evaluation pipeline")
    logger.info(f"EGRA CSV:        {args.egra_csv}")
    logger.info(f"Meta CSV:        {args.meta_csv}")
    logger.info(f"Passages CSV:    {args.passages_csv}")
    logger.info(f"TextGrids root:  {BASE_TG_DIR}")
    logger.info(f"Manifest(s):     {args.nemo_manifest}")
    logger.info(f"Outputs -> detailed: {args.out_csv} | summary: {args.summary_csv}")

    # -------------------------
    # Load tabular inputs
    # -------------------------
    df_egra = pd.read_csv(args.egra_csv)
    df_meta = pd.read_csv(args.meta_csv)
    logger.info(f"Loaded EGRA rows: {len(df_egra)}")
    logger.info(f"Loaded META rows: {len(df_meta)}")

    # Keys for joining
    df_egra = add_audio_keys(df_egra, audio_col="audio_file")

    # -------------------------
    # Optional: attach ASR hypotheses from NeMo manifest
    # -------------------------
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

    # -------------------------
    # Attach human reference (REF) from TextGrids
    # -------------------------
    df_egra = add_refs_from_textgrid(
        df_egra,
        base_dir=BASE_TG_DIR,
        textgrid_col="textgrid",
        tier_name="child",
        logger=logger,
    )

    # -------------------------
    # Fill canonical passage texts where missing
    # -------------------------
    if args.passages_csv and Path(args.passages_csv).exists():
        logger.info(f"Attaching passage texts from {args.passages_csv}")
        df_egra = attach_passage_texts(df_egra, args.passages_csv, logger=logger)
    else:
        logger.info("No passages CSV provided or file not found; skipping passage text attach.")

    # -------------------------
    # Evaluate and write outputs
    # -------------------------


    df_results = evaluate(df_egra, df_meta)
    df_results.to_csv(args.out_csv, index=False)
    logger.info(f"Wrote detailed results -> {args.out_csv} (rows: {len(df_results)})")

    # Keep your original summary (by gender × audio_type)
    summary = macro_summary(df_results, by=["gender", "audio_type"])
    summary.to_csv(args.summary_csv, index=False)
    logger.info(f"Wrote summary -> {args.summary_csv} (groups: {len(summary)})")

    # --- New summaries (written into the same folder as out_csv) ---
    out_dir = Path(args.out_csv).resolve().parent

    # 1) Per-learner (all metrics)
    per_learner_path = out_dir / "egra_eval_summary_per_learner.csv"
    per_learner = per_learner_full(df_results)
    per_learner.to_csv(per_learner_path, index=False)
    logger.info(f"Wrote per-learner summary -> {per_learner_path} (learners: {len(per_learner)})")

    # 2) Overall (single row, all metrics)
    overall_path = out_dir / "egra_eval_summary_overall.csv"
    overall = overall_full(df_results)
    overall.to_csv(overall_path, index=False)
    logger.info(f"Wrote overall summary -> {overall_path}")

    # 3) (Optional) Per-learner × audio_type (all metrics)
    per_learner_audio_path = out_dir / "egra_eval_summary_per_learner_audio.csv"
    per_learner_audio = mean_by(df_results, by=["learner_id", "audio_type"])
    per_learner_audio.to_csv(per_learner_audio_path, index=False)
    logger.info(f"Wrote per-learner - audio_type summary -> {per_learner_audio_path} "
                f"(groups: {len(per_learner_audio)})")

    print("Detailed results:", args.out_csv)
    print("Summary (gender - audio_type):", args.summary_csv)
    print("Summary (per learner):", per_learner_path)
    print("Summary (overall):", overall_path)
    print("Summary (per learner - audio_type):", per_learner_audio_path)

if __name__ == "__main__":
    main()

