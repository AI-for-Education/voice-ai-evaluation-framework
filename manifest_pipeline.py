#!/usr/bin/env python3
"""Build and clean evaluation manifest (no scoring)."""

from __future__ import annotations

import argparse
import logging
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import soundfile as sf

from egra_eval2.dataset_layout import DatasetLayoutError, resolve_dataset_paths
from egra_eval2.linking import add_audio_keys, attach_hypotheses
from egra_eval2.nemo_manifest import load_many_manifests
from egra_eval2.passage_merge import attach_passage_texts
from egra_eval2.textgrid_io import add_refs_from_textgrid
from egra_eval2.manifest_builder import (
    build_reference_manifest_dataframe,
    write_manifest_jsonl,
)
from egra_eval2.manifest_cleaner import clean_manifest_jsonl
from egra_eval2.eval_utils import adjust_letter_canonical_text
from egra_eval2.manifest_integrity import ReferenceIntegrityError, validate_reference_rows
from egra_eval2.reference.views import ReferenceViewError, prepare_ipa_reference_view


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("egra_eval_manifest")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    return logger


def _default_evaluation_root(asr_manifests: list[str] | None) -> Path:
    """Reuse the model/timestamp directory created by standard inference output."""
    for manifest in asr_manifests or []:
        path = Path(manifest)
        run_dir = path.parent
        if path.name == "transcriptions.jsonl" and run_dir.parent.name == "transcripts":
            return run_dir.parent.parent / "evaluations" / run_dir.name

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    return (
        Path("input_output_data")
        / "output"
        / "evaluations"
        / f"evaluation_{timestamp}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build and clean EGRA manifest.")
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--output_root", default=None)
    p.add_argument("--egra_csv", default=None)
    p.add_argument("--passages_csv", default=None)
    p.add_argument(
        "--audio_manifest",
        "--manifest_base_in",
        dest="manifest_base_in",
        default=None,
        help=(
            "Audio manifest whose rows define the evaluation inputs; normally the "
            "same segmented manifest supplied to inference. "
            "--manifest_base_in remains a backward-compatible alias."
        ),
    )
    p.add_argument(
        "--prediction_manifest",
        "--asr_manifest",
        "--nemo_manifest",
        dest="asr_manifest",
        action="append",
        default=None,
        help=(
            "Prediction transcriptions.jsonl to attach; repeat for multiple manifests. "
            "--asr_manifest and --nemo_manifest remain backward-compatible aliases."
        ),
    )
    p.add_argument("--manifest_audio_key", default="audio_filepath")
    p.add_argument("--manifest_hyp_key", default="pred_text")
    p.add_argument("--manifest_can_key", default=None)
    p.add_argument("--match_on", choices=["stem", "name", "path"], default="stem")
    p.add_argument("--manifest_raw_out", default=None)
    p.add_argument("--manifest_clean_out", default=None)
    p.add_argument(
        "--manifest_clean_drop_empty",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop rows where cleaned ref_text becomes empty (default: true). Use --no-manifest_clean_drop_empty to keep them.",
    )
    p.add_argument("--manifest_path_prefix", default=None)
    return p.parse_args()


def _load_json_objects(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    objects: list[dict] = []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            objects = [parsed]
        elif isinstance(parsed, list):
            objects = [obj for obj in parsed if isinstance(obj, dict)]
        return objects
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    idx = 0
    n = len(text)
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        try:
            obj, next_idx = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            idx += 1
            continue
        if isinstance(obj, dict):
            objects.append(obj)
        idx = next_idx
    return objects


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


def _safe_hyp_text(value: object) -> str:
    """Normalize missing values without rewriting literal model output such as 'nan'."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def _attach_authoritative_hypotheses(
    raw_df: pd.DataFrame,
    asr_df: pd.DataFrame,
    *,
    match_on: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Attach a supplied ASR run without retaining hypotheses from the base."""
    base_df = raw_df.rename(columns={"audio_filepath": "audio_file"}).copy()
    base_df = add_audio_keys(base_df, audio_col="audio_file")
    # attach_hypotheses fills blank values by design. Blank the base explicitly so
    # the newly supplied ASR manifest is authoritative, including missing rows.
    base_df["hyp_text"] = ""
    merged = attach_hypotheses(base_df, asr_df, match_on=match_on, logger=logger)
    updated = raw_df.copy()
    updated["pred_text"] = merged.get("hyp_text", "").apply(_safe_hyp_text)
    return updated


def _validate_reference_dataframe(df: pd.DataFrame, *, source: str) -> None:
    validate_reference_rows(
        df.to_dict(orient="records"),
        text_fields=("ref_text", "can_text"),
        audio_field="audio_filepath",
        source=source,
    )


def _compute_duration_if_missing(audio_path: str, duration_value: object) -> float:
    try:
        if duration_value is not None and not pd.isna(duration_value):
            return float(duration_value)
    except Exception:
        pass
    try:
        with sf.SoundFile(audio_path) as f:
            return float(len(f) / f.samplerate)
    except Exception:
        return 0.0


def load_base_manifest_dataframe(path: str, logger: logging.Logger) -> pd.DataFrame:
    in_path = Path(path)
    if not in_path.exists():
        raise SystemExit(f"--audio_manifest file not found: {in_path}")
    objs = _load_json_objects(in_path)
    rows = []
    for obj in objs:
        audio = _safe_text(obj.get("audio_filepath", ""))
        if not audio:
            continue
        rows.append(
            {
                "audio_filepath": audio,
                "duration": _compute_duration_if_missing(audio, obj.get("duration")),
                "pred_text": _safe_text(obj.get("pred_text", "")),
                "ref_text": _safe_text(obj.get("ref_text", "")),
                "can_text": _safe_text(obj.get("can_text", "")),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        logger.info("Loaded base manifest rows: %d from %s", len(out), in_path)
        return out

    before = len(out)
    out = out.drop_duplicates(subset=["audio_filepath"], keep="first")
    removed = before - len(out)
    if removed:
        logger.warning(
            "Dropped %d duplicate row(s) by audio_filepath from base manifest: %s",
            removed,
            in_path,
        )
    logger.info("Loaded base manifest rows: %d from %s", len(out), in_path)
    return out


def main() -> None:
    logger = setup_logger()
    args = parse_args()

    try:
        layout = resolve_dataset_paths(args.dataset_root)
    except DatasetLayoutError as exc:
        raise SystemExit(str(exc)) from exc

    try:
        prepare_ipa_reference_view(
            dataset_root=layout.root,
            manifest_base_in=args.manifest_base_in,
            asr_manifests=args.asr_manifest,
            logger=logger,
        )
    except ReferenceViewError as exc:
        raise SystemExit(str(exc)) from exc

    if not args.output_root:
        args.output_root = str(_default_evaluation_root(args.asr_manifest))
        logger.info("No --output_root supplied; using default: %s", args.output_root)

    args.egra_csv = args.egra_csv or str(layout.canonical_csv)
    manifests_dir = Path(args.output_root) / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    raw_manifest_path = (
        Path(args.manifest_raw_out)
        if args.manifest_raw_out
        else manifests_dir / "ref_manifest.raw.jsonl"
    )
    clean_manifest_path = (
        Path(args.manifest_clean_out)
        if args.manifest_clean_out
        else manifests_dir / "ref_manifest.clean.jsonl"
    )

    if args.manifest_base_in:
        logger.info("Stage A: loading base manifest (preserve granularity) -> %s", args.manifest_base_in)
        raw_df = load_base_manifest_dataframe(args.manifest_base_in, logger)

        if args.asr_manifest:
            logger.info("Loading %d prediction manifest(s) to attach pred_text...", len(args.asr_manifest))
            asr_df = load_many_manifests(
                args.asr_manifest,
                audio_key=args.manifest_audio_key,
                hyp_key=args.manifest_hyp_key,
                can_key=args.manifest_can_key,
                logger=logger,
            )
            raw_df = _attach_authoritative_hypotheses(
                raw_df,
                asr_df,
                match_on=args.match_on,
                logger=logger,
            )
        else:
            logger.info("No --prediction_manifest supplied; keeping pred_text from audio manifest.")
    else:
        if not args.passages_csv:
            raise SystemExit("--passages_csv is required when --audio_manifest is not provided.")
        passages_path = Path(args.passages_csv)
        if not passages_path.exists():
            raise SystemExit(f"--passages_csv file not found: {passages_path}")

        df_egra = pd.read_csv(args.egra_csv)
        logger.info("Loaded EGRA rows: %d", len(df_egra))
        df_egra = adjust_letter_canonical_text(df_egra, logger)
        df_egra = add_audio_keys(df_egra, audio_col="audio_file")

        if args.asr_manifest:
            logger.info("Loading %d prediction manifest(s)...", len(args.asr_manifest))
            asr_df = load_many_manifests(
                args.asr_manifest,
                audio_key=args.manifest_audio_key,
                hyp_key=args.manifest_hyp_key,
                can_key=args.manifest_can_key,
                logger=logger,
            )
            df_egra = attach_hypotheses(df_egra, asr_df, match_on=args.match_on, logger=logger)
        else:
            logger.info("No --prediction_manifest supplied; pred_text will be empty in output manifest.")
            df_egra["hyp_text"] = ""

        df_egra = add_refs_from_textgrid(
            df_egra,
            base_dir=str(layout.textgrid_root),
            textgrid_col="textgrid",
            tier_name="child",
            logger=logger,
        )
        df_egra = attach_passage_texts(df_egra, args.passages_csv, logger=logger)

        logger.info("Stage A: building reference manifest -> %s", raw_manifest_path)
        raw_df = build_reference_manifest_dataframe(
            df_egra,
            dataset_root=layout.root,
            audio_root=layout.audio_root,
            path_prefix=args.manifest_path_prefix,
            logger=logger,
        )
    try:
        _validate_reference_dataframe(raw_df, source=str(raw_manifest_path))
    except ReferenceIntegrityError as exc:
        raise SystemExit(str(exc)) from exc
    write_manifest_jsonl(raw_df, raw_manifest_path)

    logger.info(
        "Stage B: cleaning manifest (train_aggressive, drop_empty=%s) -> %s",
        args.manifest_clean_drop_empty,
        clean_manifest_path,
    )
    stats = clean_manifest_jsonl(
        input_path=raw_manifest_path,
        output_path=clean_manifest_path,
        drop_empty_text=args.manifest_clean_drop_empty,
        text_input_key="ref_text",
        text_output_key="ref_text",
    )
    logger.info(
        "Manifest cleaning stats: total=%d, bad_json=%d, written=%d, dropped_empty=%d",
        stats["total_lines"],
        stats["bad_json_skipped"],
        stats["written_lines"],
        stats["dropped_empty_text"],
    )

    print("Raw manifest:", raw_manifest_path)
    print("Clean manifest:", clean_manifest_path)


if __name__ == "__main__":
    main()
