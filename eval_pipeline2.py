#!/usr/bin/env python3
"""Run EGRA evaluation from a prebuilt cleaned manifest."""

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from egra_eval2.dataset_layout import DatasetLayoutError, resolve_dataset_paths
from egra_eval2.eval_utils import adjust_letter_canonical_text
from egra_eval2.evaluate import aggregate_row_scores, evaluate_rows
from egra_eval2.manifest_integrity import ReferenceIntegrityError, validate_reference_rows
from egra_eval2.scoring_text import ScoringRepresentationError, prepare_scoring_texts
from egra_eval2.reference.views import METADATA_NAME, default_output_dir
from inference.pipeline_provenance import (
    PIPELINE_PROVENANCE_SCHEMA_VERSION,
    build_evaluation_provenance,
)


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("egra_eval")
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


def _default_evaluation_root(manifest_in: str) -> Path:
    """Keep scoring in the evaluation run directory that owns the manifest."""
    manifest_path = Path(manifest_in)
    if manifest_path.parent.name == "manifests":
        run_dir = manifest_path.parent.parent
        if run_dir.parent.name == "evaluations":
            return run_dir

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    return (
        Path("input_output_data")
        / "output"
        / "evaluations"
        / f"evaluation_{timestamp}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run EGRA evaluation from cleaned manifest."
    )
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--manifest_in", required=True, help="Cleaned manifest JSONL.")
    p.add_argument("--output_root", default=None)
    p.add_argument("--egra_csv", default=None)
    p.add_argument("--meta_csv", default=None)
    p.add_argument("--out_csv", default=None)
    p.add_argument("--manifest_audio_key", default="audio_filepath")
    p.add_argument("--manifest_ref_key", default="ref_text")
    p.add_argument("--manifest_can_key", default="can_text")
    p.add_argument("--manifest_hyp_key", default="pred_text")
    p.add_argument(
        "--scoring_representation",
        choices=["auto", "orthographic", "ipa"],
        default="auto",
        help=(
            "auto selects the model-native valid view; orthographic and ipa select "
            "an explicit valid view"
        ),
    )
    p.add_argument("--detailed", action="store_true", default=False)
    return p.parse_args()


def _output_namespace(
    scoring_representation: str,
    scoring_units: str | None,
) -> str:
    if scoring_units == "phoneme" or scoring_representation == "ipa":
        return "ipa"
    if scoring_units == "orthographic" or scoring_representation == "orthographic":
        return "orthographic"
    raise ValueError(
        "scoring_units is required to resolve an automatic evaluation output"
    )


def _representation_scoped_path(
    configured: str | None,
    default: Path,
    *,
    base: Path,
    label: str,
) -> Path:
    path = Path(configured) if configured else default
    try:
        path.resolve().relative_to(base.resolve())
    except ValueError as exc:
        raise SystemExit(
            f"{label} must stay inside the representation output directory "
            f"{base}: {path}"
        ) from exc
    return path


def resolve_outputs(
    args: argparse.Namespace,
    logger: logging.Logger,
    *,
    scoring_units: str | None = None,
) -> dict[str, Path]:
    namespace = _output_namespace(args.scoring_representation, scoring_units)
    if args.output_root:
        run_root = Path(args.output_root)
    else:
        run_root = _default_evaluation_root(args.manifest_in)
        logger.info("No --output_root supplied; using run root: %s", run_root)

    known_namespaces = {"orthographic", "ipa"}
    if run_root.name in known_namespaces:
        if run_root.name != namespace:
            raise SystemExit(
                "Output path representation does not match scoring mode: "
                f"{run_root} vs {namespace}"
            )
        base = run_root
    else:
        base = run_root / namespace
    args.output_root = str(base)
    logger.info("Writing %s evaluation under: %s", namespace, base)

    out_csv = _representation_scoped_path(
        args.out_csv,
        base / "egra_eval_detailed.csv",
        base=base,
        label="--out_csv",
    )
    args.out_csv = str(out_csv)

    outputs = {
        "base": Path(base),
        "namespace": Path(namespace),
    }
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    base.mkdir(parents=True, exist_ok=True)
    return outputs


def write_evaluation_metadata(
    *,
    base: Path,
    manifest_in: str,
    requested_representation: str,
    scoring_units: str,
    namespace: str,
    reference_metadata_path: str | Path | None = None,
) -> Path:
    payload = {
        "schema_version": 2,
        "provenance_schema_version": PIPELINE_PROVENANCE_SCHEMA_VERSION,
        "status": "complete",
        "completed_at": datetime.now().astimezone().isoformat(),
        "source_manifest": str(manifest_in),
        "requested_scoring_representation": requested_representation,
        "effective_scoring_units": scoring_units,
        "output_namespace": namespace,
        "representation_compatible": True,
        "reference_integrity_enforced": True,
        "pipeline_provenance": build_evaluation_provenance(
            base=base,
            manifest_in=manifest_in,
            requested_representation=requested_representation,
            scoring_units=scoring_units,
            namespace=namespace,
            reference_metadata_path=reference_metadata_path,
        ),
    }
    path = base / "evaluation_metadata.json"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def load_eval_manifest(
    path: str,
    *,
    audio_key: str,
    ref_key: str,
    can_key: str,
    hyp_key: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    rows = []
    total = 0
    bad = 0
    in_path = Path(path)
    if not in_path.exists():
        raise SystemExit(f"--manifest_in file not found: {in_path}")

    text = in_path.read_text(encoding="utf-8")
    objects: list[dict[str, Any]] = []

    # 1) Try strict JSON first (single object or array of objects).
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            objects = [parsed]
        elif isinstance(parsed, list):
            objects = [obj for obj in parsed if isinstance(obj, dict)]
            bad += sum(1 for obj in parsed if not isinstance(obj, dict))
        else:
            bad += 1
    except json.JSONDecodeError:
        # 2) Fallback: concatenated JSON objects (object after object, possibly pretty-printed).
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
                bad += 1
                # Move forward to avoid infinite loop on malformed content.
                idx += 1
                continue
            if isinstance(obj, dict):
                objects.append(obj)
            else:
                bad += 1
            idx = next_idx

    for obj in objects:
        total += 1
        audio = str(obj.get(audio_key, "") or "").strip()
        if not audio:
            bad += 1
            continue
        p = Path(audio)
        rows.append(
            {
                "audio_path": audio,
                "audio_name": p.name,
                "audio_stem": p.stem,
                "manifest_ref_text": obj.get(ref_key, ""),
                "manifest_can_text": obj.get(can_key, ""),
                "manifest_hyp_text": obj.get(hyp_key, ""),
                "manifest_hyp_present": hyp_key in obj,
                "manifest_wer": obj.get("wer", None),
                "manifest_tokens": obj.get("tokens", None),
                "manifest_ins_rate": obj.get("ins_rate", None),
                "manifest_del_rate": obj.get("del_rate", None),
                "manifest_sub_rate": obj.get("sub_rate", None),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        # Keep expected columns to avoid downstream KeyError and emit a clear failure.
        out = pd.DataFrame(
            columns=[
                "audio_path",
                "audio_name",
                "audio_stem",
                "manifest_ref_text",
                "manifest_can_text",
                "manifest_hyp_text",
                "manifest_hyp_present",
                "manifest_wer",
                "manifest_tokens",
                "manifest_ins_rate",
                "manifest_del_rate",
                "manifest_sub_rate",
            ]
        )
    try:
        validate_reference_rows(
            out.to_dict(orient="records"),
            text_fields=("manifest_ref_text", "manifest_can_text"),
            audio_field="audio_path",
            source=str(in_path),
        )
    except ReferenceIntegrityError as exc:
        raise SystemExit(str(exc)) from exc
    logger.info(
        "Loaded eval manifest %s | rows=%d | total_lines=%d | skipped=%d",
        in_path,
        len(out),
        total,
        bad,
    )
    return out


def _valid_text_mask(series: pd.Series) -> pd.Series:
    return series.notna() & (series.astype(str).str.strip() != "")


def _extract_learner_id(audio_stem: str) -> str:
    base = audio_stem.split("-audio", 1)[0]
    m = re.match(r"^(\d{6}_\d{6}_[A-Za-z])$", base)
    if m:
        return m.group(1)
    parts = base.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:3])
    return base


def _infer_audio_type(audio_stem: str) -> str:
    source = (
        audio_stem.split("-audio_", 1)[1] if "-audio_" in audio_stem else audio_stem
    )
    source = re.sub(r"_segment\d+$", "", source)
    s = source.lower()

    if "letters_grids_letters_grid" in s or "letters_grid" in s:
        return "full_letter_grid"
    if "non_words_grid_non_words_grid" in s or "non_words_grid" in s or "nonword" in s:
        return "full_nonword_grid"
    if (
        "grid_syllables_1_syllables_grid_1" in s
        or "syllables_grid_1" in s
        or "syllables_grid" in s
    ):
        return "full_syllable1_grid"

    m = re.search(r"iso_letter_(\d+)", s)
    if m:
        return f"iso_letter_{m.group(1)}"
    m = re.search(r"iso_random_letters(\d+)(?:_(\d+))?", s)
    if m:
        suffix = f"_{m.group(2)}" if m.group(2) else ""
        return f"iso_random_letters{m.group(1)}{suffix}"
    m = re.search(r"iso_non_word_?(\d+)", s)
    if m:
        return f"iso_non_word{m.group(1)}"
    m = re.search(r"iso_random_nw(?:_([a-z0-9]+))?(?:_(\d+))?", s)
    if m:
        name = m.group(1) or "unknown"
        idx = f"_{m.group(2)}" if m.group(2) else ""
        return f"iso_random_nw_{name}{idx}"
    m = re.search(r"iso_syllable_?(\d+)", s)
    if m:
        return f"iso_syllable{m.group(1)}"
    m = re.search(r"random_syl_?(\d+)", s)
    if m:
        return f"random_syl{m.group(1)}"
    m = re.search(r"rand_syllable_?(\d+)", s)
    if m:
        return f"rand_syllable{m.group(1)}"
    m = re.search(r"passage_?(\d+)", s)
    if m:
        return f"passage_num{m.group(1)}"

    return "unknown"


def build_eval_rows_from_manifest(
    manifest_df: pd.DataFrame, logger: logging.Logger
) -> pd.DataFrame:
    if manifest_df is None or manifest_df.empty:
        raise SystemExit(
            "Manifest loaded with 0 valid rows. Ensure --manifest_in is JSONL, JSON array, or concatenated JSON objects "
            "and that each record has a non-empty audio filepath key."
        )

    out = manifest_df.copy()
    out["audio_file"] = out["audio_name"].astype(str)

    seg_mask = (
        out["audio_name"]
        .astype(str)
        .str.contains(r"_segment\d+\.wav$", regex=True, na=False)
    )
    seg_count = int(seg_mask.sum())
    nonseg_count = int((~seg_mask).sum())
    if seg_count and nonseg_count:
        logger.warning(
            "Manifest contains mixed granularity; evaluating all rows (segment=%d, non-segment=%d).",
            seg_count,
            nonseg_count,
        )
    elif seg_count:
        logger.info("Manifest appears segment-based (rows=%d).", seg_count)
    else:
        logger.info("Manifest appears non-segment-based (rows=%d).", len(out))

    if out.empty:
        raise SystemExit("No rows found in manifest after parsing.")

    out["learner_id"] = out["audio_stem"].astype(str).apply(_extract_learner_id)
    out["audio_type"] = out["audio_stem"].astype(str).apply(_infer_audio_type)
    out["ref_text"] = out["manifest_ref_text"].fillna("").astype(str)
    out["canonical_text"] = out["manifest_can_text"].fillna("").astype(str)
    out["hyp_text"] = out["manifest_hyp_text"].fillna("").astype(str)
    out["hyp_is_present"] = out["manifest_hyp_present"].fillna(False).astype(bool)

    # Preserve optional NeMo per-file scoring fields (if present in manifest).
    out["nemo_wer"] = pd.to_numeric(out.get("manifest_wer"), errors="coerce")
    out["nemo_tokens"] = pd.to_numeric(out.get("manifest_tokens"), errors="coerce")
    out["nemo_ins_rate"] = pd.to_numeric(out.get("manifest_ins_rate"), errors="coerce")
    out["nemo_del_rate"] = pd.to_numeric(out.get("manifest_del_rate"), errors="coerce")
    out["nemo_sub_rate"] = pd.to_numeric(out.get("manifest_sub_rate"), errors="coerce")

    logger.info(
        "Prepared evaluation rows from manifest: rows=%d | non-empty REF=%d | non-empty CAN=%d | non-empty HYP=%d",
        len(out),
        int(_valid_text_mask(out["ref_text"]).sum()),
        int(_valid_text_mask(out["canonical_text"]).sum()),
        int(_valid_text_mask(out["hyp_text"]).sum()),
    )
    return out


def merge_segmented_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def parse_segment(audio_file: str):
        s = str(audio_file)
        m = re.search(r"_segment(\d+)(\.[^.]+)$", s)
        if m:
            seg_num = int(m.group(1))
            base_audio_file = re.sub(r"_segment\d+(\.[^.]+)$", r"\1", s)
            return base_audio_file, seg_num, True
        return s, None, False

    parsed = df["audio_file"].astype(str).apply(parse_segment)
    df["_base_audio_file"] = parsed.apply(lambda x: x[0])
    df["_segment_num"] = parsed.apply(lambda x: x[1])
    df["_is_segment"] = parsed.apply(lambda x: x[2])

    rows = []

    for _, g in df.groupby("_base_audio_file", sort=False):
        has_segments = g["_is_segment"].any()

        if has_segments:
            g_seg = g.sort_values("_segment_num", kind="stable")

            row = g_seg.iloc[0].copy()
            row["audio_file"] = g_seg["_base_audio_file"].iloc[0]
            row["canonical_text"] = g_seg["canonical_text"].iloc[0]
            row["ref_text"] = " ".join(
                g_seg["ref_text"].fillna("").astype(str).str.strip()
            ).strip()
            row["hyp_text"] = " ".join(
                g_seg["hyp_text"].fillna("").astype(str).str.strip()
            ).strip()

            rows.append(row)
        else:
            rows.extend(g.iloc[i].copy() for i in range(len(g)))

    out = pd.DataFrame(rows).drop(
        columns=["_base_audio_file", "_segment_num", "_is_segment"],
        errors="ignore",
    )

    return out.reset_index(drop=True)


def main() -> None:
    logger = setup_logger()
    args = parse_args()

    try:
        layout = resolve_dataset_paths(args.dataset_root)
    except DatasetLayoutError as exc:
        raise SystemExit(str(exc)) from exc

    args.meta_csv = args.meta_csv or str(layout.metadata_csv)
    df_meta = pd.read_csv(args.meta_csv)
    logger.info("Loaded META rows: %d", len(df_meta))

    manifest_df = load_eval_manifest(
        args.manifest_in,
        audio_key=args.manifest_audio_key,
        ref_key=args.manifest_ref_key,
        can_key=args.manifest_can_key,
        hyp_key=args.manifest_hyp_key,
        logger=logger,
    )
    try:
        manifest_df, scoring_units = prepare_scoring_texts(
            manifest_df,
            dataset_root=layout.root,
            manifest_in=args.manifest_in,
            logger=logger,
            scoring_representation=args.scoring_representation,
        )
    except ScoringRepresentationError as exc:
        raise SystemExit(str(exc)) from exc
    outputs = resolve_outputs(args, logger, scoring_units=scoring_units)
    df_eval = build_eval_rows_from_manifest(manifest_df, logger)
    if scoring_units == "orthographic":
        df_eval = adjust_letter_canonical_text(df_eval, logger)

    # Simplify to only required columns
    required_cols = [
        "learner_id",
        "audio_file",
        "audio_type",
        "canonical_text",
        "ref_text",
        "hyp_text",
    ]
    df_eval = df_eval[[c for c in required_cols if c in df_eval.columns]].copy()

    # Merge segments
    df_eval = merge_segmented_rows(df_eval)

    # Calculate per-row scores
    df_scores_per_row = evaluate_rows(df_eval)

    # Attach learner metadata to the scored rows.
    meta_cols = ["learner_id", "gender", "child_grade", "child_age", "region"]
    df_detailed = df_scores_per_row.merge(
        df_meta[meta_cols], on="learner_id", how="left"
    )

    # Write out individual scores (the whole dataframe with meta data)
    base = Path(args.output_root)
    args.out_csv = args.out_csv or str(base / "egra_eval_detailed.csv")
    logger.info(f"Writing: {args.out_csv}")
    df_detailed.to_csv(args.out_csv)

    # The row-wise table can be filtered later for region or other cohort views.

    # Aggregate scores
    scores_dict = aggregate_row_scores(
        df_scores_per_row,
        outputs["base"],
        args.detailed,
        scoring_units=scoring_units,
    )

    # Write summary to file
    fn = base / "egra_eval_summary.txt"
    logger.info(f"Writing: {fn}")
    with open(fn, "w", encoding="utf-8") as f:
        for key in [
            "global",
            "passage_passage",
            "syllables_grid",
            "syllables_isolated",
            "nonwords_grid",
            "nonwords_isolated",
            "letters_grid",
            "letters_isolated",
        ]:
            subdict = scores_dict[key]
            f.write(f"{key.upper()}\n")
            for metric, value in subdict.items():
                if metric in ["corr"]:
                    f.write(f"  {metric}: {value:.4f}\n")
                else:
                    f.write(f"  {metric}: {value:.2f}%\n")
            f.write("\n")

    metadata_path = write_evaluation_metadata(
        base=base,
        manifest_in=args.manifest_in,
        requested_representation=args.scoring_representation,
        scoring_units=scoring_units,
        namespace=outputs["namespace"].name,
        reference_metadata_path=default_output_dir(layout.root) / METADATA_NAME,
    )
    logger.info("Writing: %s", metadata_path)

    print()
    for key in [
        "global",
        "passage_passage",
        "syllables_grid",
        "syllables_isolated",
        "nonwords_grid",
        "nonwords_isolated",
        "letters_grid",
        "letters_isolated",
    ]:
        subdict = scores_dict[key]
        print(f"{key.upper()}")
        for metric, value in subdict.items():
            if metric in ["corr"]:
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value:.2f}%")
        print()


if __name__ == "__main__":
    main()
