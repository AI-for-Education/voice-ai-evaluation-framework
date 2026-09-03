#!/usr/bin/env python3
"""Verify and migrate legacy generic IPA evaluations into exact G2P namespaces."""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from egra_eval2.evaluate import aggregate_row_scores, evaluate_rows
from egra_eval2.leaderboard import parse_summary
from egra_eval2.manifest_cleaner import clean_text
from egra_eval2.reference.g2p import G2PSystem, resolve_g2p_system
from egra_eval2.reference.ipa_inventory_maps import (
    IPAInventoryMapError,
    build_ipa_aligner,
    describe_ipa_inventory_route,
)
from egra_eval2.reference.views import build_reference_views
from eval_pipeline2 import build_eval_rows_from_manifest, merge_segmented_rows
from inference.pipeline_provenance import (
    PIPELINE_PROVENANCE_SCHEMA_VERSION,
    build_evaluation_provenance,
    inference_profile_from_run_metadata,
)


LOGGER = logging.getLogger("migrate_legacy_ipa")


class LegacyIPAMigrationError(ValueError):
    """Raised when legacy evidence cannot be verified without guessing."""


@dataclass(frozen=True)
class VerifiedLegacyEvaluation:
    run_dir: Path
    legacy_dir: Path
    source_manifest: Path
    aligned_manifest: Path
    rows: tuple[dict[str, str], ...]
    hypothesis_route: str
    evaluation_metadata: dict[str, Any]


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LegacyIPAMigrationError(f"Invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise LegacyIPAMigrationError(f"Invalid {label}: {path}")
    return value


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise LegacyIPAMigrationError(
                        f"{label} row {line_number} is not an object: {path}"
                    )
                rows.append(value)
    except json.JSONDecodeError as exc:
        raise LegacyIPAMigrationError(
            f"Invalid {label} JSON at line {line_number}: {path}"
        ) from exc
    except OSError as exc:
        raise LegacyIPAMigrationError(f"Could not read {label}: {path}") from exc
    if not rows:
        raise LegacyIPAMigrationError(f"{label} contains no rows: {path}")
    return rows


def _normalized_ipa(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return " ".join(unicodedata.normalize("NFC", str(value)).split())


def _resolve_source_manifest(run_dir: Path, metadata: dict[str, Any]) -> Path:
    recorded = metadata.get("source_manifest")
    candidates: list[Path] = []
    if isinstance(recorded, str) and recorded.strip():
        recorded_path = Path(recorded)
        candidates.append(recorded_path)
        if not recorded_path.is_absolute():
            candidates.append(Path.cwd() / recorded_path)
    candidates.append(run_dir / "manifests" / "ref_manifest.clean.jsonl")
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise LegacyIPAMigrationError(
        f"Source manifest is missing for legacy evaluation: {run_dir.name}"
    )


def _resolve_legacy_aligned_manifest(run_dir: Path, source_manifest: Path) -> Path:
    expected = source_manifest.with_name(f"{source_manifest.stem}.ipa_aligned.jsonl")
    if expected.is_file():
        return expected
    matches = sorted((run_dir / "manifests").glob("*.ipa_aligned.jsonl"))
    if len(matches) == 1:
        return matches[0].resolve()
    raise LegacyIPAMigrationError(
        f"A unique legacy aligned IPA manifest was not found: {run_dir}"
    )


def _run_profile(evaluations_root: Path, run_dir: Path) -> dict[str, Any]:
    path = evaluations_root.parent / "transcripts" / run_dir.name / "run_metadata.json"
    metadata = _load_json_object(path, "source run metadata")
    profile = inference_profile_from_run_metadata(metadata)
    if not profile:
        raise LegacyIPAMigrationError(f"Source run has no inference profile: {path}")
    return profile


def _recompute_aligned_rows(
    source_rows: Sequence[dict[str, Any]],
    profile: dict[str, Any],
    system: G2PSystem,
    reference_cache: dict[tuple[str, str], tuple[str, str]] | None = None,
) -> tuple[list[dict[str, str]], str]:
    required = {"audio_filepath", "can_text", "ref_text", "pred_text"}
    for index, row in enumerate(source_rows, start=1):
        missing = sorted(required - set(row))
        if missing:
            raise LegacyIPAMigrationError(
                f"Source manifest row {index} is missing: {', '.join(missing)}"
            )

    # Reference views apply the representation-safe orthographic cleaner to
    # CAN/REF before either phonemizer. HYP remains the authoritative model text.
    can_texts = [clean_text(str(row["can_text"] or "")) for row in source_rows]
    ref_texts = [clean_text(str(row["ref_text"] or "")) for row in source_rows]
    cache = reference_cache if reference_cache is not None else {}
    reference_keys = list(zip(can_texts, ref_texts))
    missing_keys = list(dict.fromkeys(key for key in reference_keys if key not in cache))
    if missing_keys:
        converted_references = system.phonemize(
            [key[0] for key in missing_keys] + [key[1] for key in missing_keys]
        )
        if len(converted_references) != len(missing_keys) * 2:
            raise LegacyIPAMigrationError(
                "G2P changed row cardinality while recomputing legacy references"
            )
        split = len(missing_keys)
        for key, can_ipa, ref_ipa in zip(
            missing_keys,
            converted_references[:split],
            converted_references[split:],
        ):
            cache[key] = (str(can_ipa), str(ref_ipa))
    can_values = [cache[key][0] for key in reference_keys]
    ref_values = [cache[key][1] for key in reference_keys]
    output_units = profile.get("output_units")
    if output_units == "orthographic":
        hyp_texts = [str(row["pred_text"] or "") for row in source_rows]
        hyp_values = system.phonemize(hyp_texts)
        if len(hyp_values) != len(source_rows):
            raise LegacyIPAMigrationError(
                "G2P changed row cardinality while recomputing legacy IPA"
            )
        route = f"orthographic -> {system.display_name} ({system.system_id})"
    elif output_units == "phoneme":
        source_inventory = profile.get("output_inventory")
        if profile.get("output_notation") != "ipa" or not isinstance(
            source_inventory, str
        ):
            raise LegacyIPAMigrationError(
                "Native-phoneme run lacks an explicit IPA inventory"
            )
        try:
            align = build_ipa_aligner(source_inventory, system.inventory)
            hyp_values = [align(row["pred_text"]) for row in source_rows]
        except IPAInventoryMapError as exc:
            raise LegacyIPAMigrationError(str(exc)) from exc
        route = describe_ipa_inventory_route(source_inventory, system.inventory)
    else:
        raise LegacyIPAMigrationError(
            f"Unsupported source output units: {output_units!r}"
        )

    return (
        [
            {
                "audio_filepath": str(source["audio_filepath"]),
                "can_text": _normalized_ipa(can_text),
                "ref_text": _normalized_ipa(ref_text),
                "pred_text": _normalized_ipa(hyp_text),
            }
            for source, can_text, ref_text, hyp_text in zip(
                source_rows, can_values, ref_values, hyp_values
            )
        ],
        route,
    )


def _compare_aligned_rows(
    recomputed: Sequence[dict[str, str]], legacy: Sequence[dict[str, Any]]
) -> None:
    if len(recomputed) != len(legacy):
        raise LegacyIPAMigrationError(
            f"Aligned row-count mismatch: recomputed={len(recomputed)}, legacy={len(legacy)}"
        )
    fields = ("audio_filepath", "can_text", "ref_text", "pred_text")
    for index, (expected, observed) in enumerate(zip(recomputed, legacy), start=1):
        normalized_observed = {
            "audio_filepath": str(observed.get("audio_filepath", "")),
            "can_text": _normalized_ipa(observed.get("can_text", "")),
            "ref_text": _normalized_ipa(observed.get("ref_text", "")),
            "pred_text": _normalized_ipa(observed.get("pred_text", "")),
        }
        mismatched = [
            field for field in fields if expected[field] != normalized_observed[field]
        ]
        if mismatched:
            raise LegacyIPAMigrationError(
                f"Aligned row {index} differs in: {', '.join(mismatched)}"
            )


def _evaluation_rows_for_aligned(rows: Sequence[dict[str, str]]) -> pd.DataFrame:
    manifest = pd.DataFrame(
        [
            {
                "audio_path": row["audio_filepath"],
                "audio_name": Path(row["audio_filepath"]).name,
                "audio_stem": Path(row["audio_filepath"]).stem,
                "manifest_ref_text": row["ref_text"],
                "manifest_can_text": row["can_text"],
                "manifest_hyp_text": row["pred_text"],
                "manifest_hyp_present": True,
            }
            for row in rows
        ]
    )
    evaluation = build_eval_rows_from_manifest(manifest, LOGGER)
    evaluation = evaluation[
        [
            "learner_id",
            "audio_file",
            "audio_type",
            "canonical_text",
            "ref_text",
            "hyp_text",
        ]
    ]
    evaluation = merge_segmented_rows(evaluation)
    return evaluation


def _metrics_for_rows(
    rows: Sequence[dict[str, str]],
    legacy_detailed_path: Path | None = None,
) -> dict[str, dict[str, float]]:
    evaluation = _evaluation_rows_for_aligned(rows)
    if legacy_detailed_path is None:
        row_scores = evaluate_rows(evaluation)
    else:
        try:
            row_scores = pd.read_csv(legacy_detailed_path)
        except (OSError, pd.errors.ParserError) as exc:
            raise LegacyIPAMigrationError(
                f"Could not read legacy detailed scores: {legacy_detailed_path}"
            ) from exc
        if len(row_scores) != len(evaluation):
            raise LegacyIPAMigrationError(
                "Detailed-score row count does not match recomputed aligned rows"
            )
        comparisons = (
            ("audio_file", "audio_file", False),
            ("canonical_text", "CAN", True),
            ("ref_text", "REF", True),
            ("hyp_text", "HYP", True),
        )
        for expected_column, recorded_column, normalize in comparisons:
            if recorded_column not in row_scores:
                raise LegacyIPAMigrationError(
                    f"Detailed scores lack {recorded_column}: {legacy_detailed_path}"
                )
            expected_values = evaluation[expected_column].tolist()
            recorded_values = row_scores[recorded_column].tolist()
            if normalize:
                expected_values = [_normalized_ipa(value) for value in expected_values]
                recorded_values = [_normalized_ipa(value) for value in recorded_values]
            else:
                expected_values = [str(value) for value in expected_values]
                recorded_values = [str(value) for value in recorded_values]
            if expected_values != recorded_values:
                raise LegacyIPAMigrationError(
                    f"Detailed scores do not match recomputed {recorded_column} rows"
                )
    return aggregate_row_scores(
        row_scores, Path.cwd(), detailed=False, scoring_units="phoneme"
    )


def _equal_at_published_precision(actual: float, recorded: float, metric: str) -> bool:
    if math.isnan(actual) or math.isnan(recorded):
        return math.isnan(actual) and math.isnan(recorded)
    decimals = 4 if metric == "corr" else 2
    return round(float(actual), decimals) == round(float(recorded), decimals)


def _compare_summary(
    recomputed: dict[str, dict[str, float]], recorded: dict[str, dict[str, float]]
) -> None:
    for section, metrics in recorded.items():
        actual_metrics = recomputed.get(section)
        if actual_metrics is None:
            raise LegacyIPAMigrationError(
                f"Recomputed metrics have no {section} section"
            )
        for metric, expected in metrics.items():
            actual = actual_metrics.get(metric)
            if actual is None or not _equal_at_published_precision(
                float(actual), float(expected), metric
            ):
                raise LegacyIPAMigrationError(
                    f"Aggregate mismatch for {section}.{metric}: "
                    f"recomputed={actual!r}, recorded={expected!r}"
                )


def verify_legacy_evaluation(
    evaluations_root: Path,
    run_dir: Path,
    system: G2PSystem,
    reference_cache: dict[tuple[str, str], tuple[str, str]] | None = None,
) -> VerifiedLegacyEvaluation:
    """Recompute and compare one generic ``run/ipa`` result without writing."""
    legacy_dir = run_dir / "ipa"
    metadata_path = legacy_dir / "evaluation_metadata.json"
    summary_path = legacy_dir / "egra_eval_summary.txt"
    metadata = _load_json_object(metadata_path, "legacy evaluation metadata")
    source_manifest = _resolve_source_manifest(run_dir, metadata)
    aligned_manifest = _resolve_legacy_aligned_manifest(run_dir, source_manifest)
    source_rows = _load_jsonl(source_manifest, "source manifest")
    profile = _run_profile(evaluations_root, run_dir)
    rows, route = _recompute_aligned_rows(
        source_rows, profile, system, reference_cache=reference_cache
    )
    _compare_aligned_rows(rows, _load_jsonl(aligned_manifest, "legacy aligned manifest"))
    _compare_summary(
        _metrics_for_rows(rows, legacy_dir / "egra_eval_detailed.csv"),
        parse_summary(summary_path),
    )
    return VerifiedLegacyEvaluation(
        run_dir=run_dir,
        legacy_dir=legacy_dir,
        source_manifest=source_manifest,
        aligned_manifest=aligned_manifest,
        rows=tuple(rows),
        hypothesis_route=route,
        evaluation_metadata=metadata,
    )


def discover_legacy_evaluations(evaluations_root: str | Path) -> list[Path]:
    """Return runs with a root-level generic IPA result, excluding archives."""
    root = Path(evaluations_root)
    if not root.is_dir():
        raise LegacyIPAMigrationError(f"Evaluations root not found: {root}")
    return sorted(
        run_dir
        for run_dir in root.iterdir()
        if run_dir.is_dir()
        and (run_dir / "ipa" / "evaluation_metadata.json").is_file()
        and (run_dir / "ipa" / "egra_eval_summary.txt").is_file()
    )


def _write_jsonl(path: Path, rows: Sequence[dict[str, str]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _already_migrated(target: Path, system: G2PSystem) -> bool:
    metadata_path = target / "evaluation_metadata.json"
    if not metadata_path.is_file():
        return False
    metadata = _load_json_object(metadata_path, "migrated evaluation metadata")
    return (
        metadata.get("g2p_system") == system.metadata()
        and (target / "egra_eval_summary.txt").is_file()
        and (target / "egra_eval_detailed.csv").is_file()
    )


def _apply_verified(
    verified: VerifiedLegacyEvaluation,
    system: G2PSystem,
    *,
    reference_view_path: Path,
    reference_metadata_path: Path,
) -> str:
    target = verified.legacy_dir / system.system_id
    if target.exists():
        if _already_migrated(target, system):
            return "already_migrated"
        raise LegacyIPAMigrationError(
            f"Exact-system target exists but is not a valid migration: {target}"
        )
    staging = verified.legacy_dir / f".{system.system_id}.migration.tmp"
    if staging.exists():
        raise LegacyIPAMigrationError(
            f"A previous migration staging directory needs review: {staging}"
        )
    staging.mkdir(parents=True)
    for source in verified.legacy_dir.iterdir():
        if source.is_file() and source.name != "evaluation_metadata.json":
            shutil.copy2(source, staging / source.name)

    exact_aligned = verified.source_manifest.with_name(
        f"{verified.source_manifest.stem}.ipa_aligned.{system.system_id}.jsonl"
    )
    _write_jsonl(exact_aligned, verified.rows)
    original = verified.evaluation_metadata
    provenance = build_evaluation_provenance(
        base=staging,
        manifest_in=verified.source_manifest,
        requested_representation="ipa",
        scoring_units="phoneme",
        namespace="ipa",
        reference_metadata_path=reference_metadata_path,
        reference_view_path=reference_view_path,
        aligned_manifest_path=exact_aligned,
        g2p_system=system.metadata(),
        hypothesis_route=verified.hypothesis_route,
        recording_mode="migration_time_exact_recomputation",
        limitations=(
            "Copied score artifacts were admitted only after exact aligned-row "
            "and published-precision aggregate verification.",
        ),
    )
    for link in provenance.get("links", {}).values():
        if isinstance(link, dict) and isinstance(link.get("path"), str):
            link["path"] = link["path"].replace(str(staging), str(target), 1)
    metadata = {
        "schema_version": 3,
        "provenance_schema_version": PIPELINE_PROVENANCE_SCHEMA_VERSION,
        "status": "complete",
        "completed_at": original.get("completed_at", ""),
        "source_manifest": str(verified.source_manifest),
        "requested_scoring_representation": "ipa",
        "effective_scoring_units": "phoneme",
        "output_namespace": "ipa",
        "representation_compatible": True,
        "reference_integrity_enforced": True,
        "hypothesis_route": verified.hypothesis_route,
        "g2p_system": system.metadata(),
        "migration": {
            "source": str(verified.legacy_dir),
            "method": "exact_recomputation_and_published_precision_comparison",
            "legacy_preserved": True,
        },
        "pipeline_provenance": provenance,
    }
    metadata_path = staging / "evaluation_metadata.json"
    temporary = metadata_path.with_name(f".{metadata_path.name}.tmp")
    temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(metadata_path)
    staging.replace(target)
    return "migrated"


def migrate_legacy_ipa(
    *,
    evaluations_root: str | Path,
    dataset_root: str | Path,
    reference_manifest: str | Path | None = None,
    apply: bool = False,
    system: G2PSystem | None = None,
) -> dict[str, Any]:
    """Verify all generic IPA results and optionally copy verified artifacts."""
    root = Path(evaluations_root)
    selected = system or resolve_g2p_system("africa_g2p")
    runs = discover_legacy_evaluations(root)
    reference_paths = None
    if apply:
        if reference_manifest is None:
            raise LegacyIPAMigrationError(
                "--reference-manifest is required with --apply so migrated metadata "
                "can link the exact reference view"
            )
        reference_paths = build_reference_views(
            dataset_root=dataset_root,
            manifest_in=reference_manifest,
            g2p_system=selected,
        )

    report: dict[str, Any] = {
        "mode": "apply" if apply else "dry-run",
        "g2p_system": selected.metadata(),
        "discovered": len(runs),
        "verified": 0,
        "migrated": 0,
        "already_migrated": 0,
        "rejected": 0,
        "runs": [],
    }
    reference_cache: dict[tuple[str, str], tuple[str, str]] = {}
    for run_dir in runs:
        try:
            verified = verify_legacy_evaluation(
                root, run_dir, selected, reference_cache=reference_cache
            )
            report["verified"] += 1
            status = "verified"
            if apply and reference_paths is not None:
                status = _apply_verified(
                    verified,
                    selected,
                    reference_view_path=reference_paths.ipa,
                    reference_metadata_path=reference_paths.metadata,
                )
                report[status] += 1
            report["runs"].append({"run_id": run_dir.name, "status": status})
        except (
            LegacyIPAMigrationError,
            ValueError,
            OSError,
            KeyError,
            TypeError,
            AssertionError,
        ) as exc:
            report["rejected"] += 1
            report["runs"].append(
                {"run_id": run_dir.name, "status": "rejected", "reason": str(exc)}
            )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run verification and optional exact-system migration of legacy "
            "generic IPA evaluations."
        )
    )
    parser.add_argument("--dataset-root", "--dataset_root", required=True)
    parser.add_argument(
        "--evaluations-root",
        "--evaluations_root",
        default="input_output_data/output/evaluations",
    )
    parser.add_argument(
        "--reference-manifest",
        "--reference_manifest",
        default=None,
        help="Required with --apply; source for the exact Africa G2P reference view.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Copy only verified artifacts; without this flag the command never writes.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args(argv)
    try:
        report = migrate_legacy_ipa(
            evaluations_root=args.evaluations_root,
            dataset_root=args.dataset_root,
            reference_manifest=args.reference_manifest,
            apply=args.apply,
        )
    except LegacyIPAMigrationError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["rejected"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
