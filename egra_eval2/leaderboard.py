"""Build representation-compatible model leaderboards from evaluation outputs."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from egra_eval2.leaderboard_context import (
    artifact_context as _artifact_context,
    contract_references as _contract_references,
    decoding_label as _decoding_label,
    execution_stack_context as _execution_stack_context,
    execution_target_label as _execution_target_label,
    inference_engine_version as _inference_engine_version,
    nonnegative_int as _nonnegative_int,
    preprocessing_context as _preprocessing_context,
)
from egra_eval2.model_presentation import (
    MODEL_NAME_MAPPING_COLUMNS,
    MODEL_PRESENTATION_REGISTRY_PATH,
    build_name_mapping_rows,
    default_model_presentation_registry,
    presentation_for_inference_setup,
    structured_model_label,
)


SUMMARY_SECTIONS = (
    "global",
    "passage_passage",
    "syllables_grid",
    "syllables_isolated",
    "nonwords_grid",
    "nonwords_isolated",
    "letters_grid",
    "letters_isolated",
)

TASK_SECTIONS = SUMMARY_SECTIONS[1:]
CORRELATION_SECTIONS = {
    "passage_passage",
    "syllables_grid",
    "nonwords_grid",
    "letters_grid",
}
ISOLATED_SECTIONS = {
    "syllables_isolated",
    "nonwords_isolated",
    "letters_isolated",
}

REPRESENTATIONS = {
    "orthographic": {
        "scoring_units": "orthographic",
        "error_metric": "wer",
        "filename": "leaderboard_orthographic.csv",
    },
    "ipa": {
        "scoring_units": "phoneme",
        "error_metric": "per",
        "filename": "leaderboard_ipa.csv",
    },
}

# These retired configurations remain in the inference registry so historical run
# metadata stays reproducible, but they must not appear in active leaderboards or
# stakeholder-report packages.
RETIRED_LEADERBOARD_MODEL_IDS = frozenset(
    {
        "bookbot-orthographic-ctc",
        "bookbot-orthographic-ctc-5gram",
    }
)

_METRIC_RE = re.compile(
    r"^\s+([a-z][a-z0-9_]*):\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)|nan)%?\s*$",
    re.IGNORECASE,
)


class LeaderboardError(ValueError):
    """Raised when a leaderboard input cannot be parsed safely."""


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LeaderboardError(f"Invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise LeaderboardError(f"Invalid {label}: {path}")
    return payload


def parse_summary(path: str | Path) -> dict[str, dict[str, float]]:
    """Parse one ``egra_eval_summary.txt`` into section/metric values."""
    summary_path = Path(path)
    try:
        lines = summary_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise LeaderboardError(f"Could not read evaluation summary: {summary_path}") from exc

    result: dict[str, dict[str, float]] = {}
    section: str | None = None
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        candidate = line.lower()
        if candidate in SUMMARY_SECTIONS:
            section = candidate
            result.setdefault(section, {})
            continue
        match = _METRIC_RE.match(raw_line)
        if match and section is not None:
            metric, raw_value = match.groups()
            result[section][metric.lower()] = float(raw_value)

    if "global" not in result:
        raise LeaderboardError(f"Evaluation summary has no GLOBAL section: {summary_path}")
    return result


def _run_metadata_for_run(
    evaluations_root: Path,
    run_name: str,
) -> dict[str, Any]:
    transcript_metadata = (
        evaluations_root.parent / "transcripts" / run_name / "run_metadata.json"
    )
    return _load_json_object(transcript_metadata, "ASR run metadata")


def _postprocessing_for_run(
    evaluations_root: Path,
    run_name: str,
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    summary = run_metadata.get("postprocessing")
    adjusted = 0
    removed = 0
    method = "none"
    source = "none"
    total_results = _nonnegative_int(
        (run_metadata.get("output") or {}).get("results")
        if isinstance(run_metadata.get("output"), dict)
        else 0
    )

    if isinstance(summary, dict):
        configured_method = summary.get("method")
        method = (
            configured_method.strip()
            if isinstance(configured_method, str) and configured_method.strip()
            else "unspecified"
        )
        adjusted = _nonnegative_int(summary.get("adjusted_results"))
        removed = _nonnegative_int(summary.get("total_words_removed"))
        source = "run_metadata"
    else:
        backend = run_metadata.get("backend")
        guard_configured = (
            isinstance(backend, dict)
            and backend.get("hallucination_guard") is not None
        )
        if guard_configured:
            method = "hallucination_guard"
            transcript_path = (
                evaluations_root.parent
                / "transcripts"
                / run_name
                / "transcriptions.jsonl"
            )
            counted_rows = 0
            try:
                with transcript_path.open("r", encoding="utf-8") as stream:
                    for line in stream:
                        if not line.strip():
                            continue
                        counted_rows += 1
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        raw = row.get("raw_pred_text")
                        if not isinstance(raw, str):
                            continue
                        pred = row.get("pred_text")
                        pred = pred if isinstance(pred, str) else ""
                        adjusted += 1
                        removed += max(0, len(raw.split()) - len(pred.split()))
                source = "raw_pred_text fallback"
                if not total_results:
                    total_results = counted_rows
            except OSError:
                source = "legacy metadata only"

    rate = 100.0 * adjusted / total_results if total_results else 0.0
    if adjusted:
        scored_hypothesis = "pred_text (post-processed)"
    elif method != "none":
        scored_hypothesis = "pred_text (guard enabled; no changes)"
    else:
        scored_hypothesis = "pred_text"
    return {
        "scored_hypothesis": scored_hypothesis,
        "postprocessing_method": method,
        "postprocessed_rows": adjusted,
        "postprocessed_rows_pct": rate,
        "postprocessing_words_removed": removed,
        "postprocessing_audit_source": source,
    }


def _hypothesis_route(namespace: str, native_output_units: str) -> str:
    if namespace == "orthographic":
        return "native orthographic"
    if native_output_units == "phoneme":
        return "native IPA -> canonical IPA"
    return "orthographic -> IPA (Africa G2P)"


def _evaluation_status(run_metadata: dict[str, Any]) -> str:
    output = run_metadata.get("output")
    if not isinstance(output, dict):
        return "scored"
    results = _nonnegative_int(output.get("results"))
    scope = "smoke test" if output.get("smoke_test") is True else "full run"
    if results:
        return f"scored ({scope}, {results:,} items)"
    return f"scored ({scope})"


def _flatten_summary(
    summary: dict[str, dict[str, float]], error_metric: str
) -> dict[str, float]:
    values: dict[str, float] = {}
    for section in SUMMARY_SECTIONS:
        section_values = summary.get(section, {})
        values[f"{section}_{error_metric}"] = section_values.get(
            error_metric, math.nan
        )
        if section == "global":
            values["global_mer"] = section_values.get("mer", math.nan)
            continue
        values[f"{section}_mer"] = section_values.get("mer", math.nan)
        if section in CORRELATION_SECTIONS:
            values[f"{section}_corr"] = section_values.get("corr", math.nan)
        elif section in ISOLATED_SECTIONS:
            values[f"{section}_accuracy"] = section_values.get(
                "accuracy", math.nan
            )
    return values


def _candidate_row(
    evaluations_root: Path,
    run_dir: Path,
    namespace: str,
) -> dict[str, Any]:
    config = REPRESENTATIONS[namespace]
    representation_dir = run_dir / namespace
    evaluation_metadata_path = representation_dir / "evaluation_metadata.json"
    summary_path = representation_dir / "egra_eval_summary.txt"

    evaluation_metadata = _load_json_object(
        evaluation_metadata_path, "evaluation metadata"
    )
    expected_units = config["scoring_units"]
    checks = {
        "status=complete": evaluation_metadata.get("status") == "complete",
        "matching output_namespace": (
            evaluation_metadata.get("output_namespace") == namespace
        ),
        "matching effective_scoring_units": (
            evaluation_metadata.get("effective_scoring_units") == expected_units
        ),
        "representation_compatible=true": (
            evaluation_metadata.get("representation_compatible") is True
        ),
    }
    failed = [label for label, passed in checks.items() if not passed]
    if failed:
        raise LeaderboardError(
            f"Ineligible {namespace} evaluation {representation_dir}: "
            + ", ".join(failed)
        )

    run_metadata = _run_metadata_for_run(evaluations_root, run_dir.name)
    profile = run_metadata.get("inference_profile", run_metadata.get("profile"))
    if not isinstance(profile, dict):
        metadata_path = (
            evaluations_root.parent
            / "transcripts"
            / run_dir.name
            / "run_metadata.json"
        )
        raise LeaderboardError(f"ASR run metadata has no profile: {metadata_path}")
    inference_setup_id = profile.get(
        "inference_setup_id", profile.get("id")
    )
    native_output_units = profile.get("output_units")
    if not isinstance(inference_setup_id, str) or not inference_setup_id.strip():
        raise LeaderboardError(
            f"Run inference profile has no inference setup id: {run_dir.name}"
        )
    if native_output_units not in {"orthographic", "phoneme"}:
        raise LeaderboardError(
            f"Run profile has unsupported output units: {run_dir.name}"
        )
    if namespace == "orthographic" and native_output_units != "orthographic":
        raise LeaderboardError(
            f"Native phoneme run cannot enter the orthographic leaderboard: {run_dir.name}"
        )

    summary = parse_summary(summary_path)
    error_metric = str(config["error_metric"])
    global_error = summary["global"].get(error_metric)
    if global_error is None or math.isnan(global_error):
        raise LeaderboardError(
            f"Evaluation summary has no finite global {error_metric.upper()}: {summary_path}"
        )

    completed_at = evaluation_metadata.get("completed_at")
    if not isinstance(completed_at, str):
        completed_at = ""
    references, context_evidence = _contract_references(profile, run_metadata)
    execution_stack, stack_launch_observed = _execution_stack_context(
        profile, run_metadata, references
    )
    inference_library = profile.get(
        "inference_library", profile.get("framework")
    )
    if stack_launch_observed:
        context_evidence += "; execution environment observed"
    elif inference_library in {"onnxruntime", "sherpa_onnx"} and (
        _inference_engine_version(run_metadata)[0]
    ):
        context_evidence += "; inference engine observed"
    decoding = _decoding_label(profile)
    execution_target = _execution_target_label(profile, run_metadata, references)
    presentation = presentation_for_inference_setup(inference_setup_id.strip())
    architecture = presentation["architecture"]
    model_artifact = _artifact_context(profile, run_metadata, references)
    input_processing = _preprocessing_context(references)
    inference_setup = f"{input_processing}; {decoding} decoding"
    row: dict[str, Any] = {
        "inference_setup_id": inference_setup_id.strip(),
        "run_id": run_dir.name,
        "model_label": "",
        "model_group": presentation["model_group"],
        "model_name": presentation["model_name"],
        "model_variant": presentation["variant"],
        "official_model_url": presentation["official_model_url"],
        "official_model_url_note": presentation["official_model_url_note"],
        "architecture": architecture["label"],
        "architecture_evidence_status": architecture["evidence_status"],
        "execution_target": execution_target,
        "decoding": decoding,
        "model_artifact": model_artifact,
        "inference_setup": inference_setup,
        "execution_stack": execution_stack,
        "context_evidence": context_evidence,
        "evaluation_status": _evaluation_status(run_metadata),
        "native_output_units": native_output_units,
        "hypothesis_route": _hypothesis_route(namespace, native_output_units),
        "completed_at": completed_at,
        "summary_path": str(summary_path),
        # Deprecated v1 aliases retained for downstream CSV readers.
        "model_id": inference_setup_id.strip(),
        "run_name": run_dir.name,
        "platform": execution_target,
        "artifact_context": model_artifact,
        "preprocessing_context": input_processing,
        "runtime_context": execution_stack,
    }
    row["model_label"] = structured_model_label(presentation, decoding)
    row.update(
        _postprocessing_for_run(evaluations_root, run_dir.name, run_metadata)
    )
    row.update(_flatten_summary(summary, error_metric))
    return row


def _columns_for(namespace: str) -> list[str]:
    metric = str(REPRESENTATIONS[namespace]["error_metric"])
    columns = [
        "rank",
        "model_label",
        "evaluation_status",
        "model_group",
        "model_name",
        "model_variant",
        "official_model_url",
        "official_model_url_note",
        "architecture",
        "architecture_evidence_status",
        "inference_setup_id",
        "run_id",
        "execution_target",
        "model_artifact",
        "inference_setup",
        "execution_stack",
        # Deprecated v1 aliases are written after their canonical fields.
        "model_id",
        "run_name",
        "platform",
        "decoding",
        "artifact_context",
        "preprocessing_context",
        "runtime_context",
        "context_evidence",
        "native_output_units",
        "hypothesis_route",
        "scored_hypothesis",
        "postprocessing_method",
        "postprocessed_rows",
        "postprocessed_rows_pct",
        "postprocessing_words_removed",
        "postprocessing_audit_source",
        f"global_{metric}",
        "global_mer",
    ]
    for section in TASK_SECTIONS:
        columns.extend([f"{section}_{metric}", f"{section}_mer"])
        if section in CORRELATION_SECTIONS:
            columns.append(f"{section}_corr")
        else:
            columns.append(f"{section}_accuracy")
    columns.extend(["completed_at", "summary_path"])
    return columns


def build_leaderboard(
    evaluations_root: str | Path,
    namespace: str,
    *,
    latest_only: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Discover valid completed runs for one scoring representation."""
    if namespace not in REPRESENTATIONS:
        raise ValueError(f"Unsupported leaderboard namespace: {namespace}")
    root = Path(evaluations_root)
    if not root.is_dir():
        raise LeaderboardError(f"Evaluations root not found: {root}")

    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for run_dir in sorted((path for path in root.iterdir() if path.is_dir())):
        representation_dir = run_dir / namespace
        if not representation_dir.is_dir():
            continue
        try:
            row = _candidate_row(root, run_dir, namespace)
            if str(row["inference_setup_id"]) in RETIRED_LEADERBOARD_MODEL_IDS:
                continue
            rows.append(row)
        except LeaderboardError as exc:
            skipped.append(str(exc))

    columns = _columns_for(namespace)
    if not rows:
        return pd.DataFrame(columns=columns), skipped

    frame = pd.DataFrame(rows)
    metric = str(REPRESENTATIONS[namespace]["error_metric"])
    global_metric = f"global_{metric}"
    frame = frame.sort_values(
        ["inference_setup_id", "completed_at", "run_id"],
        ascending=[True, False, False],
        kind="stable",
    )
    if latest_only:
        frame = frame.drop_duplicates(subset="inference_setup_id", keep="first")
    frame = frame.sort_values(
        [global_metric, "inference_setup_id"],
        ascending=[True, True],
        kind="stable",
    ).reset_index(drop=True)
    frame.insert(0, "rank", range(1, len(frame) + 1))
    return frame[columns], skipped


def build_leaderboards(
    evaluations_root: str | Path,
    *,
    latest_only: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict[str, list[str]]]:
    """Build the valid orthographic/WER and IPA/PER leaderboards."""
    frames: dict[str, pd.DataFrame] = {}
    skipped: dict[str, list[str]] = {}
    for namespace in REPRESENTATIONS:
        frames[namespace], skipped[namespace] = build_leaderboard(
            evaluations_root,
            namespace,
            latest_only=latest_only,
        )
    return frames, skipped


def _latest_evaluated_rows(
    frames: dict[str, pd.DataFrame],
) -> dict[str, dict[str, Any]]:
    """Index the newest row per inference setup across namespaces."""
    evaluated = pd.concat(frames.values(), ignore_index=True)
    if evaluated.empty:
        return {}
    latest = evaluated.sort_values(
        ["inference_setup_id", "completed_at"],
        ascending=[True, False],
        kind="stable",
    ).drop_duplicates(subset="inference_setup_id", keep="first")
    return {
        str(row["inference_setup_id"]): row
        for row in latest.to_dict(orient="records")
    }


def _profile_presentation_contexts() -> dict[str, dict[str, str]]:
    """Recover decoder/platform labels for registered but unevaluated profiles."""
    from inference.profile import load_profile

    repository = Path(__file__).resolve().parents[1]
    contexts: dict[str, dict[str, str]] = {}
    profile_paths = sorted((repository / "inference").glob("*/profiles/*.yaml"))
    for path in profile_paths:
        profile = load_profile(path).to_dict()
        setup_id = str(profile["inference_setup_id"])
        if setup_id in RETIRED_LEADERBOARD_MODEL_IDS:
            continue
        references, _ = _contract_references(profile, {})
        contexts[setup_id] = {
            "decoder": _decoding_label(profile),
            "platform": _execution_target_label(profile, {}, references),
        }
    return contexts


def write_leaderboards(
    evaluations_root: str | Path,
    output_dir: str | Path,
    *,
    latest_only: bool = True,
) -> tuple[dict[str, Path], dict[str, list[str]]]:
    """Write separate orthographic and IPA leaderboard CSV files."""
    from inference.provenance import file_identity

    frames, skipped = build_leaderboards(
        evaluations_root, latest_only=latest_only
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for namespace, frame in frames.items():
        path = destination / str(REPRESENTATIONS[namespace]["filename"])
        temporary = path.with_name(f".{path.name}.tmp")
        frame.to_csv(temporary, index=False, float_format="%.4f")
        temporary.replace(path)
        paths[namespace] = path

    registry = default_model_presentation_registry()
    mapping_rows = build_name_mapping_rows(
        _latest_evaluated_rows(frames),
        _profile_presentation_contexts(),
    )
    mapping_rows = [
        row
        for row in mapping_rows
        if str(row["previous_inference_setup_id"])
        not in RETIRED_LEADERBOARD_MODEL_IDS
    ]
    mapping = pd.DataFrame(mapping_rows, columns=MODEL_NAME_MAPPING_COLUMNS)
    mapping_path = destination / "leaderboard_model_name_mapping.csv"
    temporary = mapping_path.with_name(f".{mapping_path.name}.tmp")
    mapping.to_csv(temporary, index=False)
    temporary.replace(mapping_path)
    paths["name_mapping"] = mapping_path

    metadata = {
        "schema_version": 5,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluations_root": str(Path(evaluations_root)),
        "latest_completed_run_per_inference_setup": latest_only,
        "latest_completed_run_per_model": latest_only,
        "deprecated_aliases": {
            "latest_completed_run_per_model": (
                "latest_completed_run_per_inference_setup"
            ),
            "columns": {
                "model_id": "inference_setup_id",
                "run_name": "run_id",
                "platform": "execution_target",
                "artifact_context": "model_artifact",
                "preprocessing_context": "inference_setup",
                "runtime_context": "execution_stack",
            },
        },
        "presentation_naming": {
            "format": registry["naming_format"],
            "registry": file_identity(MODEL_PRESENTATION_REGISTRY_PATH),
            "registered_inference_setups": len(
                set(registry["inference_setups"])
                - RETIRED_LEADERBOARD_MODEL_IDS
            ),
            "registered_models": len(
                set(registry["inference_setups"])
                - RETIRED_LEADERBOARD_MODEL_IDS
            ),
            "deprecated_aliases": {
                "registered_models": "registered_inference_setups",
                "mapping_columns.previous_model_id": (
                    "mapping_columns.previous_inference_setup_id"
                ),
                "mapping_columns.platform": "mapping_columns.execution_target",
            },
            "old_to_new_mapping": {
                "path": str(mapping_path),
                "rows": len(mapping),
            },
            "decoder_source": "embedded completed-run profile",
            "model_source": (
                "presentation registry keyed by stable inference_setup_id"
            ),
            "stable_identity_renamed": False,
        },
        "comparison_factors": {
            "order": [
                "model_artifact",
                "inference_setup",
                "execution_stack",
            ],
            "definitions": {
                "model_artifact": "Which checkpoint, export, or deployment artifact was selected.",
                "inference_setup": "Input processing, chunking, and decoding selected for the run.",
                "execution_stack": "Inference library and engine plus the observed execution environment; never inferred from the model artifact.",
            },
            "evidence_priority": [
                "observed completed-run environment and inference-adapter metadata",
                "pipeline contract embedded in completed run metadata",
                "contract recoverable from the embedded completed-run profile",
                "legacy profile/adapter fallback with unavailable facts stated explicitly",
            ],
        },
        "leaderboards": {
            namespace: {
                "path": str(paths[namespace]),
                "rows": len(frames[namespace]),
                "scoring_units": config["scoring_units"],
                "ranking_metric": config["error_metric"],
                "skipped": skipped[namespace],
            }
            for namespace, config in REPRESENTATIONS.items()
        },
    }
    metadata_path = destination / "leaderboard_metadata.json"
    temporary = metadata_path.with_name(f".{metadata_path.name}.tmp")
    temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(metadata_path)
    paths["metadata"] = metadata_path
    return paths, skipped


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build separate orthographic/WER and IPA/PER leaderboards."
    )
    parser.add_argument(
        "--evaluations_root",
        "--evaluations-root",
        default="input_output_data/output/evaluations",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        default="input_output_data/output/leaderboards",
    )
    parser.add_argument(
        "--all_runs",
        "--all-runs",
        action="store_true",
        help="Include every completed run instead of only the newest per model id.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths, skipped = write_leaderboards(
        args.evaluations_root,
        args.output_dir,
        latest_only=not args.all_runs,
    )
    for namespace in REPRESENTATIONS:
        print(f"{namespace}: {paths[namespace]}")
        for reason in skipped[namespace]:
            print(f"[SKIP] {reason}")
    print(f"name mapping: {paths['name_mapping']}")
    print(f"metadata: {paths['metadata']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
