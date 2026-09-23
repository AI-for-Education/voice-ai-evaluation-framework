"""Build representation-compatible ASR leaderboards from evaluation outputs."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from egra_eval2.reference.g2p import (
    canonical_identity_sha256,
    system_id_for_identity,
)
from egra_eval2.reference.ipa_inventory_maps import (
    IPAInventoryMapError,
    describe_ipa_inventory_route,
)
from inference.pipeline_provenance import inference_profile_from_run_metadata
from egra_eval2.leaderboard_context import (
    model_artifact_label,
    contract_references as _contract_references,
    decoding_label as _decoding_label,
    execution_stack_context as _execution_stack_context,
    execution_target_label as _execution_target_label,
    inference_engine_version as _inference_engine_version,
    input_processing_label,
)
from egra_eval2.leaderboard_policy import REQUIRED_ELIGIBILITY_POLICY
from egra_eval2.model_presentation import (
    MODEL_PRESENTATION_COLUMNS,
    MODEL_PRESENTATION_REGISTRY_PATH,
    build_presentation_rows,
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
    },
}

# These retired inference setups remain documented but do not appear in active
# leaderboards or stakeholder-report packages.
RETIRED_LEADERBOARD_INFERENCE_SETUP_IDS = frozenset(
    {
        "bookbot-orthographic-ctc",
        "bookbot-orthographic-ctc-5gram",
    }
)

_METRIC_RE = re.compile(
    r"^\s+([a-z][a-z0-9_]*):\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)|nan)%?\s*$",
    re.IGNORECASE,
)
_G2P_SYSTEM_ID_RE = re.compile(r"^[a-z0-9_-]+-[0-9a-f]{12}$")


class LeaderboardError(ValueError):
    """Raised when a leaderboard input cannot be parsed safely."""


def _g2p_display_name(tool_id: object, recorded_name: object) -> str:
    """Use the canonical public name while accepting historical metadata."""
    if tool_id == "babygruut":
        return "babygruut"
    return str(recorded_name or "")


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
    run_id: str,
) -> dict[str, Any]:
    transcript_metadata = (
        evaluations_root.parent / "transcripts" / run_id / "run_metadata.json"
    )
    return _load_json_object(transcript_metadata, "ASR run metadata")


def _hypothesis_route(namespace: str) -> str:
    if namespace == "orthographic":
        return "native orthographic"
    raise LeaderboardError("IPA hypothesis route must come from evaluation metadata")


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
    g2p_system_id: str | None = None,
) -> dict[str, Any]:
    config = REPRESENTATIONS[namespace]
    representation_dir = (
        run_dir / "ipa" / str(g2p_system_id)
        if namespace == "ipa"
        else run_dir / "orthographic"
    )
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
    output_metadata = run_metadata.get("output")
    if (
        isinstance(output_metadata, dict)
        and output_metadata.get("smoke_test") is True
    ):
        raise LeaderboardError(
            f"Smoke-test run cannot enter the leaderboard: {run_dir.name}"
        )
    profile = inference_profile_from_run_metadata(run_metadata)
    if not profile:
        metadata_path = (
            evaluations_root.parent
            / "transcripts"
            / run_dir.name
            / "run_metadata.json"
        )
        raise LeaderboardError(f"ASR run metadata has no profile: {metadata_path}")
    inference_setup_id = profile.get("inference_setup_id")
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

    g2p_metadata: dict[str, Any] | None = None
    hypothesis_route = _hypothesis_route(namespace) if namespace == "orthographic" else ""
    if namespace == "ipa":
        raw_g2p = evaluation_metadata.get("g2p_system")
        if not isinstance(raw_g2p, dict):
            raise LeaderboardError(
                f"IPA evaluation has no exact G2P identity: {representation_dir}"
            )
        g2p_metadata = raw_g2p
        identity = g2p_metadata.get("identity")
        if not isinstance(identity, dict):
            raise LeaderboardError(
                f"IPA evaluation has invalid G2P identity: {representation_dir}"
            )
        recorded_id = g2p_metadata.get("system_id")
        recorded_sha = g2p_metadata.get("identity_sha256")
        computed_id = system_id_for_identity(identity)
        computed_sha = canonical_identity_sha256(identity)
        identity_fields_match = all(
            g2p_metadata.get(field) == identity.get(field)
            for field in ("tool_id", "language", "inventory")
        )
        if (
            recorded_id != g2p_system_id
            or computed_id != g2p_system_id
            or recorded_sha != computed_sha
            or representation_dir.name != g2p_system_id
            or not identity_fields_match
        ):
            raise LeaderboardError(
                f"IPA evaluation G2P identity/path mismatch: {representation_dir}"
            )
        hypothesis_route = evaluation_metadata.get("hypothesis_route")
        if not isinstance(hypothesis_route, str) or not hypothesis_route.strip():
            raise LeaderboardError(
                f"IPA evaluation has no hypothesis route: {representation_dir}"
            )
        display_name = _g2p_display_name(
            g2p_metadata.get("tool_id"), g2p_metadata.get("display_name")
        )
        recorded_display_name = str(g2p_metadata.get("display_name", ""))
        if native_output_units == "phoneme":
            source_inventory = profile.get("output_inventory")
            target_inventory = g2p_metadata.get("inventory")
            if not isinstance(source_inventory, str) or not isinstance(
                target_inventory, str
            ):
                raise LeaderboardError(
                    f"Native IPA inventory is not recorded: {representation_dir}"
                )
            try:
                hypothesis_route = describe_ipa_inventory_route(
                    source_inventory.strip(), target_inventory.strip()
                )
            except IPAInventoryMapError as exc:
                raise LeaderboardError(str(exc)) from exc
        elif recorded_display_name and recorded_display_name != display_name:
            hypothesis_route = hypothesis_route.replace(
                recorded_display_name, display_name
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
    inference_library = profile.get("inference_library")
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
    model_artifact = model_artifact_label(profile, run_metadata, references)
    input_processing = input_processing_label(references)
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
        "native_output_units": native_output_units,
        "hypothesis_route": hypothesis_route,
        "completed_at": completed_at,
        "summary_path": str(summary_path),
    }
    if g2p_metadata is not None:
        row.update(
            {
                "g2p_tool": str(g2p_metadata.get("tool_id", "")),
                "g2p_system_id": str(g2p_metadata.get("system_id", "")),
                "g2p_display_name": display_name,
                "g2p_identity_sha256": str(
                    g2p_metadata.get("identity_sha256", "")
                ),
                "target_inventory": str(g2p_metadata.get("inventory", "")),
            }
        )
    row["model_label"] = structured_model_label(presentation, decoding)
    row.update(_flatten_summary(summary, error_metric))
    return row


def _columns_for(namespace: str) -> list[str]:
    metric = str(REPRESENTATIONS[namespace]["error_metric"])
    columns = [
        "rank",
        "model_label",
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
        "decoding",
        "context_evidence",
        "native_output_units",
        "hypothesis_route",
    ]
    if namespace == "ipa":
        columns.extend(
            [
                "g2p_tool",
                "g2p_system_id",
                "g2p_display_name",
                "g2p_identity_sha256",
                "target_inventory",
            ]
        )
    columns.extend([f"global_{metric}", "global_mer"])
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
    g2p_system_id: str | None = None,
    latest_only: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Discover valid completed runs for one scoring representation."""
    if namespace not in REPRESENTATIONS:
        raise ValueError(f"Unsupported leaderboard namespace: {namespace}")
    if namespace == "ipa" and not g2p_system_id:
        raise ValueError("IPA leaderboards require an exact g2p_system_id")
    if namespace == "orthographic" and g2p_system_id is not None:
        raise ValueError("Orthographic leaderboards do not use a G2P system")
    root = Path(evaluations_root)
    if not root.is_dir():
        raise LeaderboardError(f"Evaluations root not found: {root}")

    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for run_dir in sorted((path for path in root.iterdir() if path.is_dir())):
        representation_dir = (
            run_dir / "ipa" / str(g2p_system_id)
            if namespace == "ipa"
            else run_dir / "orthographic"
        )
        if not representation_dir.is_dir():
            continue
        try:
            row = _candidate_row(
                root, run_dir, namespace, g2p_system_id=g2p_system_id
            )
            if str(row["inference_setup_id"]) in RETIRED_LEADERBOARD_INFERENCE_SETUP_IDS:
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


def discover_g2p_system_ids(evaluations_root: str | Path) -> list[str]:
    """Discover exact G2P-system namespaces present below completed runs."""
    root = Path(evaluations_root)
    if not root.is_dir():
        raise LeaderboardError(f"Evaluations root not found: {root}")
    system_ids: set[str] = set()
    for run_dir in root.iterdir():
        ipa_dir = run_dir / "ipa"
        if not run_dir.is_dir() or not ipa_dir.is_dir():
            continue
        system_ids.update(
            path.name
            for path in ipa_dir.iterdir()
            if path.is_dir() and _G2P_SYSTEM_ID_RE.fullmatch(path.name)
        )
    return sorted(system_ids)


def build_leaderboards(
    evaluations_root: str | Path,
    *,
    latest_only: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build orthographic plus one isolated IPA board per exact G2P system."""
    orthographic, orthographic_skipped = build_leaderboard(
        evaluations_root, "orthographic", latest_only=latest_only
    )
    ipa_by_system: dict[str, pd.DataFrame] = {}
    ipa_skipped_by_system: dict[str, list[str]] = {}
    for system_id in discover_g2p_system_ids(evaluations_root):
        ipa_by_system[system_id], ipa_skipped_by_system[system_id] = (
            build_leaderboard(
                evaluations_root,
                "ipa",
                g2p_system_id=system_id,
                latest_only=latest_only,
            )
        )
    frames: dict[str, Any] = {
        "orthographic": orthographic,
        "ipa_by_system": ipa_by_system,
    }
    skipped: dict[str, Any] = {
        "orthographic": orthographic_skipped,
        "ipa_by_system": ipa_skipped_by_system,
    }
    return frames, skipped


def _latest_evaluated_rows(
    frames: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Index the newest row per inference setup across namespaces."""
    source_frames = [frames["orthographic"], *frames["ipa_by_system"].values()]
    evaluated = pd.concat(source_frames, ignore_index=True)
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
    """Resolve decoder and execution-target labels for registered profiles."""
    from inference.profile import load_profile

    repository = Path(__file__).resolve().parents[1]
    contexts: dict[str, dict[str, str]] = {}
    profile_paths = sorted((repository / "inference").glob("*/profiles/*.yaml"))
    for path in profile_paths:
        profile = load_profile(path).to_dict()
        setup_id = str(profile["inference_setup_id"])
        if setup_id in RETIRED_LEADERBOARD_INFERENCE_SETUP_IDS:
            continue
        references, _ = _contract_references(profile, {})
        contexts[setup_id] = {
            "decoder": _decoding_label(profile),
            "execution_target": _execution_target_label(profile, {}, references),
        }
    return contexts


def write_leaderboards(
    evaluations_root: str | Path,
    output_dir: str | Path,
    *,
    latest_only: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write orthographic and exact-system-isolated IPA leaderboard CSVs."""
    from inference.provenance import file_identity

    frames, skipped = build_leaderboards(
        evaluations_root, latest_only=latest_only
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Any] = {}
    path = destination / str(REPRESENTATIONS["orthographic"]["filename"])
    frame = frames["orthographic"]
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, float_format="%.4f")
    temporary.replace(path)
    paths["orthographic"] = path

    ipa_paths: dict[str, Path] = {}
    ipa_destination = destination / "ipa"
    if frames["ipa_by_system"]:
        ipa_destination.mkdir(parents=True, exist_ok=True)
    for system_id, frame in frames["ipa_by_system"].items():
        path = ipa_destination / f"leaderboard_{system_id}.csv"
        temporary = path.with_name(f".{path.name}.tmp")
        frame.to_csv(temporary, index=False, float_format="%.4f")
        temporary.replace(path)
        ipa_paths[system_id] = path
    paths["ipa_by_system"] = ipa_paths

    registry = default_model_presentation_registry()
    presentation_rows = build_presentation_rows(
        _latest_evaluated_rows(frames),
        _profile_presentation_contexts(),
    )
    presentation_rows = [
        row
        for row in presentation_rows
        if str(row["inference_setup_id"])
        not in RETIRED_LEADERBOARD_INFERENCE_SETUP_IDS
    ]
    presentation_table = pd.DataFrame(
        presentation_rows, columns=MODEL_PRESENTATION_COLUMNS
    )
    presentation_path = destination / "leaderboard_model_presentation.csv"
    temporary = presentation_path.with_name(f".{presentation_path.name}.tmp")
    presentation_table.to_csv(temporary, index=False)
    temporary.replace(presentation_path)
    paths["presentation"] = presentation_path

    metadata = {
        "schema_version": 6,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluations_root": str(Path(evaluations_root)),
        "latest_completed_run_per_inference_setup": latest_only,
        "eligibility_policy": dict(REQUIRED_ELIGIBILITY_POLICY),
        "presentation_naming": {
            "format": registry["naming_format"],
            "registry": file_identity(MODEL_PRESENTATION_REGISTRY_PATH),
            "registered_inference_setups": len(
                set(registry["inference_setups"])
                - RETIRED_LEADERBOARD_INFERENCE_SETUP_IDS
            ),
            "presentation_table": {
                "path": str(presentation_path),
                "rows": len(presentation_table),
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
                "explicit not-available values when current metadata lacks a fact",
            ],
        },
        "leaderboards": {
            "orthographic": {
                "path": str(paths["orthographic"]),
                "rows": len(frames["orthographic"]),
                "scoring_units": "orthographic",
                "ranking_metric": "wer",
                "skipped": skipped["orthographic"],
            },
            "ipa_by_system": {
                system_id: {
                    "path": str(paths["ipa_by_system"][system_id]),
                    "rows": len(frame),
                    "scoring_units": "phoneme",
                    "ranking_metric": "per",
                    "g2p_system": (
                        frame.iloc[0][
                            [
                                "g2p_tool",
                                "g2p_system_id",
                                "g2p_display_name",
                                "g2p_identity_sha256",
                                "target_inventory",
                            ]
                        ].to_dict()
                        if not frame.empty
                        else {"g2p_system_id": system_id}
                    ),
                    "skipped": skipped["ipa_by_system"][system_id],
                }
                for system_id, frame in frames["ipa_by_system"].items()
            },
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
        help=(
            "Include every completed run instead of only the newest per "
            "inference setup ID."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths, skipped = write_leaderboards(
        args.evaluations_root,
        args.output_dir,
        latest_only=not args.all_runs,
    )
    print(f"orthographic: {paths['orthographic']}")
    for reason in skipped["orthographic"]:
        print(f"[SKIP] {reason}")
    for system_id, path in paths["ipa_by_system"].items():
        print(f"ipa/{system_id}: {path}")
        for reason in skipped["ipa_by_system"][system_id]:
            print(f"[SKIP] {reason}")
    print(f"model presentation: {paths['presentation']}")
    print(f"metadata: {paths['metadata']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
