"""Build a local-development onboarding leaderboard status and data snapshot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from egra_eval2.leaderboard_policy import has_required_eligibility_policy


class SnapshotError(ValueError):
    """Raised when leaderboard inputs cannot produce a trustworthy snapshot."""


FILES = {
    "orthographic": "leaderboard_orthographic.csv",
    "presentation": "leaderboard_model_presentation.csv",
    "metadata": "leaderboard_metadata.json",
}

METRIC_BY_NAMESPACE = {"orthographic": "wer", "ipa": "per"}
WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
PORTABLE_MARKERS = (
    "docs/",
    "egra_eval2/",
    "inference/",
    "input_output_data/",
    "tests/",
    "tools/",
)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"Invalid JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise SnapshotError(f"Expected a JSON object: {path}")
    return payload


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise SnapshotError(f"CSV has no header: {path}")
            return list(reader.fieldnames), list(reader)
    except OSError as exc:
        raise SnapshotError(f"Could not read CSV file: {path}") from exc


def _require_columns(path: Path, columns: Sequence[str], required: set[str]) -> None:
    missing = sorted(required.difference(columns))
    if missing:
        raise SnapshotError(f"CSV is missing {missing}: {path}")


def _validate_ranks(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    expected = [str(index) for index in range(1, len(rows) + 1)]
    observed = [row.get("rank", "") for row in rows]
    if observed != expected:
        raise SnapshotError(f"CSV ranks are not contiguous and ordered: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _install_staged_file(staged: Path, destination: Path) -> None:
    """Install a validated staged file, with a Windows ACL-safe fallback."""
    # Replacing an existing file on Windows can also replace its ACL with the
    # temporary file's ACL. Overwrite its contents so checkout permissions stay.
    if os.name == "nt" and destination.exists():
        shutil.copyfile(staged, destination)
        return

    try:
        staged.replace(destination)
    except PermissionError:
        # Some managed Windows workspaces allow file writes but deny rename over
        # an existing file. Validation and staging are already complete here.
        shutil.copyfile(staged, destination)


def _portable_path(value: str) -> str:
    if value.startswith(("http://", "https://")):
        return value
    normalized = value.replace("\\", "/")
    is_absolute = WINDOWS_ABSOLUTE.match(value) is not None or normalized.startswith(
        "/"
    )
    if not is_absolute:
        return normalized
    for marker in PORTABLE_MARKERS:
        index = normalized.find(marker)
        if index >= 0:
            return normalized[index:]
    return "not_shared:absolute_path_removed"


def _portable_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _portable_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_metadata(item) for item in value]
    if isinstance(value, str):
        return _portable_path(value)
    return value


def _markdown(value: Any) -> str:
    text = str(value or "").replace("\n", " ").replace("\r", " ")
    return text.replace("|", "\\|")


def _metric(value: str) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "not available"
    return f"{number:.2f}%"


def _top_table(rows: Sequence[Mapping[str, str]], metric: str) -> list[str]:
    if not rows:
        return ["No eligible completed results were found."]
    lines = [
        f"| Rank | Model | Global {metric.upper()} | Global MER | HYP route |",
        "|---:|---|---:|---:|---|",
    ]
    for row in rows[:5]:
        lines.append(
            "| {rank} | {model} | {score} | {mer} | {route} |".format(
                rank=_markdown(row.get("rank")),
                model=_markdown(row.get("model_label")),
                score=_metric(row.get(f"global_{metric}", "")),
                mer=_metric(row.get("global_mer", "")),
                route=_markdown(row.get("hypothesis_route")),
            )
        )
    return lines


def _skipped(metadata: Mapping[str, Any], namespace: str) -> list[str]:
    leaderboards = metadata.get("leaderboards")
    if not isinstance(leaderboards, dict):
        return []
    details = leaderboards.get(namespace)
    if not isinstance(details, dict):
        return []
    skipped = details.get("skipped")
    return [str(item) for item in skipped] if isinstance(skipped, list) else []


def _render_status(
    metadata: Mapping[str, Any],
    orthographic: Sequence[Mapping[str, str]],
    ipa_by_system: Mapping[str, Sequence[Mapping[str, str]]],
    presentation: Sequence[Mapping[str, str]],
) -> str:
    statuses: dict[str, int] = {}
    for row in presentation:
        status = row.get("leaderboard_status", "unknown") or "unknown"
        statuses[status] = statuses.get(status, 0) + 1
    profile_only = [
        row
        for row in presentation
        if row.get("leaderboard_status") == "profile_only"
    ]
    skipped_orthographic = _skipped(metadata, "orthographic")
    leaderboard_metadata = metadata.get("leaderboards") or {}
    raw_ipa_metadata = leaderboard_metadata.get("ipa_by_system") or {}
    skipped_ipa = {
        system_id: [str(item) for item in details.get("skipped", [])]
        for system_id, details in raw_ipa_metadata.items()
        if isinstance(details, dict)
    }
    ipa_rows = sum(len(rows) for rows in ipa_by_system.values())
    generated = _markdown(metadata.get("generated_at", "not available"))
    latest_only = metadata.get("latest_completed_run_per_inference_setup") is True

    lines = [
        "# Current leaderboard status",
        "",
        "> Generated by `run_refresh_onboarding.sh`. Do not edit this file by hand.",
        "",
        f"**Leaderboard generated at:** `{generated}`",
        "",
        "## Coverage",
        "",
        "| Item | Current count |",
        "|---|---:|",
        f"| Registered inference setups | {len(presentation)} |",
        f"| Evaluated setups | {statuses.get('evaluated', 0)} |",
        f"| Profile-only setups | {statuses.get('profile_only', 0)} |",
        f"| Eligible orthographic/WER rows | {len(orthographic)} |",
        f"| Exact G2P systems with IPA/PER boards | {len(ipa_by_system)} |",
        f"| Eligible IPA/PER rows across isolated boards | {ipa_rows} |",
        f"| Skipped orthographic results | {len(skipped_orthographic)} |",
        f"| Skipped IPA results | {sum(map(len, skipped_ipa.values()))} |",
        "",
        (
            "The generated view keeps the newest completed run for each stable "
            f"inference setup ID: **{'yes' if latest_only else 'no'}**."
        ),
        "",
        "## Leading orthographic results",
        "",
        (
            "Lower global WER is better. Only compatible written-text "
            "evaluations appear here."
        ),
        "",
        *_top_table(orthographic, "wer"),
        "",
    ]
    if ipa_by_system:
        for system_id, ipa_rows_for_system in ipa_by_system.items():
            system_details = raw_ipa_metadata.get(system_id, {})
            system_identity = system_details.get("g2p_system", {})
            system_name = system_identity.get("g2p_display_name", system_id)
            inventory = system_identity.get("target_inventory", "not available")
            lines.extend(
                [
                    f"## Leading IPA results — {_markdown(system_name)}",
                    "",
                    f"Exact system ID: `{_markdown(system_id)}`; target inventory: "
                    f"`{_markdown(inventory)}`. Lower global PER is better. This "
                    "ranking is not combined with any other G2P system.",
                    "",
                    *_top_table(ipa_rows_for_system, "per"),
                    "",
                ]
            )
    else:
        lines.extend(["## Leading IPA results", "", "No exact-system IPA boards found.", ""])
    lines.extend(["## Registered but not yet evaluated", ""])
    if profile_only:
        for row in profile_only:
            name = row.get("presentation_name") or row.get("inference_setup_id")
            lines.append(f"- {_markdown(name)}")
    else:
        lines.append("All registered profiles currently have an eligible evaluation.")

    lines.extend(["", "## Skipped-result status", ""])
    if not skipped_orthographic and not any(skipped_ipa.values()):
        lines.append(
            "No discovered completed result was skipped in either leaderboard."
        )
    else:
        for namespace, items in [("Orthographic", skipped_orthographic), *[
            (f"IPA / {system_id}", items)
            for system_id, items in skipped_ipa.items()
        ]]:
            for item in items:
                lines.append(f"- **{namespace}:** {_markdown(item)}")

    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            (
                "- WER and PER are different representations and are never "
                "combined into one rank."
            ),
            (
                "- PER results produced by different exact G2P systems are never "
                "combined into one rank."
            ),
            (
                "- Rankings describe this benchmark dataset and the recorded "
                "model artifact, inference setup, and execution stack."
            ),
            (
                "- A PC-executed Android-behaviour proxy does not measure physical-"
                "phone latency, memory, battery use, or thermal behavior."
            ),
            "- Smoke-test results are not eligible evidence of full benchmark quality.",
            "",
            "## Attached data",
            "",
            "- `data/leaderboard_orthographic.csv` — complete ranked WER rows.",
            "- `data/ipa/leaderboard_<g2p_system_id>.csv` — one complete PER "
            "ranking per exact G2P system.",
            (
                "- `data/leaderboard_model_presentation.csv` — registered and "
                "profile-only inference setups."
            ),
            (
                "- `data/leaderboard_metadata.json` — portable generation metadata "
                "and source snapshot hash."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def build_snapshot(leaderboards_root: Path, package_root: Path) -> dict[str, Any]:
    """Validate leaderboard outputs and atomically refresh the package snapshot."""
    source_paths = {
        name: leaderboards_root / filename for name, filename in FILES.items()
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise SnapshotError("Missing leaderboard files: " + ", ".join(missing))

    metadata = _read_json_object(source_paths["metadata"])
    generated_at = metadata.get("generated_at")
    if not isinstance(generated_at, str) or not generated_at.strip():
        raise SnapshotError("Leaderboard metadata has no generated_at timestamp")
    if not has_required_eligibility_policy(metadata):
        raise SnapshotError(
            "Leaderboard metadata does not guarantee the required eligibility policy"
        )

    rows: dict[str, list[dict[str, str]]] = {}
    headers: dict[str, list[str]] = {}
    for namespace in ("orthographic",):
        path = source_paths[namespace]
        headers[namespace], rows[namespace] = _read_csv(path)
        if "evaluation_status" in headers[namespace]:
            raise SnapshotError(f"CSV contains obsolete run-status data: {path}")
        metric = METRIC_BY_NAMESPACE[namespace]
        _require_columns(
            path,
            headers[namespace],
            {
                "rank",
                "inference_setup_id",
                "model_label",
                f"global_{metric}",
                "global_mer",
                "hypothesis_route",
            },
        )
        _validate_ranks(path, rows[namespace])
        board_metadata = (metadata.get("leaderboards") or {}).get(namespace)
        if not isinstance(board_metadata, dict):
            raise SnapshotError(f"Missing {namespace} leaderboard metadata")
        if board_metadata.get("rows") != len(rows[namespace]):
            raise SnapshotError(f"{namespace} CSV row count disagrees with metadata")
        if board_metadata.get("ranking_metric") != metric:
            raise SnapshotError(f"Unexpected {namespace} ranking metric")

    leaderboards_metadata = metadata.get("leaderboards")
    if not isinstance(leaderboards_metadata, dict):
        raise SnapshotError("Missing leaderboard metadata")
    ipa_metadata = leaderboards_metadata.get("ipa_by_system")
    if not isinstance(ipa_metadata, dict):
        raise SnapshotError("Missing ipa_by_system leaderboard metadata")
    ipa_paths: dict[str, Path] = {}
    ipa_rows: dict[str, list[dict[str, str]]] = {}
    for system_id, board_metadata in sorted(ipa_metadata.items()):
        if not isinstance(system_id, str) or not re.fullmatch(r"[a-z0-9_-]+", system_id):
            raise SnapshotError(f"Unsafe G2P system ID in metadata: {system_id!r}")
        if not isinstance(board_metadata, dict):
            raise SnapshotError(f"Invalid IPA leaderboard metadata: {system_id}")
        path = leaderboards_root / "ipa" / f"leaderboard_{system_id}.csv"
        if not path.is_file():
            raise SnapshotError(f"Missing leaderboard file: {path}")
        columns, system_rows = _read_csv(path)
        if "evaluation_status" in columns:
            raise SnapshotError(f"CSV contains obsolete run-status data: {path}")
        _require_columns(
            path,
            columns,
            {
                "rank",
                "inference_setup_id",
                "model_label",
                "global_per",
                "global_mer",
                "hypothesis_route",
                "g2p_system_id",
            },
        )
        _validate_ranks(path, system_rows)
        if any(row.get("g2p_system_id") != system_id for row in system_rows):
            raise SnapshotError(f"IPA CSV mixes G2P systems: {path}")
        if board_metadata.get("rows") != len(system_rows):
            raise SnapshotError(f"IPA CSV row count disagrees with metadata: {system_id}")
        if board_metadata.get("ranking_metric") != "per":
            raise SnapshotError(f"Unexpected IPA ranking metric: {system_id}")
        ipa_paths[system_id] = path
        ipa_rows[system_id] = system_rows

    presentation_header, presentation_rows = _read_csv(source_paths["presentation"])
    _require_columns(
        source_paths["presentation"],
        presentation_header,
        {"inference_setup_id", "presentation_name", "leaderboard_status"},
    )
    declared_presentation_rows = (
        (metadata.get("presentation_naming") or {}).get("presentation_table") or {}
    ).get("rows")
    if declared_presentation_rows != len(presentation_rows):
        raise SnapshotError("Model presentation row count disagrees with metadata")

    portable = _portable_metadata(metadata)
    portable["package_snapshot"] = {
        "source_file": FILES["metadata"],
        "source_sha256": _sha256(source_paths["metadata"]),
        "portable_paths": True,
    }
    status = _render_status(
        metadata, rows["orthographic"], ipa_rows, presentation_rows
    )

    package_root.mkdir(parents=True, exist_ok=True)
    data_root = package_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="onboarding-snapshot-", dir=package_root
    ) as temporary:
        staging = Path(temporary)
        staged_status = staging / "05-current-leaderboard-status.md"
        staged_status.write_text(status, encoding="utf-8", newline="\n")
        for namespace in ("orthographic", "presentation"):
            shutil.copyfile(source_paths[namespace], staging / FILES[namespace])
        staged_ipa = staging / "ipa"
        staged_ipa.mkdir()
        for system_id, source in ipa_paths.items():
            shutil.copyfile(source, staged_ipa / source.name)
        staged_metadata = staging / FILES["metadata"]
        staged_metadata.write_text(
            json.dumps(portable, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        _install_staged_file(staged_status, package_root / staged_status.name)
        for name in ("orthographic", "presentation", "metadata"):
            _install_staged_file(staging / FILES[name], data_root / FILES[name])
        data_ipa = data_root / "ipa"
        data_ipa.mkdir(exist_ok=True)
        expected_ipa_names = {source.name for source in ipa_paths.values()}
        for existing in data_ipa.glob("leaderboard_*.csv"):
            if existing.name not in expected_ipa_names:
                existing.unlink()
        for system_id, source in ipa_paths.items():
            _install_staged_file(staged_ipa / source.name, data_ipa / source.name)

    return {
        "generated_at": generated_at,
        "registered_inference_setups": len(presentation_rows),
        "evaluated_inference_setups": sum(
            row.get("leaderboard_status") == "evaluated"
            for row in presentation_rows
        ),
        "orthographic_rows": len(rows["orthographic"]),
        "ipa_rows": sum(len(system_rows) for system_rows in ipa_rows.values()),
        "ipa_systems": len(ipa_rows),
        "package_root": str(package_root),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh the shareable onboarding leaderboard snapshot."
    )
    parser.add_argument(
        "--leaderboards-root",
        default="input_output_data/output/leaderboards",
    )
    parser.add_argument(
        "--package-root",
        default="docs/shared-onboarding",
        help="Local-development output directory (Git-ignored by default).",
    )
    args = parser.parse_args()
    result = build_snapshot(Path(args.leaderboards_root), Path(args.package_root))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
