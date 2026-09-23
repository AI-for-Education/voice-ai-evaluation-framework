from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest

from tools.build_onboarding_leaderboard_snapshot import (
    SnapshotError,
    build_snapshot,
)


SYSTEM_ID = "africa_g2p-swh-fixture123456"


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _leaderboard_row(
    rank: int, inference_setup_id: str, metric: str
) -> dict[str, str]:
    row = {
        "rank": str(rank),
        "inference_setup_id": inference_setup_id,
        "model_label": f"Group · {inference_setup_id} · variant (greedy)",
        f"global_{metric}": "12.3400",
        "global_mer": "8.9000",
        "hypothesis_route": "native orthographic",
    }
    if metric == "per":
        row["g2p_system_id"] = SYSTEM_ID
    return row


def _fixture(
    root: Path,
    *,
    orthographic_rows: list[dict[str, str]] | None = None,
    ipa_rows: list[dict[str, str]] | None = None,
    skipped: list[str] | None = None,
) -> None:
    orthographic_rows = orthographic_rows or []
    ipa_rows = ipa_rows or []
    skipped = skipped or []
    common = [
        "rank",
        "inference_setup_id",
        "model_label",
        "global_mer",
        "hypothesis_route",
    ]
    _write_csv(
        root / "leaderboard_orthographic.csv",
        common + ["global_wer"],
        orthographic_rows,
    )
    if ipa_rows:
        _write_csv(
            root / "ipa" / f"leaderboard_{SYSTEM_ID}.csv",
            common + ["global_per", "g2p_system_id"],
            ipa_rows,
        )
    presentation = [
        {
            "inference_setup_id": "evaluated-model",
            "presentation_name": "Group · evaluated model · variant (greedy)",
            "leaderboard_status": "evaluated",
        },
        {
            "inference_setup_id": "future-model",
            "presentation_name": "Group · future model · variant (greedy)",
            "leaderboard_status": "profile_only",
        },
    ]
    _write_csv(
        root / "leaderboard_model_presentation.csv",
        ["inference_setup_id", "presentation_name", "leaderboard_status"],
        presentation,
    )
    metadata = {
        "schema_version": 6,
        "generated_at": "2026-08-19T10:37:58+00:00",
        "latest_completed_run_per_inference_setup": True,
        "eligibility_policy": {
            "completed_evaluations_only": True,
            "representation_compatible_only": True,
            "smoke_tests_excluded": True,
        },
        "presentation_naming": {
            "registry": {
                "path": "D:\\private\\repo\\egra_eval2\\model_presentation.json"
            },
            "presentation_table": {"rows": len(presentation)},
        },
        "leaderboards": {
            "orthographic": {
                "rows": len(orthographic_rows),
                "ranking_metric": "wer",
                "skipped": skipped,
            },
            "ipa_by_system": (
                {
                    SYSTEM_ID: {
                        "rows": len(ipa_rows),
                        "ranking_metric": "per",
                        "skipped": [],
                        "g2p_system": {
                            "g2p_display_name": "Africa G2P",
                            "target_inventory": "africa_g2p_swh_ipa_v1",
                        },
                    }
                }
                if ipa_rows
                else {}
            ),
        },
    }
    (root / "leaderboard_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )


def test_build_snapshot_writes_portable_status_and_exact_csvs(tmp_path: Path) -> None:
    source = tmp_path / "leaderboards"
    package = tmp_path / "package"
    orthographic = [_leaderboard_row(1, "evaluated-model", "wer")]
    ipa = [_leaderboard_row(1, "evaluated-model", "per")]
    _fixture(source, orthographic_rows=orthographic, ipa_rows=ipa)

    result = build_snapshot(source, package)

    assert result["registered_inference_setups"] == 2
    assert result["evaluated_inference_setups"] == 1
    status = (package / "05-current-leaderboard-status.md").read_text(encoding="utf-8")
    assert "Eligible orthographic/WER rows | 1" in status
    assert "Group · future model · variant (greedy)" in status
    assert "12.34%" in status
    for filename in (
        "leaderboard_orthographic.csv",
        "leaderboard_model_presentation.csv",
    ):
        assert (package / "data" / filename).read_bytes() == (
            source / filename
        ).read_bytes()
    ipa_filename = f"leaderboard_{SYSTEM_ID}.csv"
    assert (package / "data" / "ipa" / ipa_filename).read_bytes() == (
        source / "ipa" / ipa_filename
    ).read_bytes()

    packaged_metadata = json.loads(
        (package / "data" / "leaderboard_metadata.json").read_text(encoding="utf-8")
    )
    assert packaged_metadata["presentation_naming"]["registry"]["path"] == (
        "egra_eval2/model_presentation.json"
    )
    assert packaged_metadata["package_snapshot"]["portable_paths"] is True
    assert re.search(r"[A-Za-z]:[\\/]", json.dumps(packaged_metadata)) is None


def test_empty_and_skipped_status_is_explicit(tmp_path: Path) -> None:
    source = tmp_path / "leaderboards"
    package = tmp_path / "package"
    _fixture(source, skipped=["incompatible example"])

    build_snapshot(source, package)

    status = (package / "05-current-leaderboard-status.md").read_text(encoding="utf-8")
    assert status.count("No eligible completed results were found.") == 1
    assert "No exact-system IPA boards found." in status
    assert "Skipped orthographic results | 1" in status
    assert "incompatible example" in status


def test_snapshot_falls_back_when_windows_denies_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "leaderboards"
    package = tmp_path / "package"
    _fixture(
        source,
        orthographic_rows=[_leaderboard_row(1, "evaluated-model", "wer")],
        ipa_rows=[_leaderboard_row(1, "evaluated-model", "per")],
    )
    package.mkdir()
    status = package / "05-current-leaderboard-status.md"
    status.write_text("old snapshot", encoding="utf-8")

    def deny_replace(self: Path, target: Path) -> None:
        raise PermissionError

    monkeypatch.setattr(Path, "replace", deny_replace)

    build_snapshot(source, package)

    assert "old snapshot" not in status.read_text(encoding="utf-8")
    assert (
        package / "data" / "ipa" / f"leaderboard_{SYSTEM_ID}.csv"
    ).is_file()


@pytest.mark.parametrize("failure", ["missing", "malformed", "row_mismatch"])
def test_invalid_inputs_do_not_replace_existing_status(
    tmp_path: Path, failure: str
) -> None:
    source = tmp_path / "leaderboards"
    package = tmp_path / "package"
    _fixture(
        source,
        orthographic_rows=[_leaderboard_row(1, "evaluated-model", "wer")],
        ipa_rows=[_leaderboard_row(1, "evaluated-model", "per")],
    )
    package.mkdir()
    status = package / "05-current-leaderboard-status.md"
    status.write_text("existing snapshot", encoding="utf-8")

    if failure == "missing":
        (
            source / "ipa" / f"leaderboard_{SYSTEM_ID}.csv"
        ).unlink()
    elif failure == "malformed":
        (source / "leaderboard_metadata.json").write_text("{", encoding="utf-8")
    else:
        metadata_path = source / "leaderboard_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["leaderboards"]["ipa_by_system"][SYSTEM_ID]["rows"] = 2
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(SnapshotError):
        build_snapshot(source, package)
    assert status.read_text(encoding="utf-8") == "existing snapshot"


def test_missing_eligibility_policy_does_not_replace_existing_status(
    tmp_path: Path,
) -> None:
    source = tmp_path / "leaderboards"
    package = tmp_path / "package"
    _fixture(
        source,
        orthographic_rows=[_leaderboard_row(1, "evaluated-model", "wer")],
    )
    package.mkdir()
    status = package / "05-current-leaderboard-status.md"
    status.write_text("existing snapshot", encoding="utf-8")
    metadata_path = source / "leaderboard_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("eligibility_policy")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(SnapshotError, match="eligibility policy"):
        build_snapshot(source, package)
    assert status.read_text(encoding="utf-8") == "existing snapshot"


def test_obsolete_run_status_data_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "leaderboards"
    package = tmp_path / "package"
    row = _leaderboard_row(1, "evaluated-model", "wer")
    _fixture(source, orthographic_rows=[row])
    row["evaluation_status"] = "scored (full run, 5,143 items)"
    orthographic_path = source / "leaderboard_orthographic.csv"
    _write_csv(
        orthographic_path,
        list(row),
        [row],
    )

    with pytest.raises(SnapshotError, match="obsolete run-status data"):
        build_snapshot(source, package)
    assert not package.exists()


def test_local_onboarding_package_is_self_contained_when_present() -> None:
    # Optional local-development artifact; docs/* is ignored except vocabulary.md.
    package = Path("docs/shared-onboarding")
    if not package.is_dir():
        pytest.skip("The generated local onboarding package is not present")
    link_pattern = re.compile(r"\[[^]]+\]\(([^)]+)\)")
    absolute_pattern = re.compile(r"(?<![A-Za-z])(?:[A-Za-z]:[\\/]|/d/data/|/work/)")

    for path in sorted(package.rglob("*")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8-sig")
        assert absolute_pattern.search(text) is None, path
        if path.suffix != ".md":
            continue
        for match in link_pattern.finditer(text):
            target = match.group(1)
            if target.startswith(("#", "https://", "http://")):
                continue
            relative = target.split("#", 1)[0]
            assert not relative.startswith("../"), (path, target)
            assert (path.parent / relative).exists(), (path, target)
