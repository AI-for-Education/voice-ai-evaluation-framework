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


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _leaderboard_row(
    rank: int, inference_setup_id: str, metric: str
) -> dict[str, str]:
    return {
        "rank": str(rank),
        "inference_setup_id": inference_setup_id,
        "model_label": f"Group · {inference_setup_id} · variant (greedy)",
        f"global_{metric}": "12.3400",
        "global_mer": "8.9000",
        "hypothesis_route": "native orthographic",
    }


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
    _write_csv(root / "leaderboard_ipa.csv", common + ["global_per"], ipa_rows)
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
        "schema_version": 4,
        "generated_at": "2026-08-19T10:37:58+00:00",
        "latest_completed_run_per_inference_setup": True,
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
            "ipa": {
                "rows": len(ipa_rows),
                "ranking_metric": "per",
                "skipped": [],
            },
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
        "leaderboard_ipa.csv",
        "leaderboard_model_presentation.csv",
    ):
        assert (package / "data" / filename).read_bytes() == (
            source / filename
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
    assert status.count("No eligible completed results were found.") == 2
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
    assert (package / "data" / "leaderboard_ipa.csv").is_file()


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
        (source / "leaderboard_ipa.csv").unlink()
    elif failure == "malformed":
        (source / "leaderboard_metadata.json").write_text("{", encoding="utf-8")
    else:
        metadata_path = source / "leaderboard_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["leaderboards"]["ipa"]["rows"] = 2
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(SnapshotError):
        build_snapshot(source, package)
    assert status.read_text(encoding="utf-8") == "existing snapshot"


def test_local_onboarding_package_is_self_contained_when_present() -> None:
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
