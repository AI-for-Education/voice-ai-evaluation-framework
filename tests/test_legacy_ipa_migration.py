from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from egra_eval2.reference.g2p import make_test_system
from tools.migrations import migrate_legacy_ipa as migration


def _system():
    return make_test_system(
        phonemize=lambda values: [f"ipa {value}" for value in values],
        version_value="migration-test",
    )


def _legacy_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "output" / "evaluations"
    run = root / "run-one"
    legacy = run / "ipa"
    manifests = run / "manifests"
    legacy.mkdir(parents=True)
    manifests.mkdir()
    source = manifests / "ref_manifest.clean.jsonl"
    source.write_text(
        json.dumps(
            {
                "audio_filepath": "audio.wav",
                "can_text": "can",
                "ref_text": "ref",
                "pred_text": "hyp",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    aligned = manifests / "ref_manifest.clean.ipa_aligned.jsonl"
    aligned.write_text("{}\n", encoding="utf-8")
    (legacy / "evaluation_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "completed_at": "2026-01-02T00:00:00+00:00",
                "source_manifest": str(source),
            }
        ),
        encoding="utf-8",
    )
    (legacy / "egra_eval_summary.txt").write_text(
        "GLOBAL\n  per: 0.00%\n", encoding="utf-8"
    )
    (legacy / "egra_eval_detailed.csv").write_text("score\n", encoding="utf-8")
    return root, run, source


def _verified(run: Path, source: Path) -> migration.VerifiedLegacyEvaluation:
    return migration.VerifiedLegacyEvaluation(
        run_dir=run,
        legacy_dir=run / "ipa",
        source_manifest=source,
        aligned_manifest=source.with_name("ref_manifest.clean.ipa_aligned.jsonl"),
        rows=(
            {
                "audio_filepath": "audio.wav",
                "can_text": "ipa can",
                "ref_text": "ipa ref",
                "pred_text": "ipa hyp",
            },
        ),
        hypothesis_route="orthographic -> Africa G2P (test)",
        evaluation_metadata={
            "schema_version": 2,
            "completed_at": "2026-01-02T00:00:00+00:00",
        },
    )


def test_discovery_ignores_nested_exact_system_and_archives(tmp_path: Path) -> None:
    root, run, _ = _legacy_fixture(tmp_path)
    nested = run / "ipa" / "africa_g2p-swh-id"
    nested.mkdir()
    (nested / "evaluation_metadata.json").write_text("{}", encoding="utf-8")
    archive = run / "pre_reference_fix" / "ipa"
    archive.mkdir(parents=True)
    (archive / "evaluation_metadata.json").write_text("{}", encoding="utf-8")
    (archive / "egra_eval_summary.txt").write_text("GLOBAL\n", encoding="utf-8")

    assert migration.discover_legacy_evaluations(root) == [run]


def test_dry_run_never_copies_verified_legacy_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, run, source = _legacy_fixture(tmp_path)
    system = _system()
    monkeypatch.setattr(
        migration,
        "verify_legacy_evaluation",
        lambda *args, **kwargs: _verified(run, source),
    )

    report = migration.migrate_legacy_ipa(
        evaluations_root=root,
        dataset_root=tmp_path,
        apply=False,
        system=system,
    )

    assert report["mode"] == "dry-run"
    assert report["verified"] == 1
    assert report["migrated"] == 0
    assert not (run / "ipa" / system.system_id).exists()


def test_apply_is_idempotent_and_preserves_generic_legacy_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, run, source = _legacy_fixture(tmp_path)
    system = _system()
    reference = tmp_path / "reference.jsonl"
    reference.write_text("{}\n", encoding="utf-8")
    reference_metadata = tmp_path / "reference.metadata.json"
    reference_metadata.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        migration,
        "verify_legacy_evaluation",
        lambda *args, **kwargs: _verified(run, source),
    )
    monkeypatch.setattr(
        migration,
        "build_reference_views",
        lambda **kwargs: SimpleNamespace(
            ipa=reference, metadata=reference_metadata
        ),
    )

    first = migration.migrate_legacy_ipa(
        evaluations_root=root,
        dataset_root=tmp_path,
        reference_manifest=source,
        apply=True,
        system=system,
    )
    second = migration.migrate_legacy_ipa(
        evaluations_root=root,
        dataset_root=tmp_path,
        reference_manifest=source,
        apply=True,
        system=system,
    )

    target = run / "ipa" / system.system_id
    assert first["migrated"] == 1
    assert second["already_migrated"] == 1
    assert (target / "egra_eval_summary.txt").is_file()
    assert (run / "ipa" / "egra_eval_summary.txt").is_file()
    metadata = json.loads(
        (target / "evaluation_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["g2p_system"] == system.metadata()
    assert metadata["migration"]["legacy_preserved"] is True


def test_mismatch_is_reported_and_left_unranked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, run, _ = _legacy_fixture(tmp_path)
    system = _system()
    monkeypatch.setattr(
        migration,
        "verify_legacy_evaluation",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            migration.LegacyIPAMigrationError("aligned mismatch")
        ),
    )

    report = migration.migrate_legacy_ipa(
        evaluations_root=root,
        dataset_root=tmp_path,
        system=system,
    )

    assert report["rejected"] == 1
    assert "aligned mismatch" in report["runs"][0]["reason"]
    assert not (run / "ipa" / system.system_id).exists()
