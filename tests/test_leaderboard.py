from __future__ import annotations

import json
from pathlib import Path

from egra_eval2.leaderboard import build_leaderboards, write_leaderboards


def _write_run_metadata(
    output_root: Path,
    run_name: str,
    model_id: str,
    output_units: str,
) -> None:
    destination = output_root / "transcripts" / run_name
    destination.mkdir(parents=True)
    (destination / "run_metadata.json").write_text(
        json.dumps(
            {
                "profile": {
                    "id": model_id,
                    "output_units": output_units,
                }
            }
        ),
        encoding="utf-8",
    )


def _write_evaluation(
    output_root: Path,
    run_name: str,
    namespace: str,
    metric: str,
    value: float,
    completed_at: str,
    *,
    compatible: bool = True,
) -> None:
    destination = output_root / "evaluations" / run_name / namespace
    destination.mkdir(parents=True)
    scoring_units = "phoneme" if namespace == "ipa" else "orthographic"
    (destination / "evaluation_metadata.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "completed_at": completed_at,
                "effective_scoring_units": scoring_units,
                "output_namespace": namespace,
                "representation_compatible": compatible,
            }
        ),
        encoding="utf-8",
    )
    (destination / "egra_eval_summary.txt").write_text(
        "\n".join(
            [
                "GLOBAL",
                f"  {metric}: {value:.2f}%",
                "  mer: 12.34%",
                "",
                "PASSAGE_PASSAGE",
                f"  {metric}: {value + 0.5:.2f}%",
                "  mer: 10.00%",
                "  corr: 0.9123",
                "",
                "LETTERS_ISOLATED",
                f"  {metric}: {value + 1:.2f}%",
                "  mer: 8.00%",
                "  accuracy: 88.00%",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_builds_separate_wer_and_per_leaderboards(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    evaluations_root = output_root / "evaluations"

    _write_run_metadata(output_root, "orthographic_run", "orthographic-model", "orthographic")
    _write_evaluation(
        output_root,
        "orthographic_run",
        "orthographic",
        "wer",
        20.0,
        "2026-01-02T00:00:00+00:00",
    )
    _write_evaluation(
        output_root,
        "orthographic_run",
        "ipa",
        "per",
        10.0,
        "2026-01-02T00:01:00+00:00",
    )

    _write_run_metadata(output_root, "phoneme_run", "phoneme-model", "phoneme")
    _write_evaluation(
        output_root,
        "phoneme_run",
        "ipa",
        "per",
        15.0,
        "2026-01-02T00:02:00+00:00",
    )

    frames, skipped = build_leaderboards(evaluations_root)

    orthographic = frames["orthographic"]
    assert orthographic["model_id"].tolist() == ["orthographic-model"]
    assert orthographic["global_wer"].tolist() == [20.0]
    assert orthographic["global_mer"].tolist() == [12.34]
    assert orthographic["passage_passage_wer"].tolist() == [20.5]
    assert orthographic["passage_passage_mer"].tolist() == [10.0]
    assert orthographic["passage_passage_corr"].tolist() == [0.9123]
    assert orthographic["letters_isolated_wer"].tolist() == [21.0]
    assert orthographic["letters_isolated_mer"].tolist() == [8.0]
    assert orthographic["letters_isolated_accuracy"].tolist() == [88.0]
    assert "global_per" not in orthographic.columns

    ipa = frames["ipa"]
    assert ipa["model_id"].tolist() == ["orthographic-model", "phoneme-model"]
    assert ipa["global_per"].tolist() == [10.0, 15.0]
    assert ipa["hypothesis_route"].tolist() == [
        "orthographic -> IPA (Africa G2P)",
        "native IPA -> canonical IPA",
    ]
    assert skipped == {"orthographic": [], "ipa": []}


def test_excludes_incompatible_and_archived_results(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    evaluations_root = output_root / "evaluations"
    run_name = "model_run"
    _write_run_metadata(output_root, run_name, "model", "orthographic")
    _write_evaluation(
        output_root,
        run_name,
        "orthographic",
        "wer",
        30.0,
        "2026-01-02T00:00:00+00:00",
        compatible=False,
    )

    archive = evaluations_root / run_name / "pre_reference_fix_20260813" / "ipa"
    archive.mkdir(parents=True)
    (archive / "egra_eval_summary.txt").write_text(
        "GLOBAL\n  per: 0.01%\n", encoding="utf-8"
    )

    frames, skipped = build_leaderboards(evaluations_root)

    assert frames["orthographic"].empty
    assert frames["ipa"].empty
    assert len(skipped["orthographic"]) == 1
    assert "representation_compatible=true" in skipped["orthographic"][0]
    assert skipped["ipa"] == []


def test_latest_completed_run_per_model_is_selected(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    evaluations_root = output_root / "evaluations"
    for run_name, value, completed_at in (
        ("model_old", 10.0, "2026-01-01T00:00:00+00:00"),
        ("model_new", 25.0, "2026-01-02T00:00:00+00:00"),
    ):
        _write_run_metadata(output_root, run_name, "same-model", "orthographic")
        _write_evaluation(
            output_root,
            run_name,
            "orthographic",
            "wer",
            value,
            completed_at,
        )

    latest, _ = build_leaderboards(evaluations_root)
    assert latest["orthographic"]["run_name"].tolist() == ["model_new"]

    all_runs, _ = build_leaderboards(evaluations_root, latest_only=False)
    assert all_runs["orthographic"]["run_name"].tolist() == [
        "model_old",
        "model_new",
    ]


def test_writes_two_csvs_and_generation_metadata(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    _write_run_metadata(output_root, "model_run", "model", "orthographic")
    _write_evaluation(
        output_root,
        "model_run",
        "orthographic",
        "wer",
        20.0,
        "2026-01-02T00:00:00+00:00",
    )
    _write_evaluation(
        output_root,
        "model_run",
        "ipa",
        "per",
        15.0,
        "2026-01-02T00:01:00+00:00",
    )

    paths, skipped = write_leaderboards(
        output_root / "evaluations", tmp_path / "leaderboards"
    )

    assert paths["orthographic"].name == "leaderboard_orthographic.csv"
    assert paths["ipa"].name == "leaderboard_ipa.csv"
    assert paths["metadata"].name == "leaderboard_metadata.json"
    assert all(path.is_file() for path in paths.values())
    orthographic_csv = paths["orthographic"].read_text(encoding="utf-8")
    assert "passage_passage_corr" in orthographic_csv.splitlines()[0]
    assert "0.9123" in orthographic_csv
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["leaderboards"]["orthographic"]["ranking_metric"] == "wer"
    assert metadata["leaderboards"]["ipa"]["ranking_metric"] == "per"
    assert skipped == {"orthographic": [], "ipa": []}


def test_leaderboard_shows_new_and_legacy_postprocessing_audits(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    evaluations_root = output_root / "evaluations"

    _write_run_metadata(output_root, "new_run", "new-model", "orthographic")
    new_metadata_path = output_root / "transcripts" / "new_run" / "run_metadata.json"
    new_metadata = json.loads(new_metadata_path.read_text(encoding="utf-8"))
    new_metadata["output"] = {"results": 10}
    new_metadata["postprocessing"] = {
        "stage": "post_decode_pre_evaluation",
        "method": "hallucination_guard",
        "scored_text_field": "pred_text",
        "raw_text_field": "raw_pred_text",
        "adjusted_results": 3,
        "total_words_removed": 7,
    }
    new_metadata_path.write_text(json.dumps(new_metadata), encoding="utf-8")
    _write_evaluation(
        output_root,
        "new_run",
        "orthographic",
        "wer",
        20.0,
        "2026-01-02T00:00:00+00:00",
    )

    _write_run_metadata(output_root, "legacy_run", "legacy-model", "orthographic")
    legacy_dir = output_root / "transcripts" / "legacy_run"
    legacy_metadata_path = legacy_dir / "run_metadata.json"
    legacy_metadata = json.loads(legacy_metadata_path.read_text(encoding="utf-8"))
    legacy_metadata["backend"] = {"hallucination_guard": {"max_words_per_second": 8.0}}
    legacy_metadata["output"] = {"results": 2}
    legacy_metadata_path.write_text(json.dumps(legacy_metadata), encoding="utf-8")
    (legacy_dir / "transcriptions.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"pred_text": "one", "raw_pred_text": "one two three"}),
                json.dumps({"pred_text": "four"}),
            ]
        ),
        encoding="utf-8",
    )
    _write_evaluation(
        output_root,
        "legacy_run",
        "orthographic",
        "wer",
        25.0,
        "2026-01-02T00:00:00+00:00",
    )

    frames, _ = build_leaderboards(evaluations_root, latest_only=False)
    frame = frames["orthographic"].set_index("run_name")

    assert frame.loc["new_run", "scored_hypothesis"] == "pred_text (post-processed)"
    assert frame.loc["new_run", "postprocessing_method"] == "hallucination_guard"
    assert frame.loc["new_run", "postprocessed_rows"] == 3
    assert frame.loc["new_run", "postprocessed_rows_pct"] == 30.0
    assert frame.loc["new_run", "postprocessing_words_removed"] == 7
    assert frame.loc["new_run", "postprocessing_audit_source"] == "run_metadata"

    assert frame.loc["legacy_run", "postprocessing_method"] == "hallucination_guard"
    assert frame.loc["legacy_run", "postprocessed_rows"] == 1
    assert frame.loc["legacy_run", "postprocessed_rows_pct"] == 50.0
    assert frame.loc["legacy_run", "postprocessing_words_removed"] == 2
    assert frame.loc["legacy_run", "postprocessing_audit_source"] == "raw_pred_text fallback"
