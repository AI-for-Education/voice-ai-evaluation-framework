from __future__ import annotations

import json
from pathlib import Path

import pytest

from egra_eval2.model_presentation import load_model_presentation_registry
from egra_eval2.leaderboard import build_leaderboards, write_leaderboards
from inference.profile import load_profile


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
                    "framework": "transformers",
                    "adapter": "ctc",
                    "output_units": output_units,
                    "decoding": {
                        "strategy": "greedy",
                        "generation_kwargs": {},
                    },
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


# Representation separation and result eligibility


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
    assert orthographic["model_group"].tolist() == ["Uncatalogued"]
    assert orthographic["model_name"].tolist() == ["orthographic-model"]
    assert orthographic["model_variant"].tolist() == ["Unspecified"]
    assert orthographic["official_model_url"].tolist() == [""]
    assert orthographic["official_model_url_note"].tolist() == [
        "No official model page is registered"
    ]
    assert orthographic["architecture"].tolist() == ["not available"]
    assert orthographic["architecture_evidence_status"].tolist() == [
        "not_available"
    ]
    assert orthographic["platform"].tolist() == ["transformers"]
    assert orthographic["decoding"].tolist() == ["greedy"]
    assert orthographic["artifact_context"].tolist() == [
        "canonical model-owner artifact"
    ]
    assert orthographic["preprocessing_context"].tolist() == [
        "shared SoundFile/librosa 16 kHz audio; canonical Transformers frontend; no chunking"
    ]
    assert orthographic["runtime_context"].tolist() == [
        "shared ASR image / Transformers runtime"
    ]
    assert orthographic["context_evidence"].tolist() == [
        "profile-derived contract fallback"
    ]
    assert orthographic["evaluation_status"].tolist() == ["scored"]
    assert orthographic["model_label"].tolist() == [
        "Uncatalogued · orthographic-model · Unspecified (greedy)"
    ]
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


def test_excludes_retired_bookbot_orthographic_models(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    evaluations_root = output_root / "evaluations"

    for index, model_id in enumerate(
        ("bookbot-orthographic-ctc", "bookbot-orthographic-ctc-5gram")
    ):
        run_name = f"retired_{index}"
        _write_run_metadata(output_root, run_name, model_id, "orthographic")
        _write_evaluation(
            output_root,
            run_name,
            "orthographic",
            "wer",
            1.0 + index,
            f"2026-01-02T00:0{index}:00+00:00",
        )
        _write_evaluation(
            output_root,
            run_name,
            "ipa",
            "per",
            1.0 + index,
            f"2026-01-02T00:0{index}:30+00:00",
        )

    _write_run_metadata(output_root, "active", "active-model", "orthographic")
    _write_evaluation(
        output_root,
        "active",
        "orthographic",
        "wer",
        50.0,
        "2026-01-02T00:03:00+00:00",
    )
    _write_evaluation(
        output_root,
        "active",
        "ipa",
        "per",
        40.0,
        "2026-01-02T00:03:30+00:00",
    )

    frames, skipped = build_leaderboards(evaluations_root)

    assert frames["orthographic"][["rank", "model_id"]].values.tolist() == [
        [1, "active-model"]
    ]
    assert frames["ipa"][["rank", "model_id"]].values.tolist() == [
        [1, "active-model"]
    ]
    assert skipped == {"orthographic": [], "ipa": []}


# Run selection and generated outputs


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
    assert paths["name_mapping"].name == "leaderboard_model_name_mapping.csv"
    assert paths["metadata"].name == "leaderboard_metadata.json"
    assert all(path.is_file() for path in paths.values())
    orthographic_csv = paths["orthographic"].read_text(encoding="utf-8")
    assert "passage_passage_corr" in orthographic_csv.splitlines()[0]
    assert "model_label" in orthographic_csv.splitlines()[0]
    assert "0.9123" in orthographic_csv
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 4
    assert metadata["presentation_naming"]["format"] == (
        "model_group · official_model_name [· variant] (decoder)"
    )
    assert metadata["presentation_naming"]["stable_identity_renamed"] is False
    assert metadata["presentation_naming"]["old_to_new_mapping"]["rows"] == 22
    assert metadata["presentation_naming"]["registered_models"] == 21
    mapping = paths["name_mapping"].read_text(encoding="utf-8")
    assert "previous_model_id" in mapping.splitlines()[0]
    assert "new_presentation_name" in mapping.splitlines()[0]
    assert "architecture_evidence_status" in mapping.splitlines()[0]
    assert "official_model_url" in mapping.splitlines()[0]
    assert metadata["comparison_factors"]["order"] == [
        "artifact_context",
        "preprocessing_context",
        "runtime_context",
    ]
    assert metadata["leaderboards"]["orthographic"]["ranking_metric"] == "wer"
    assert metadata["leaderboards"]["ipa"]["ranking_metric"] == "per"
    assert skipped == {"orthographic": [], "ipa": []}


# Audit and controlled-comparison columns


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


def test_controlled_onnx_comparison_keeps_three_factors_separate(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    evaluations_root = output_root / "evaluations"
    variants = {
        "000": {
            "artifact": "project_export",
            "audio": "shared_soundfile_librosa_16khz",
            "frontend": "compatible_onnx_asr_nemo_frontend",
            "runtime": "onnxruntime_shared_image",
        },
        "100": {
            "artifact": "published_deployment_artifact",
            "audio": "shared_soundfile_librosa_16khz",
            "frontend": "compatible_onnx_asr_nemo_frontend",
            "runtime": "onnxruntime_shared_image",
        },
        "110": {
            "artifact": "published_deployment_artifact",
            "audio": "android_pcm16_linear_16khz",
            "frontend": "deployment_parity_android_frontend",
            "runtime": "onnxruntime_shared_image",
        },
        "111": {
            "artifact": "published_deployment_artifact",
            "audio": "android_pcm16_linear_16khz",
            "frontend": "deployment_parity_android_frontend",
            "runtime": "android_pinned_runtime_proxy",
        },
    }
    for index, (variant, factors) in enumerate(variants.items()):
        run_name = f"controlled_{variant}"
        _write_run_metadata(
            output_root, run_name, f"controlled-{variant}", "orthographic"
        )
        metadata_path = output_root / "transcripts" / run_name / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["profile"].update(
            {
                "framework": "onnxruntime",
                "adapter": "android_ctc" if variant in {"110", "111"} else "ctc",
                "artifact": "model.onnx",
            }
        )
        metadata["backend"] = {
            "quantization": "int8",
            "onnxruntime_version": "1.23.2" if variant != "111" else None,
            "onnxruntime_version_required": "1.22.0" if variant == "111" else None,
        }
        metadata["pipeline_provenance"] = {
            "recording": {"mode": "retrospective_recovery"},
            "contract": {
                "references": {
                    "schema_version": 1,
                    "audio_preparation": factors["audio"],
                    "inference": {
                        "artifact": factors["artifact"],
                        "frontend": factors["frontend"],
                        "runtime": factors["runtime"],
                        "chunking": "none",
                    },
                    "evaluation": "direct_pred_text_by_output_units",
                    "runtime_resolution": "observed_effective_values_v1",
                }
            },
        }
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        _write_evaluation(
            output_root,
            run_name,
            "orthographic",
            "wer",
            20.0 + index,
            f"2026-01-0{index + 1}T00:00:00+00:00",
        )

    frame, skipped = build_leaderboards(evaluations_root, latest_only=False)
    controlled = frame["orthographic"].set_index("model_id")

    assert skipped["orthographic"] == []
    assert controlled.loc["controlled-000", "artifact_context"] == (
        "project-exported ONNX artifact"
    )
    assert controlled.loc["controlled-100", "artifact_context"] == (
        "published Android deployment artifact"
    )
    assert controlled.loc["controlled-100", "preprocessing_context"] == (
        "shared SoundFile/librosa 16 kHz audio; compatible onnx-asr NeMo frontend; no chunking"
    )
    assert controlled.loc["controlled-110", "preprocessing_context"] == (
        "Android PCM16/linear 16 kHz audio; Android deployment-parity frontend; no chunking"
    )
    assert controlled.loc["controlled-100", "runtime_context"] == (
        "shared ASR image / ONNX Runtime (ONNX Runtime 1.23.2 observed)"
    )
    assert controlled.loc["controlled-100", "platform"] == "onnxruntime-desktop"
    assert controlled.loc["controlled-110", "runtime_context"] == (
        "shared ASR image / ONNX Runtime (ONNX Runtime 1.23.2 observed)"
    )
    assert controlled.loc["controlled-110", "context_evidence"] == (
        "pipeline contract (retrospective recovery); runtime package/backend observed"
    )
    assert controlled.loc["controlled-111", "runtime_context"] == (
        "dedicated Android-parity runtime (PC proxy) (ONNX Runtime 1.22.0 required)"
    )
    assert controlled.loc["controlled-111", "platform"] == (
        "onnxruntime-android-proxy"
    )


# Presentation registry


def test_presentation_registry_covers_profiles_and_groups_all_bookbot_models() -> None:
    repository = Path(__file__).resolve().parents[1]
    registry = load_model_presentation_registry()
    registered = set(registry["models"])
    profile_ids = {
        load_profile(path).id
        for path in (repository / "inference").glob("*/profiles/*.yaml")
    }

    assert registered == profile_ids
    bookbot_ids = {
        "bookbot-orthographic-ctc",
        "bookbot-orthographic-ctc-5gram",
        "bookbot-phoneme-ctc",
        "zipformer-streaming-robust-sw-v4",
        "zipformer-streaming-robust-sw-v4-modified-beam4",
    }
    assert {
        registry["models"][model_id]["model_group"] for model_id in bookbot_ids
    } == {"Bookbot"}
    assert all(
        registry["models"][model_id]["architecture"]["evidence_status"]
        in {"artifact_verified", "owner_documented"}
        for model_id in registered
    )
    assert all(
        registry["models"][model_id]["official_model_url"].startswith("https://")
        for model_id in registered
    )


def test_presentation_registry_requires_a_naming_format(tmp_path: Path) -> None:
    registry_path = tmp_path / "model_presentation.json"
    registry_path.write_text(
        json.dumps({"schema_version": 1, "naming_format": "", "models": {}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="has no naming format"):
        load_model_presentation_registry(registry_path)
