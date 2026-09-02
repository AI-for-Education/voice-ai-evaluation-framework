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
    inference_setup_id: str,
    output_units: str,
) -> None:
    destination = output_root / "transcripts" / run_name
    destination.mkdir(parents=True)
    (destination / "run_metadata.json").write_text(
        json.dumps(
            {
                "metadata_schema_version": 2,
                "inference_setup_id": inference_setup_id,
                "run_id": run_name,
                "inference_profile": {
                    "profile_schema_version": 2,
                    "inference_setup_id": inference_setup_id,
                    "inference_library": "transformers",
                    "adapter": "ctc",
                    "artifact": "model",
                    "output_units": output_units,
                    "pipeline_contract": {
                        "schema_version": 2,
                        "audio_preparation": "shared_soundfile_librosa_16khz",
                        "model_artifact": "canonical_checkpoint",
                        "input_processing": "canonical_transformers_auto_processor",
                        "execution_stack": "shared_transformers_image",
                        "chunking": "none",
                        "evaluation": "direct_pred_text_by_output_units",
                        "observation_policy": "observed_effective_values_v2",
                    },
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
                "schema_version": 2,
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
    assert orthographic["inference_setup_id"].tolist() == [
        "orthographic-model"
    ]
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
    assert orthographic["execution_target"].tolist() == ["transformers"]
    assert orthographic["decoding"].tolist() == ["greedy"]
    assert orthographic["model_artifact"].tolist() == [
        "canonical model-owner artifact"
    ]
    assert orthographic["inference_setup"].tolist() == [
        "shared SoundFile/librosa 16 kHz audio; canonical Transformers frontend; no chunking; greedy decoding"
    ]
    assert orthographic["execution_stack"].tolist() == [
        "Transformers → PyTorch → shared ASR container"
    ]
    assert orthographic["context_evidence"].tolist() == [
        "inference-profile contract"
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
    assert ipa["inference_setup_id"].tolist() == [
        "orthographic-model",
        "phoneme-model",
    ]
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

    for index, inference_setup_id in enumerate(
        ("bookbot-orthographic-ctc", "bookbot-orthographic-ctc-5gram")
    ):
        run_name = f"retired_{index}"
        _write_run_metadata(
            output_root, run_name, inference_setup_id, "orthographic"
        )
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

    assert frames["orthographic"][["rank", "inference_setup_id"]].values.tolist() == [
        [1, "active-model"]
    ]
    assert frames["ipa"][["rank", "inference_setup_id"]].values.tolist() == [
        [1, "active-model"]
    ]
    assert skipped == {"orthographic": [], "ipa": []}


# Run selection and generated outputs


def test_latest_completed_run_per_inference_setup_is_selected(tmp_path: Path) -> None:
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
    assert latest["orthographic"]["run_id"].tolist() == ["model_new"]

    all_runs, _ = build_leaderboards(evaluations_root, latest_only=False)
    assert all_runs["orthographic"]["run_id"].tolist() == [
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
    assert paths["presentation"].name == "leaderboard_model_presentation.csv"
    assert paths["metadata"].name == "leaderboard_metadata.json"
    assert all(path.is_file() for path in paths.values())
    orthographic_csv = paths["orthographic"].read_text(encoding="utf-8")
    assert "passage_passage_corr" in orthographic_csv.splitlines()[0]
    assert "model_label" in orthographic_csv.splitlines()[0]
    assert "0.9123" in orthographic_csv
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 5
    assert metadata["presentation_naming"]["format"] == (
        "model_group · official_model_name [· variant] (decoder)"
    )
    assert metadata["presentation_naming"]["stable_identity_renamed"] is False
    assert metadata["presentation_naming"]["registered_inference_setups"] == 27
    # The table includes the uncatalogued evaluated fixture in addition to the
    # 27 active registry entries.
    assert metadata["presentation_naming"]["presentation_table"]["rows"] == 28
    presentation = paths["presentation"].read_text(encoding="utf-8")
    assert "inference_setup_id" in presentation.splitlines()[0]
    assert "execution_target" in presentation.splitlines()[0]
    assert "presentation_name" in presentation.splitlines()[0]
    assert "architecture_evidence_status" in presentation.splitlines()[0]
    assert "official_model_url" in presentation.splitlines()[0]
    assert metadata["comparison_factors"]["order"] == [
        "model_artifact",
        "inference_setup",
        "execution_stack",
    ]
    assert metadata["leaderboards"]["orthographic"]["ranking_metric"] == "wer"
    assert metadata["leaderboards"]["ipa"]["ranking_metric"] == "per"
    assert skipped == {"orthographic": [], "ipa": []}


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
            "execution_stack": "onnxruntime_shared_image",
        },
        "100": {
            "artifact": "published_deployment_artifact",
            "audio": "shared_soundfile_librosa_16khz",
            "frontend": "compatible_onnx_asr_nemo_frontend",
            "execution_stack": "onnxruntime_shared_image",
        },
        "110": {
            "artifact": "published_deployment_artifact",
            "audio": "android_pcm16_linear_16khz",
            "frontend": "deployment_parity_android_frontend",
            "execution_stack": "onnxruntime_shared_image",
        },
        "111": {
            "artifact": "published_deployment_artifact",
            "audio": "android_pcm16_linear_16khz",
            "frontend": "deployment_parity_android_frontend",
            "execution_stack": "android_pinned_execution_environment_proxy",
        },
    }
    for index, (variant, factors) in enumerate(variants.items()):
        run_name = f"controlled_{variant}"
        _write_run_metadata(
            output_root, run_name, f"controlled-{variant}", "orthographic"
        )
        metadata_path = output_root / "transcripts" / run_name / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["inference_profile"].update(
            {
                "inference_library": "onnxruntime",
                "adapter": "android_ctc" if variant in {"110", "111"} else "ctc",
                "artifact": "model.onnx",
            }
        )
        metadata["inference_adapter"] = {
            "quantization": "int8",
            "onnxruntime_version": "1.23.2" if variant != "111" else None,
            "onnxruntime_version_required": "1.22.0" if variant == "111" else None,
        }
        metadata["pipeline_provenance"] = {
            "recording": {"mode": "run_time"},
            "contract": {
                "references": {
                    "schema_version": 2,
                    "audio_preparation": factors["audio"],
                    "model_artifact": factors["artifact"],
                    "input_processing": factors["frontend"],
                    "execution_stack": factors["execution_stack"],
                    "chunking": "none",
                    "evaluation": "direct_pred_text_by_output_units",
                    "observation_policy": "observed_effective_values_v2",
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
    controlled = frame["orthographic"].set_index("inference_setup_id")

    assert skipped["orthographic"] == []
    assert controlled.loc["controlled-000", "model_artifact"] == (
        "project-exported ONNX artifact"
    )
    assert controlled.loc["controlled-100", "model_artifact"] == (
        "published Android deployment artifact"
    )
    assert controlled.loc["controlled-100", "inference_setup"] == (
        "shared SoundFile/librosa 16 kHz audio; compatible onnx-asr NeMo frontend; no chunking; greedy decoding"
    )
    assert controlled.loc["controlled-110", "inference_setup"] == (
        "Android PCM16/linear 16 kHz audio; Android deployment-parity frontend; no chunking; greedy decoding"
    )
    assert controlled.loc["controlled-100", "execution_stack"] == (
        "onnx-asr → ONNX Runtime → shared ASR container on PC "
        "(ONNX Runtime 1.23.2 observed)"
    )
    assert controlled.loc["controlled-100", "execution_target"] == "onnxruntime-desktop"
    assert controlled.loc["controlled-110", "execution_stack"] == (
        "Android-parity adapter → ONNX Runtime → shared ASR container on PC "
        "(ONNX Runtime 1.23.2 observed)"
    )
    assert controlled.loc["controlled-110", "context_evidence"] == (
        "pipeline contract (run time); inference engine observed"
    )
    assert controlled.loc["controlled-111", "execution_stack"] == (
        "Android-parity adapter → ONNX Runtime → dedicated Android-parity "
        "container on PC (ONNX Runtime 1.22.0 required)"
    )
    assert controlled.loc["controlled-111", "execution_target"] == (
        "onnxruntime-android-proxy"
    )


# Presentation registry


def test_presentation_registry_covers_profiles_and_groups_all_bookbot_models() -> None:
    repository = Path(__file__).resolve().parents[1]
    registry = load_model_presentation_registry()
    registered = set(registry["inference_setups"])
    profile_ids = {
        load_profile(path).inference_setup_id
        for path in (repository / "inference").glob("*/profiles/*.yaml")
    }

    assert registered == profile_ids
    bookbot_ids = {
        "bookbot-orthographic-ctc",
        "bookbot-orthographic-ctc-5gram",
        "bookbot-phoneme-ctc",
        "zipformer-streaming-robust-sw-v4",
        "zipformer-streaming-robust-sw-v4-modified-beam4",
        "zipformer-streaming-robust-sw-v4-onnx-int8",
        "zipformer-streaming-robust-sw-v4-onnx-int8-modified-beam4",
        "zipformer-streaming-robust-sw-v4-ort-int8",
        "zipformer-streaming-robust-sw-v4-ort-int8-modified-beam4",
        "zipformer-streaming-robust-sw-v4-torchscript",
        "zipformer-streaming-robust-sw-v4-torchscript-modified-beam4",
    }
    assert {
        registry["inference_setups"][model_id]["model_group"]
        for model_id in bookbot_ids
    } == {"Bookbot"}
    assert all(
        registry["inference_setups"][model_id]["architecture"]["evidence_status"]
        in {"artifact_verified", "owner_documented"}
        for model_id in registered
    )
    assert all(
        registry["inference_setups"][model_id]["official_model_url"].startswith(
            "https://"
        )
        for model_id in registered
    )


def test_presentation_registry_requires_a_naming_format(tmp_path: Path) -> None:
    registry_path = tmp_path / "model_presentation.json"
    registry_path.write_text(
        json.dumps(
            {"schema_version": 2, "naming_format": "", "inference_setups": {}}
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="has no naming format"):
        load_model_presentation_registry(registry_path)


def test_presentation_registry_rejects_removed_schema_version(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "model_presentation.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "naming_format": "model_group · model_name",
                "inference_setups": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema_version must be 2"):
        load_model_presentation_registry(registry_path)
