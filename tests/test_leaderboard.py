from __future__ import annotations

import json
from pathlib import Path

import pytest

from egra_eval2.model_presentation import load_model_presentation_registry
from egra_eval2.leaderboard import (
    _g2p_display_name,
    build_leaderboards,
    write_leaderboards,
)
from egra_eval2.leaderboard_context import (
    execution_stack_context,
    execution_target_label,
)
from egra_eval2.reference.g2p import G2PSystem, make_test_system
from inference.profile import load_profile


def _system(tool_id: str = "africa_g2p") -> G2PSystem:
    return make_test_system(
        tool_id=tool_id,
        display_name="babygruut" if tool_id == "babygruut" else "Africa G2P",
        language="sw" if tool_id == "babygruut" else "swh",
        inventory=(
            "babygruut_sw_ipa_v1"
            if tool_id == "babygruut"
            else "africa_g2p_swh_ipa_v1"
        ),
        version_value="leaderboard-test",
        phonemize=lambda values: [str(value) for value in values],
    )


def _write_run_metadata(
    output_root: Path,
    run_name: str,
    inference_setup_id: str,
    output_units: str,
) -> None:
    destination = output_root / "transcripts" / run_name
    destination.mkdir(parents=True)
    profile = {
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
    if output_units == "phoneme":
        profile.update(
            {
                "output_notation": "ipa",
                "output_inventory": "africa_g2p_swh_ipa_v1",
            }
        )
    (destination / "run_metadata.json").write_text(
        json.dumps(
            {
                "metadata_schema_version": 2,
                "inference_setup_id": inference_setup_id,
                "run_id": run_name,
                "inference_profile": profile,
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
    g2p_system: G2PSystem | None = None,
) -> None:
    selected_system = g2p_system or _system()
    destination = output_root / "evaluations" / run_name / namespace
    if namespace == "ipa":
        destination /= selected_system.system_id
    destination.mkdir(parents=True)
    scoring_units = "phoneme" if namespace == "ipa" else "orthographic"
    metadata = {
        "schema_version": 3,
        "status": "complete",
        "completed_at": completed_at,
        "effective_scoring_units": scoring_units,
        "output_namespace": namespace,
        "representation_compatible": compatible,
    }
    if namespace == "ipa":
        metadata.update(
            {
                "g2p_system": selected_system.metadata(),
                "hypothesis_route": (
                    f"orthographic -> {selected_system.display_name} "
                    f"({selected_system.system_id})"
                ),
            }
        )
    (destination / "evaluation_metadata.json").write_text(
        json.dumps(metadata),
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


def test_babygruut_uses_canonical_public_name_for_historical_metadata() -> None:
    assert _g2p_display_name("babygruut", "BabyGroot") == "babygruut"
    assert _g2p_display_name("africa_g2p", "Africa G2P") == "Africa G2P"


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

    system_id = _system().system_id
    ipa = frames["ipa_by_system"][system_id]
    assert ipa["inference_setup_id"].tolist() == [
        "orthographic-model",
        "phoneme-model",
    ]
    assert ipa["global_per"].tolist() == [10.0, 15.0]
    assert ipa["hypothesis_route"].tolist() == [
        f"orthographic -> Africa G2P ({system_id})",
        "native IPA africa_g2p_swh_ipa_v1 — no phoneme conversion",
    ]
    assert skipped == {
        "orthographic": [],
        "ipa_by_system": {system_id: []},
    }


def test_same_run_is_ranked_only_inside_each_exact_g2p_system(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    evaluations_root = output_root / "evaluations"
    africa = _system("africa_g2p")
    baby = _system("babygruut")
    _write_run_metadata(output_root, "same_run", "same-model", "orthographic")
    _write_evaluation(
        output_root,
        "same_run",
        "ipa",
        "per",
        12.0,
        "2026-01-02T00:00:00+00:00",
        g2p_system=africa,
    )
    _write_evaluation(
        output_root,
        "same_run",
        "ipa",
        "per",
        8.0,
        "2026-01-02T00:01:00+00:00",
        g2p_system=baby,
    )

    frames, skipped = build_leaderboards(evaluations_root)

    assert set(frames["ipa_by_system"]) == {africa.system_id, baby.system_id}
    assert frames["ipa_by_system"][africa.system_id]["global_per"].tolist() == [
        12.0
    ]
    assert frames["ipa_by_system"][baby.system_id]["global_per"].tolist() == [
        8.0
    ]
    assert all(
        set(frame["g2p_system_id"]) == {system_id}
        for system_id, frame in frames["ipa_by_system"].items()
    )
    assert set(skipped["ipa_by_system"]) == {africa.system_id, baby.system_id}


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
    diagnostic = evaluations_root / run_name / "ipa" / "can_hyp"
    diagnostic.mkdir(parents=True)
    (diagnostic / "egra_eval_summary.txt").write_text(
        "GLOBAL\n  per: 0.01%\n", encoding="utf-8"
    )

    frames, skipped = build_leaderboards(evaluations_root)

    assert frames["orthographic"].empty
    assert frames["ipa_by_system"] == {}
    assert len(skipped["orthographic"]) == 1
    assert "representation_compatible=true" in skipped["orthographic"][0]
    assert skipped["ipa_by_system"] == {}


def test_excludes_smoke_test_runs(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    evaluations_root = output_root / "evaluations"
    run_name = "smoke_run"
    _write_run_metadata(output_root, run_name, "smoke-model", "orthographic")
    metadata_path = output_root / "transcripts" / run_name / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["output"] = {"smoke_test": True, "results": 1}
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    _write_evaluation(
        output_root,
        run_name,
        "orthographic",
        "wer",
        1.0,
        "2026-01-02T00:00:00+00:00",
    )
    system = _system()
    _write_evaluation(
        output_root,
        run_name,
        "ipa",
        "per",
        2.0,
        "2026-01-02T00:00:00+00:00",
        g2p_system=system,
    )

    frames, skipped = build_leaderboards(evaluations_root)

    assert frames["orthographic"].empty
    assert len(skipped["orthographic"]) == 1
    assert "Smoke-test run cannot enter" in skipped["orthographic"][0]
    assert frames["ipa_by_system"][system.system_id].empty
    assert len(skipped["ipa_by_system"][system.system_id]) == 1
    assert (
        "Smoke-test run cannot enter"
        in skipped["ipa_by_system"][system.system_id][0]
    )


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
    system_id = _system().system_id
    assert frames["ipa_by_system"][system_id][
        ["rank", "inference_setup_id"]
    ].values.tolist() == [
        [1, "active-model"]
    ]
    assert skipped == {
        "orthographic": [],
        "ipa_by_system": {system_id: []},
    }


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


def test_writes_isolated_csvs_and_generation_metadata(tmp_path: Path) -> None:
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
    system_id = _system().system_id
    assert paths["ipa_by_system"][system_id].name == (
        f"leaderboard_{system_id}.csv"
    )
    assert paths["ipa_by_system"][system_id].parent.name == "ipa"
    assert paths["presentation"].name == "leaderboard_model_presentation.csv"
    assert paths["metadata"].name == "leaderboard_metadata.json"
    assert paths["orthographic"].is_file()
    assert paths["ipa_by_system"][system_id].is_file()
    assert paths["presentation"].is_file()
    assert paths["metadata"].is_file()
    orthographic_csv = paths["orthographic"].read_text(encoding="utf-8")
    assert "passage_passage_corr" in orthographic_csv.splitlines()[0]
    assert "model_label" in orthographic_csv.splitlines()[0]
    assert "evaluation_status" not in orthographic_csv.splitlines()[0]
    assert "0.9123" in orthographic_csv
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 6
    assert metadata["eligibility_policy"] == {
        "completed_evaluations_only": True,
        "representation_compatible_only": True,
        "smoke_tests_excluded": True,
    }
    assert metadata["presentation_naming"]["format"] == (
        "model_group · official_model_name [· variant] (decoder)"
    )
    assert metadata["presentation_naming"]["stable_identity_renamed"] is False
    assert metadata["presentation_naming"]["registered_inference_setups"] == 41
    # The table also includes profile-only and uncatalogued evaluated entries.
    assert metadata["presentation_naming"]["presentation_table"]["rows"] == 42
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
    assert metadata["leaderboards"]["ipa_by_system"][system_id][
        "ranking_metric"
    ] == "per"
    assert skipped == {
        "orthographic": [],
        "ipa_by_system": {system_id: []},
    }


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


def test_structured_unavailable_compose_service_uses_contract_fallback() -> None:
    profile = {
        "inference_library": "openrouter",
        "adapter": "openrouter_audio",
    }
    run_metadata = {
        "execution_stack": {
            "launch_context": {
                "status": "observed",
                "compose_service": {
                    "status": "not_available",
                    "reason": "The Compose service was not recorded",
                },
            }
        }
    }
    references = {"execution_stack": "openrouter_eu_api"}

    stack, observed = execution_stack_context(profile, run_metadata, references)

    assert stack == (
        "OpenRouter API → eligible EU/ZDR provider → observed container launch"
    )
    assert observed is True
    assert (
        execution_target_label(profile, run_metadata, references)
        == "openrouter-eu-zdr"
    )


@pytest.mark.parametrize("region", ["eu", "global", None])
@pytest.mark.parametrize("observed", [True, False])
def test_openrouter_labels_use_recorded_region(region, observed) -> None:
    profile = {"inference_library": "openrouter", "api": {"routing_region": region}}
    run = {
        "execution_stack": {
            "launch_context": {"status": "observed", "compose_service": "openrouter-asr"}
        }
    } if observed else {}
    references = {
        "execution_stack": {
            "eu": "openrouter_eu_api", "global": "openrouter_global_zdr_api"
        }.get(region)
    }
    stack, _ = execution_stack_context(profile, run, references)
    target = execution_target_label(profile, run, references)
    if region == "eu":
        assert "EU/ZDR" in stack
        assert target == "openrouter-eu-zdr"
    else:
        assert "EU" not in stack
        assert "eu" not in target
        if region == "global":
            assert "global" in stack
            assert target == "openrouter-global-zdr"


@pytest.mark.parametrize("observed_launch", [True, False])
def test_openrouter_observed_gateway_takes_precedence_over_profile(observed_launch) -> None:
    profile = {"inference_library": "openrouter", "api": {"routing_region": "eu"}}
    run = {
        "inference_adapter": {"region_enforcement": {"gateway": "global"}},
        "execution_stack": {
            "launch_context": {"status": "observed", "compose_service": "openrouter-asr"}
        },
    }
    if not observed_launch:
        run.pop("execution_stack")
    references = {"execution_stack": "openrouter_eu_api"}
    stack, _ = execution_stack_context(profile, run, references)
    assert "EU" not in stack
    assert execution_target_label(profile, run, references) == "openrouter-global-zdr"


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
