from __future__ import annotations

import json
import sys
import types
import wave
from pathlib import Path

import pytest

from inference.pipeline_provenance import (
    build_pipeline_provenance,
    execution_environment_from_environment,
    infer_contract_references,
    resolve_contract_references,
    summarize_wav_headers,
)
from inference.profile import load_profile
from tools.migrations.backfill_pipeline_provenance import _parse_args, backfill


def test_all_tracked_profiles_resolve_their_declared_contract() -> None:
    repository = Path(__file__).resolve().parents[1]
    paths = sorted((repository / "inference").glob("*/profiles/*.yaml"))
    assert len(paths) == 29
    for path in paths:
        profile = load_profile(path)
        payload = profile.to_dict()
        references = infer_contract_references(payload)
        assert references == payload["pipeline_contract"]
        resolved = resolve_contract_references(references)
        assert resolved["audio_preparation"]["authority"]
        assert resolved["input_processing"]["classification"] in {
            "canonical_model_frontend",
            "compatible_third_party_frontend",
            "deployment_parity_frontend",
        }
        assert resolved["evaluation"]["hypothesis_field"] == "pred_text"
        assert resolved["observation_policy"]["unavailable_value_policy"] == {
            "status": "not_available",
            "require_reason": True,
            "guessing_forbidden": True,
        }


def test_wav_header_summary_observes_format_without_decoding(tmp_path: Path) -> None:
    wav_path = tmp_path / "sample.wav"
    with wave.open(str(wav_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(b"\x00\x00" * 8)

    summary = summarize_wav_headers([str(wav_path)])

    assert summary["status"] == "observed"
    assert summary["sample_rate_hz_counts"] == {"8000": 1}
    assert summary["channel_counts"] == {"1": 1}
    assert summary["sample_width_byte_counts"] == {"2": 1}


def test_backfill_default_repository_root_is_the_project_root() -> None:
    args = _parse_args([])

    assert Path(args.repository_root) == Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "metadata",
    [
        {
            "inference_profile": {},
            "inference_adapter": {},
            "execution_stack": {},
            "model_artifact": {"identity": {}},
        },
        {
            "profile": {},
            "backend": {},
            "runtime": {},
            "model_identity": {},
        },
    ],
    ids=("current-metadata", "historical-metadata"),
)
def test_backfill_dry_run_accepts_current_and_historical_metadata(
    tmp_path: Path,
    metadata: dict[str, object],
) -> None:
    run_dir = tmp_path / "transcripts" / "sample_run"
    run_dir.mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    (run_dir / "transcriptions.jsonl").write_text("{}\n", encoding="utf-8")

    summary = backfill(
        output_root=tmp_path,
        repository_root=Path(__file__).resolve().parents[1],
        apply=False,
        refresh=False,
    )

    assert summary["run_metadata"] == {
        "discovered": 1,
        "updated": 1,
        "skipped": 0,
        "errors": 0,
    }
    assert summary["error_details"] == []


def test_execution_environment_records_container_identity_without_guessing() -> None:
    context = execution_environment_from_environment(
        {
            "PIPELINE_LAUNCH_ORCHESTRATOR": "docker_compose",
            "PIPELINE_LAUNCHER": "run_onnxruntime_inference.sh",
            "PIPELINE_COMPOSE_SERVICE": "onnxruntime-asr",
            "PIPELINE_IMAGE_REFERENCE": "voice-ai-evaluation-framework-asr:latest",
            "PIPELINE_IMAGE_ID": "sha256:abc123",
            "PIPELINE_IMAGE_REPO_DIGESTS_JSON": '["example/asr@sha256:def456"]',
        }
    )

    assert context == {
        "status": "observed",
        "orchestrator": "docker_compose",
        "launcher": "run_onnxruntime_inference.sh",
        "compose_service": "onnxruntime-asr",
        "image": {
            "reference": "voice-ai-evaluation-framework-asr:latest",
            "id": "sha256:abc123",
            "repo_digests": ["example/asr@sha256:def456"],
        },
    }
    unavailable = execution_environment_from_environment({})
    assert unavailable["status"] == "not_available"


def test_all_profile_launchers_use_the_central_execution_environment_helper() -> None:
    repository = Path(__file__).resolve().parents[1]
    expected = {
        "run_nemo_inference.sh": ("nemo-asr", "voice-ai-evaluation-framework-asr:latest"),
        "run_transformers_inference.sh": (
            "transformers-asr",
            "voice-ai-evaluation-framework-asr:latest",
        ),
        "run_sherpa_onnx_inference.sh": (
            "sherpa-onnx-asr",
            "voice-ai-evaluation-framework-asr:latest",
        ),
        "run_torch_inference.sh": (
            "torch-asr",
            "voice-ai-evaluation-framework-asr:latest",
        ),
        "run_onnxruntime_inference.sh": (
            "onnxruntime-asr",
            "voice-ai-evaluation-framework-asr:latest",
        ),
        "run_onnxruntime_android_inference.sh": (
            "onnxruntime-android-asr",
            "voice-ai-evaluation-framework-onnxruntime-android:latest",
        ),
        "run_multimodal_inference.sh": (
            "multimodal-asr",
            "voice-ai-evaluation-framework-asr:latest",
        ),
        "run_phi4_multimodal_inference.sh": (
            "phi4-multimodal-asr",
            "voice-ai-evaluation-framework-phi4:latest",
        ),
        "run_qwen_omni_inference.sh": (
            "qwen-omni-asr",
            "voice-ai-evaluation-framework-qwen-omni:latest",
        ),
    }

    for launcher, (service, image) in expected.items():
        source = (repository / launcher).read_text(encoding="utf-8")
        assert (
            'source "$SCRIPT_DIR/inference/execution_environment_identity.sh"'
            in source
        )
        assert "pipeline_execution_environment_docker_args" in source
        assert f'"{service}"' in source
        assert f'"{image}"' in source
        assert '"${PIPELINE_EXECUTION_ENVIRONMENT_DOCKER_ARGS[@]}"' in source

def test_missing_current_contract_is_explicitly_not_available(tmp_path: Path) -> None:
    transcript = tmp_path / "transcriptions.jsonl"
    transcript.write_text("", encoding="utf-8")

    pipeline = build_pipeline_provenance(
        profile={
            "profile_schema_version": 2,
            "inference_setup_id": "unknown",
            "inference_library": "transformers",
        },
        adapter_metadata={},
        execution_stack={},
        model_artifact_identity=None,
        profile_link=None,
        run_metadata_path=tmp_path / "run_metadata.json",
        transcriptions_path=transcript,
        input_audio_summary=None,
        recording_mode="run_time",
    )

    assert pipeline["contract"]["references"]["status"] == "not_available"
    assert (
        pipeline["model_artifact"]["identity"]["status"]
        == "not_available"
    )
    assert pipeline["links"]["profile"]["status"] == "not_available"


def test_evaluation_metadata_links_the_source_run_and_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluate_stub = types.ModuleType("egra_eval2.evaluate")
    evaluate_stub.aggregate_row_scores = lambda *args, **kwargs: {}
    evaluate_stub.evaluate_rows = lambda value: value
    monkeypatch.setitem(sys.modules, "egra_eval2.evaluate", evaluate_stub)
    sys.modules.pop("eval_pipeline2", None)
    from eval_pipeline2 import write_evaluation_metadata

    output = tmp_path / "output"
    run_name = "model_run"
    run_dir = output / "transcripts" / run_name
    run_dir.mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "metadata_schema_version": 2,
                "inference_setup_id": "model",
                "inference_profile": {
                    "profile_schema_version": 2,
                    "inference_setup_id": "model",
                    "inference_library": "transformers",
                    "adapter": "ctc",
                    "artifact": "model",
                    "output_units": "orthographic",
                }
            }
        ),
        encoding="utf-8",
    )
    evaluation_root = output / "evaluations" / run_name
    manifest_dir = evaluation_root / "manifests"
    manifest_dir.mkdir(parents=True)
    manifest = manifest_dir / "ref_manifest.clean.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    base = evaluation_root / "orthographic"
    base.mkdir()
    (base / "egra_eval_detailed.csv").write_text("a\n", encoding="utf-8")
    (base / "egra_eval_summary.txt").write_text("GLOBAL\n", encoding="utf-8")
    reference_metadata = tmp_path / "reference_views.metadata.json"
    reference_metadata.write_text("{}\n", encoding="utf-8")

    path = write_evaluation_metadata(
        base=base,
        manifest_in=str(manifest),
        requested_representation="orthographic",
        scoring_units="orthographic",
        namespace="orthographic",
        reference_metadata_path=reference_metadata,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 2
    pipeline = payload["pipeline_provenance"]
    assert pipeline["stage"] == "evaluation"
    assert pipeline["recording"]["mode"] == "evaluation_time"
    assert pipeline["links"]["source_run_metadata"]["exists"] is True
    assert pipeline["links"]["source_manifest"]["exists"] is True
    assert pipeline["links"]["reference_view_metadata"]["exists"] is True
    assert pipeline["links"]["summary"]["exists"] is True
