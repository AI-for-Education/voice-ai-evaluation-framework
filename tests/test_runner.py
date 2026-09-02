from __future__ import annotations

import json
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pytest

import inference.runner as runner_module
from egra_eval2.prediction_manifest import load_prediction_manifest
from inference.contracts import TranscriptionResult
from inference.profile import parse_profile
from inference.runner import resolve_transcript_output_dir, run_backend


class FakeBackend:
    def __init__(self) -> None:
        self.closed = False

    def transcribe_batch(self, audio_paths: Sequence[str]) -> list[TranscriptionResult]:
        return [
            TranscriptionResult(path, 1.23456, f"text:{Path(path).name}")
            for path in audio_paths
        ]

    def metadata(self):
        return {"inference_library": "fake", "device": "cpu"}

    def close(self) -> None:
        self.closed = True


class ReorderingBackend(FakeBackend):
    def transcribe_batch(self, audio_paths: Sequence[str]) -> list[TranscriptionResult]:
        return list(reversed(super().transcribe_batch(audio_paths)))


class CompletedStateBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.processed = 0

    def transcribe_batch(self, audio_paths: Sequence[str]) -> list[TranscriptionResult]:
        rows = super().transcribe_batch(audio_paths)
        self.processed += len(rows)
        return rows

    def metadata(self):
        return {
            "inference_library": "fake",
            "device": "cpu",
            "processed": self.processed,
        }


class DiagnosticBackend(FakeBackend):
    def transcribe_batch(self, audio_paths: Sequence[str]) -> list[TranscriptionResult]:
        warnings.warn("mock recoverable decoder warning", RuntimeWarning)
        rows = super().transcribe_batch(audio_paths)
        if "second.wav" in audio_paths:
            index = list(audio_paths).index("second.wav")
            rows[index] = TranscriptionResult(
                audio_filepath="second.wav",
                duration=1.23456,
                pred_text="",
                error="mock audio decode failure",
            )
        return rows

    def metadata(self):
        return {
            "inference_library": "fake",
            "device": "cpu",
            "provider": "MockExecutionProvider",
            "processor_class": "MockProcessor",
            "sampling_rate": 16000,
            "backend_version": "mock-1.0",
        }


class ExplodingBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def transcribe_batch(self, audio_paths: Sequence[str]) -> list[TranscriptionResult]:
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("mock catastrophic backend failure")
        return super().transcribe_batch(audio_paths)


def _profile():
    return parse_profile(
        {
            "parameter_evidence": [
                {
                    "applies_to": ["decoding.strategy"],
                    "level": "exact",
                    "source": "https://example.test/model-card",
                    "rationale": "The source defines greedy decoding.",
                }
            ],
            "profile_schema_version": 2,
            "inference_setup_id": "runner-test",
            "inference_library": "transformers",
            "adapter": "ctc",
            "artifact": "model",
            "decoding": {"strategy": "greedy"},
        }
    )


def _manifest(path: Path) -> None:
    path.write_text(
        "\n".join(
            json.dumps({"audio_filepath": audio})
            for audio in ("first.wav", "second.wav", "third.wav")
        ),
        encoding="utf-8",
    )


def test_runner_preserves_schema_order_writes_metadata_and_reports_progress(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "input.jsonl"
    output_root = tmp_path / "output"
    _manifest(manifest)
    backend = FakeBackend()
    monkeypatch.setenv("PIPELINE_LAUNCH_ORCHESTRATOR", "docker_compose")
    monkeypatch.setenv("PIPELINE_LAUNCHER", "run_transformers_inference.sh")
    monkeypatch.setenv("PIPELINE_COMPOSE_SERVICE", "transformers-asr")
    monkeypatch.setenv(
        "PIPELINE_IMAGE_REFERENCE", "voice-ai-evaluation-framework-asr:latest"
    )

    with warnings.catch_warnings(record=True) as startup_warnings:
        warnings.simplefilter("always")
        warnings.warn("model initialization fallback", RuntimeWarning)

    output = run_backend(
        backend=backend,
        profile=_profile(),
        profile_path="profile.yaml",
        model_path="model",
        root_audio_dir=None,
        audio_manifest=str(manifest),
        output_root=str(output_root),
        batch_size=2,
        startup_warnings=startup_warnings,
    )

    rows = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["audio_filepath"] for row in rows] == [
        "first.wav",
        "second.wav",
        "third.wav",
    ]
    assert set(rows[0]) == {"audio_filepath", "duration", "pred_text"}
    assert rows[0]["duration"] == 1.235
    assert output.parent.parent == output_root / "transcripts"
    assert re.fullmatch(
        r"runner-test_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_UTC",
        output.parent.name,
    )
    metadata = json.loads(
        (output.parent / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["status"] == "completed"
    assert metadata["metadata_schema_version"] == 2
    assert metadata["provenance_schema_version"] == 2
    assert metadata["inference_setup_id"] == "runner-test"
    assert metadata["run_id"] == output.parent.name
    assert metadata["inference_profile"]["inference_setup_id"] == "runner-test"
    assert metadata["pipeline_provenance"]["recording"]["mode"] == "run_time"
    assert "inference_setup" in metadata["pipeline_provenance"]
    assert "execution_stack" in metadata["pipeline_provenance"]
    assert (
        metadata["pipeline_provenance"]["inference_setup"]["audio_preparation"]
        ["input_audio"]["status"]
        == "not_available"
    )
    assert re.fullmatch(
        r"[0-9a-f]{64}",
        metadata["inference_profile_identity"]["canonical_content_sha256"],
    )
    assert metadata["execution_stack"]["batch_size"] == 2
    assert metadata["execution_stack"]["launch_context"]["compose_service"] == (
        "transformers-asr"
    )
    assert (
        metadata["pipeline_provenance"]["execution_stack"]["observed"]
        ["environment"]["launch"]["launcher"]
        == "run_transformers_inference.sh"
    )
    assert len(metadata["warnings"]) == 1
    assert metadata["warnings"][0]["phase"] == "initialization"
    assert metadata["warnings"][0]["category"] == "RuntimeWarning"
    assert metadata["warnings"][0]["message"] == "model initialization fallback"
    assert metadata["inference_profile"]["parameter_evidence"][0]["level"] == "exact"
    assert metadata["inference_profile"]["parameter_evidence"][0]["applies_to"] == [
        "decoding.strategy"
    ]
    assert metadata["output"]["results"] == 3
    assert metadata["output"]["directory"] == str(output.parent)
    assert metadata["output"]["smoke_test"] is False
    assert (
        metadata["pipeline_provenance"]["links"]["transcriptions"]["path"]
        == str(output)
    )
    assert metadata["pipeline_provenance"]["links"]["transcriptions"]["exists"]
    assert backend.closed is True

    captured = capsys.readouterr()
    assert "[INFO] Audio files: 3 | batch_size=2" in captured.out
    assert "[INFO] Completed: 3 file(s), 0 error(s)" in captured.out
    assert "Transcribing runner-test" in captured.err
    assert "3/3" in captured.err

    # This is the same loader used by manifest_pipeline.py for --prediction_manifest.
    merged_input = load_prediction_manifest(str(output))
    assert merged_input["audio_path"].tolist() == [
        "first.wav",
        "second.wav",
        "third.wav",
    ]
    assert merged_input["hyp_text"].tolist() == [
        "text:first.wav",
        "text:second.wav",
        "text:third.wav",
    ]


def test_runner_captures_backend_metadata_after_all_batches(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "input.jsonl"
    _manifest(manifest)
    backend = CompletedStateBackend()

    output = run_backend(
        backend=backend,
        profile=_profile(),
        profile_path="profile.yaml",
        model_path="model",
        root_audio_dir=None,
        audio_manifest=str(manifest),
        output_root=str(tmp_path / "output"),
        batch_size=2,
    )

    rows = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 3
    assert rows[0]["pred_text"] == "text:first.wav"

    metadata = json.loads(
        (output.parent / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["inference_adapter"]["processed"] == 3
    assert "postprocessing" not in metadata


def test_mock_smoke_run_records_errors_warnings_and_execution_stack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "input.jsonl"
    _manifest(manifest)
    monkeypatch.setenv("PIPELINE_LAUNCH_ORCHESTRATOR", "docker_compose")
    monkeypatch.setenv("PIPELINE_LAUNCHER", "run_transformers_inference.sh")
    monkeypatch.setenv("PIPELINE_COMPOSE_SERVICE", "transformers-asr")
    monkeypatch.setenv(
        "PIPELINE_IMAGE_REFERENCE", "voice-ai-evaluation-framework-asr:latest"
    )
    monkeypatch.setenv("PIPELINE_IMAGE_ID", "sha256:mock-image")
    monkeypatch.setenv(
        "PIPELINE_IMAGE_REPO_DIGESTS_JSON",
        '["voice-ai-evaluation-framework-asr@sha256:mock-digest"]',
    )

    output = run_backend(
        backend=DiagnosticBackend(),
        profile=_profile(),
        profile_path="profile.yaml",
        model_path="model",
        root_audio_dir=None,
        audio_manifest=str(manifest),
        output_root=str(tmp_path / "output"),
        batch_size=2,
        smoke_test=True,
    )
    rows = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    metadata = json.loads(
        (output.parent / "run_metadata.json").read_text(encoding="utf-8")
    )

    assert output.parent.parent == tmp_path / "output" / "smoke_tests" / "transcripts"
    assert rows[1]["audio_filepath"] == "second.wav"
    assert rows[1]["error"] == "mock audio decode failure"
    assert metadata["output"]["errors"] == 1
    assert metadata["warnings"][0]["phase"] == "transcription"
    assert metadata["warnings"][0]["message"] == (
        "mock recoverable decoder warning"
    )
    assert metadata["inference_adapter"]["provider"] == "MockExecutionProvider"
    assert metadata["execution_stack"]["launch_context"]["image"] == {
        "reference": "voice-ai-evaluation-framework-asr:latest",
        "id": "sha256:mock-image",
        "repo_digests": [
            "voice-ai-evaluation-framework-asr@sha256:mock-digest"
        ],
    }
    provenance = metadata["pipeline_provenance"]
    assert provenance["inference_setup"]["input_processing"]["observed"][
        "processor_class"
    ] == "MockProcessor"
    assert provenance["execution_stack"]["observed"]["environment"]["launch"] == (
        metadata["execution_stack"]["launch_context"]
    )


def test_fail_on_error_writes_metadata_then_exits_nonzero(tmp_path: Path) -> None:
    manifest = tmp_path / "input.jsonl"
    _manifest(manifest)
    output_root = tmp_path / "output"

    with pytest.raises(SystemExit, match="produced 1 per-file error"):
        run_backend(
            backend=DiagnosticBackend(),
            profile=_profile(),
            profile_path="profile.yaml",
            model_path="model",
            root_audio_dir=None,
            audio_manifest=str(manifest),
            output_root=str(output_root),
            batch_size=2,
            fail_on_error=True,
        )

    metadata_paths = list(output_root.glob("transcripts/*/run_metadata.json"))
    assert len(metadata_paths) == 1
    metadata = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
    assert metadata["output"]["errors"] == 1
    assert metadata["output"]["fail_on_error"] is True


def test_same_second_runs_publish_to_distinct_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "input.jsonl"
    _manifest(manifest)
    candidate = tmp_path / "output" / "transcripts" / "runner-test_fixed_UTC"
    monkeypatch.setattr(
        runner_module,
        "resolve_transcript_output_dir",
        lambda **_kwargs: candidate,
    )

    first = run_backend(
        backend=FakeBackend(),
        profile=_profile(),
        profile_path="profile.yaml",
        model_path="model",
        root_audio_dir=None,
        audio_manifest=str(manifest),
        output_root=str(tmp_path / "output"),
        batch_size=2,
    )
    first_contents = first.read_text(encoding="utf-8")
    second = run_backend(
        backend=FakeBackend(),
        profile=_profile(),
        profile_path="profile.yaml",
        model_path="model",
        root_audio_dir=None,
        audio_manifest=str(manifest),
        output_root=str(tmp_path / "output"),
        batch_size=2,
    )

    assert first.parent == candidate
    assert second.parent == candidate.with_name(f"{candidate.name}__2")
    assert first.read_text(encoding="utf-8") == first_contents
    assert not any(
        path.name.endswith(".in_progress")
        for path in candidate.parent.iterdir()
    )


def test_catastrophic_backend_failure_does_not_publish_partial_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "input.jsonl"
    _manifest(manifest)
    candidate = tmp_path / "output" / "transcripts" / "runner-test_fixed_UTC"
    backend = ExplodingBackend()
    monkeypatch.setattr(
        runner_module,
        "resolve_transcript_output_dir",
        lambda **_kwargs: candidate,
    )

    with pytest.raises(RuntimeError, match="mock catastrophic backend failure"):
        run_backend(
            backend=backend,
            profile=_profile(),
            profile_path="profile.yaml",
            model_path="model",
            root_audio_dir=None,
            audio_manifest=str(manifest),
            output_root=str(tmp_path / "output"),
            batch_size=2,
        )

    assert not candidate.exists()
    assert not list(candidate.parent.iterdir())
    assert backend.closed is True


def test_standard_transcript_layout_supports_smoke_tests(tmp_path: Path) -> None:
    started = datetime(2026, 8, 12, 13, 14, 15, tzinfo=timezone.utc)

    normal = resolve_transcript_output_dir(
        output_root=tmp_path,
        inference_setup_id="BookBot / orthographic",
        smoke_test=False,
        started_at=started,
    )
    smoke = resolve_transcript_output_dir(
        output_root=tmp_path,
        inference_setup_id="BookBot / orthographic",
        smoke_test=True,
        started_at=started,
    )

    assert normal == (
        tmp_path / "transcripts" / "BookBot_orthographic_2026_08_12_13_14_15_UTC"
    )
    assert smoke == (
        tmp_path
        / "smoke_tests"
        / "transcripts"
        / "BookBot_orthographic_2026_08_12_13_14_15_UTC"
    )


def test_runner_rejects_backend_reordering_and_still_closes(tmp_path: Path) -> None:
    manifest = tmp_path / "input.jsonl"
    _manifest(manifest)
    backend = ReorderingBackend()

    with pytest.raises(RuntimeError, match="changed result order"):
        run_backend(
            backend=backend,
            profile=_profile(),
            profile_path="profile.yaml",
            model_path="model",
            root_audio_dir=None,
            audio_manifest=str(manifest),
            output_root=str(tmp_path / "output"),
            batch_size=3,
        )
    assert backend.closed is True


def test_runner_closes_backend_when_input_resolution_fails(tmp_path: Path) -> None:
    backend = FakeBackend()

    with pytest.raises(SystemExit, match="No .wav files found"):
        run_backend(
            backend=backend,
            profile=_profile(),
            profile_path="profile.yaml",
            model_path="model",
            root_audio_dir=str(tmp_path),
            audio_manifest=None,
            output_root=str(tmp_path / "output"),
            batch_size=1,
        )

    assert backend.closed is True
