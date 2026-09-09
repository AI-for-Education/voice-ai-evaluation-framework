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


def test_atomic_json_write_retries_a_transient_windows_file_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "run_state.json"
    original_replace = Path.replace
    attempts = 0

    def temporarily_locked(source: Path, target: Path) -> Path:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("simulated transient Windows file lock")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", temporarily_locked)
    monkeypatch.setattr(runner_module.time, "sleep", lambda _seconds: None)

    runner_module._write_json_atomic(destination, {"completed_items": 4})

    assert attempts == 3
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "completed_items": 4
    }


class LowCoverageBackend(FakeBackend):
    def transcribe_batch(self, audio_paths: Sequence[str]) -> list[TranscriptionResult]:
        values = {
            "first.wav": "kept",
            "second.wav": "",
            "third.wav": "   ",
        }
        return [
            TranscriptionResult(path, 1.23456, values[Path(path).name])
            for path in audio_paths
        ]


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
    assert metadata["output"]["hypotheses"] == {
        "counts": {"total": 3, "non_empty": 3, "blank": 0, "errors": 0},
        "coverage": {
            "status": "ok",
            "warning_code": None,
            "non_empty_fraction": 1.0,
            "minimum_expected_non_empty_fraction": 0.5,
            "affects_evaluation": False,
        },
    }
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
    assert metadata["output"]["hypotheses"]["counts"] == {
        "total": 3,
        "non_empty": 2,
        "blank": 1,
        "errors": 1,
    }
    assert metadata["output"]["hypotheses"]["coverage"]["status"] == "ok"
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


def test_low_coverage_warning_is_structured_and_preserves_hypotheses(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = tmp_path / "input.jsonl"
    _manifest(manifest)

    output = run_backend(
        backend=LowCoverageBackend(),
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
    metadata = json.loads(
        (output.parent / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert [row["pred_text"] for row in rows] == ["kept", "", "   "]
    assert metadata["output"]["hypotheses"] == {
        "counts": {"total": 3, "non_empty": 1, "blank": 2, "errors": 0},
        "coverage": {
            "status": "warning",
            "warning_code": "low_non_empty_hypothesis_coverage",
            "non_empty_fraction": pytest.approx(1 / 3),
            "minimum_expected_non_empty_fraction": 0.5,
            "affects_evaluation": False,
        },
    }
    assert "[WARNING] Low non-empty hypothesis coverage: 1/3" in capsys.readouterr().out


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


def test_checkpoint_retains_interrupted_run_and_resume_skips_completed_rows(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "input.jsonl"
    _manifest(manifest)
    output_root = tmp_path / "output"
    interrupted = ExplodingBackend()

    with pytest.raises(RuntimeError, match="mock catastrophic backend failure"):
        run_backend(
            backend=interrupted,
            profile=_profile(),
            profile_path="profile.yaml",
            model_path=None,
            model_identity={
                "kind": "remote_api_model",
                "requested_model": "provider/model",
            },
            root_audio_dir=None,
            audio_manifest=str(manifest),
            output_root=str(output_root),
            batch_size=2,
            checkpoint=True,
        )

    staging_runs = list((output_root / "transcripts").glob(".*.in_progress"))
    assert len(staging_runs) == 1
    staging = staging_runs[0]
    state = json.loads((staging / "run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "in_progress"
    assert state["completed_items"] == 2

    class RecordingBackend(FakeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.paths: list[str] = []

        def transcribe_batch(
            self, audio_paths: Sequence[str]
        ) -> list[TranscriptionResult]:
            self.paths.extend(audio_paths)
            return super().transcribe_batch(audio_paths)

    resumed = RecordingBackend()
    output = run_backend(
        backend=resumed,
        profile=_profile(),
        profile_path="profile.yaml",
        model_path=None,
        model_identity={
            "kind": "remote_api_model",
            "requested_model": "provider/model",
        },
        root_audio_dir=None,
        audio_manifest=str(manifest),
        output_root=str(output_root),
        batch_size=2,
        checkpoint=True,
        resume_run=str(staging),
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["audio_filepath"] for row in rows] == [
        "first.wav",
        "second.wav",
        "third.wav",
    ]
    assert resumed.paths == ["third.wav"]
    final_state = json.loads(
        (output.parent / "run_state.json").read_text(encoding="utf-8")
    )
    assert final_state["status"] == "completed"
    assert final_state["completed_items"] == 3
    assert not staging.exists()


def _checkpoint_arguments(tmp_path: Path) -> dict:
    manifest = tmp_path / "input.jsonl"
    _manifest(manifest)
    return dict(
        profile=_profile(), profile_path="profile.yaml", model_path=None,
        model_identity={"kind": "remote_api_model", "requested_model": "provider/model"},
        root_audio_dir=None, audio_manifest=str(manifest),
        output_root=str(tmp_path / "output"), batch_size=2, checkpoint=True,
    )


@pytest.mark.parametrize("failure", [KeyboardInterrupt, RuntimeError])
def test_each_attempt_retains_its_own_observations(tmp_path: Path, failure) -> None:
    class ObservedBackend(FakeBackend):
        def __init__(self, *, interrupt: bool):
            super().__init__()
            self.interrupt = interrupt
            self.count = 0

        def transcribe_batch(self, paths):
            if self.interrupt and self.count:
                warnings.warn("warning before interruption", RuntimeWarning)
                raise failure("stop")
            self.count += len(paths)
            return super().transcribe_batch(paths)

        def metadata(self):
            return {
                "device": "first-device" if self.interrupt else "second-device",
                "statistics": {
                    "successful_requests": self.count,
                    "models_returned": {"provider/model": self.count},
                    "usage": {"cost": self.count * 0.1},
                },
            }

    args = _checkpoint_arguments(tmp_path)
    with pytest.raises(failure):
        run_backend(backend=ObservedBackend(interrupt=True), **args)
    staging = next((tmp_path / "output/transcripts").glob(".*.in_progress"))
    first = json.loads((staging / "run_state.json").read_text())["attempts"][0]
    assert first["status"] == ("interrupted" if failure is KeyboardInterrupt else "failed")
    assert first["failure_type"] == failure.__name__
    assert first["completed_items"] == 2
    assert first["observations_complete"] is True
    assert first["ended_at"] is not None
    assert first["warnings"][0]["message"] == "warning before interruption"

    output = run_backend(
        backend=ObservedBackend(interrupt=False), resume_run=str(staging), **args
    )
    metadata = json.loads((output.parent / "run_metadata.json").read_text())
    attempts = metadata["attempts"]
    assert attempts[0] == first
    assert [item["attempt_id"] for item in attempts] == [1, 2]
    assert [item["start_item"] for item in attempts] == [0, 2]
    assert [item["completed_items"] for item in attempts] == [2, 1]
    assert attempts[1]["status"] == "completed"
    for index, count in enumerate((2, 1)):
        stats = attempts[index]["inference_adapter"]["statistics"]
        assert stats["successful_requests"] == count
        assert stats["models_returned"] == {"provider/model": count}
        assert stats["usage"]["cost"] == pytest.approx(count * 0.1)
        assert {"execution_stack", "repository", "model_artifact", "started_at", "ended_at"} <= attempts[index].keys()
    assert metadata["output"]["results"] == 3
    assert metadata["attempt_history_complete"] is True
    assert metadata["observation_attempt_id"] == 2
    assert metadata["inference_adapter"] == attempts[1]["inference_adapter"]
    assert attempts[0]["inference_adapter"]["device"] == "first-device"
    assert attempts[1]["inference_adapter"]["device"] == "second-device"


@pytest.mark.parametrize("tail", [b"", b'{"audio_filepath": "third', b'{"pred_text": "\xe2\x82'])
def test_resume_recovers_rows_written_before_checkpoint_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tail: bytes
) -> None:
    args = _checkpoint_arguments(tmp_path)
    original_write = runner_module._write_json_atomic

    def interrupted_write(path, payload):
        if payload.get("completed_items", 0):
            raise OSError("storage unavailable after transcript flush")
        original_write(path, payload)

    with monkeypatch.context() as patch:
        patch.setattr(runner_module, "_write_json_atomic", interrupted_write)
        with pytest.raises(OSError, match="storage unavailable"):
            run_backend(backend=FakeBackend(), **args)
    staging = next((tmp_path / "output/transcripts").glob(".*.in_progress"))
    state = json.loads((staging / "run_state.json").read_text())
    assert state["completed_items"] == 0
    manifest = staging / "transcriptions.jsonl"
    with manifest.open("ab") as output:
        output.write(tail)

    output = run_backend(backend=CompletedStateBackend(), resume_run=str(staging), **args)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["audio_filepath"] for row in rows] == ["first.wav", "second.wav", "third.wav"]
    metadata = json.loads((output.parent / "run_metadata.json").read_text())
    first, resumed = metadata["attempts"]
    assert first["status"] == "interrupted"
    assert first["completed_items"] == 2
    assert first["ended_at"] is None
    assert first["observations_complete"] is False
    assert resumed["inference_adapter"]["processed"] == 1


@pytest.mark.parametrize("committed_items,tail", [(3, b""), (2, b"broken\n")])
def test_resume_does_not_repair_committed_or_non_tail_corruption(
    tmp_path: Path, committed_items: int, tail: bytes
) -> None:
    args = _checkpoint_arguments(tmp_path)
    with pytest.raises(RuntimeError):
        run_backend(backend=ExplodingBackend(), **args)
    staging = next((tmp_path / "output/transcripts").glob(".*.in_progress"))
    state_path = staging / "run_state.json"
    state = json.loads(state_path.read_text())
    state["completed_items"] = committed_items
    state_path.write_text(json.dumps(state))
    manifest = staging / "transcriptions.jsonl"
    original = manifest.read_bytes() + tail
    manifest.write_bytes(original)
    with pytest.raises(ValueError, match="checkpoint"):
        run_backend(backend=FakeBackend(), resume_run=str(staging), **args)
    assert manifest.read_bytes() == original


def test_resume_can_publish_a_completed_checkpoint_and_marks_missing_old_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _checkpoint_arguments(tmp_path)
    original_rename = Path.rename

    def fail_publication(path, destination):
        if path.name.endswith(".in_progress"):
            raise PermissionError("publication interrupted")
        return original_rename(path, destination)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "rename", fail_publication)
        with pytest.raises(PermissionError):
            run_backend(backend=FakeBackend(), **args)
    staging = next((tmp_path / "output/transcripts").glob(".*.in_progress"))
    state_path = staging / "run_state.json"
    state = json.loads(state_path.read_text())
    # Reproduce an older checkpoint, killed after completion but before rename.
    state["status"] = "completed"
    del state["attempts"], state["attempt_history_complete"]
    state_path.write_text(json.dumps(state))
    output = run_backend(backend=CompletedStateBackend(), resume_run=str(staging), **args)
    metadata = json.loads((output.parent / "run_metadata.json").read_text())
    assert metadata["output"]["results"] == 3
    assert metadata["inference_adapter"]["processed"] == 0
    assert metadata["attempt_history_complete"] is False
    assert not staging.exists()
