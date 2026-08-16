from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pytest

from egra_eval2.nemo_manifest import load_nemo_manifest
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
        return {"framework": "fake", "device": "cpu"}

    def close(self) -> None:
        self.closed = True


class ReorderingBackend(FakeBackend):
    def transcribe_batch(self, audio_paths: Sequence[str]) -> list[TranscriptionResult]:
        return list(reversed(super().transcribe_batch(audio_paths)))


class PostprocessingBackend(FakeBackend):
    def transcribe_batch(self, audio_paths: Sequence[str]) -> list[TranscriptionResult]:
        rows = super().transcribe_batch(audio_paths)
        first = rows[0]
        if first.audio_filepath != "first.wav":
            return rows
        rows[0] = TranscriptionResult(
            audio_filepath=first.audio_filepath,
            duration=first.duration,
            pred_text="one two",
            raw_pred_text="one two three four",
        )
        return rows

    def metadata(self):
        return {
            "framework": "fake",
            "device": "cpu",
            "hallucination_guard": {"max_words_per_second": 8.0},
        }


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
            "id": "runner-test",
            "framework": "transformers",
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
) -> None:
    manifest = tmp_path / "input.jsonl"
    output_root = tmp_path / "output"
    _manifest(manifest)
    backend = FakeBackend()

    output = run_backend(
        backend=backend,
        profile=_profile(),
        profile_path="profile.yaml",
        model_path="model",
        root_audio_dir=None,
        audio_manifest=str(manifest),
        output_root=str(output_root),
        batch_size=2,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
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
    metadata = json.loads((output.parent / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["profile"]["id"] == "runner-test"
    assert metadata["profile"]["parameter_evidence"][0]["level"] == "exact"
    assert metadata["profile"]["parameter_evidence"][0]["applies_to"] == [
        "decoding.strategy"
    ]
    assert metadata["output"]["results"] == 3
    assert metadata["output"]["directory"] == str(output.parent)
    assert metadata["output"]["smoke_test"] is False
    assert backend.closed is True

    captured = capsys.readouterr()
    assert "[INFO] Audio files: 3 | batch_size=2" in captured.out
    assert "[INFO] Completed: 3 file(s), 0 error(s)" in captured.out
    assert "Transcribing runner-test" in captured.err
    assert "3/3" in captured.err

    # This is the same loader used by manifest_pipeline.py for --asr_manifest.
    merged_input = load_nemo_manifest(str(output))
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


def test_runner_summarizes_postprocessing_without_changing_transcript_handoff(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "input.jsonl"
    _manifest(manifest)

    output = run_backend(
        backend=PostprocessingBackend(),
        profile=_profile(),
        profile_path="profile.yaml",
        model_path="model",
        root_audio_dir=None,
        audio_manifest=str(manifest),
        output_root=str(tmp_path / "output"),
        batch_size=2,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["raw_pred_text"] == "one two three four"
    assert rows[0]["pred_text"] == "one two"
    assert "postprocessing" not in rows[0]

    metadata = json.loads((output.parent / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["postprocessing"] == {
        "stage": "post_decode_pre_evaluation",
        "method": "hallucination_guard",
        "scored_text_field": "pred_text",
        "raw_text_field": "raw_pred_text",
        "adjusted_results": 1,
        "total_words_removed": 2,
    }


def test_standard_transcript_layout_supports_smoke_tests(tmp_path: Path) -> None:
    started = datetime(2026, 8, 12, 13, 14, 15, tzinfo=timezone.utc)

    normal = resolve_transcript_output_dir(
        output_root=tmp_path,
        profile_id="BookBot / orthographic",
        smoke_test=False,
        started_at=started,
    )
    smoke = resolve_transcript_output_dir(
        output_root=tmp_path,
        profile_id="BookBot / orthographic",
        smoke_test=True,
        started_at=started,
    )

    assert normal == (
        tmp_path
        / "transcripts"
        / "BookBot_orthographic_2026_08_12_13_14_15_UTC"
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
