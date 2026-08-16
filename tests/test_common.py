from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from inference.common import (
    discover_wavs,
    discover_wavs_from_manifest,
    load_audio_and_resample,
    resolve_audio_paths,
)


def _write_wav(path: Path, *, sample_rate: int = 8000, channels: int = 1) -> None:
    samples = (np.sin(np.linspace(0, 8 * np.pi, sample_rate // 20)) * 16000).astype("<i2")
    if channels == 2:
        samples = np.column_stack((samples, -samples)).reshape(-1)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(samples.tobytes())


def test_discover_wavs_is_recursive_case_insensitive_and_sorted(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    _write_wav(nested / "B.WAV")
    _write_wav(tmp_path / "a.wav")
    (tmp_path / "ignore.mp3").write_bytes(b"not audio")

    assert discover_wavs(str(tmp_path)) == [
        str(tmp_path / "a.wav"),
        str(nested / "B.WAV"),
    ]


def test_manifest_preserves_first_occurrence_order(tmp_path: Path) -> None:
    manifest = tmp_path / "segments.jsonl"
    rows = [
        {"audio_filepath": "b.wav"},
        {"audio_filepath": "a.wav"},
        {"audio_filepath": "b.wav"},
        {"not_audio": "ignored"},
    ]
    manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    assert discover_wavs_from_manifest(str(manifest)) == ["b.wav", "a.wav"]
    assert resolve_audio_paths(root_audio_dir=None, audio_manifest=str(manifest)) == [
        "b.wav",
        "a.wav",
    ]


def test_empty_manifest_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "empty.jsonl"
    manifest.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="No valid audio_filepath"):
        resolve_audio_paths(root_audio_dir=None, audio_manifest=str(manifest))


def test_audio_is_mixed_to_mono_and_resampled(tmp_path: Path) -> None:
    source = tmp_path / "stereo.wav"
    _write_wav(source, sample_rate=8000, channels=2)

    audio, sample_rate, duration, was_resampled = load_audio_and_resample(
        str(source), 16000
    )

    assert audio.ndim == 1
    assert audio.dtype == np.float32
    assert sample_rate == 16000
    assert duration == pytest.approx(0.05, rel=0.02)
    assert was_resampled is True
