#!/usr/bin/env python3
"""Framework-independent audio input helpers for ASR inference backends."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf


def discover_wavs(root_dir: str) -> list[str]:
    """Return WAV files recursively in deterministic path order."""
    root = Path(root_dir)
    if not root.is_dir():
        raise SystemExit(f"--root_audio_dir directory not found: {root}")

    return [
        str(path)
        for path in sorted(
            (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".wav"),
            key=lambda path: str(path).lower(),
        )
    ]


def discover_wavs_from_manifest(manifest_path: str) -> list[str]:
    """Load unique audio_filepath values while preserving manifest order."""
    path = Path(manifest_path)
    if not path.is_file():
        raise SystemExit(f"--manifest_in file not found: {path}")

    text = path.read_text(encoding="utf-8")
    rows: list[dict[str, Any]] = []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            rows = [parsed]
        elif isinstance(parsed, list):
            rows = [row for row in parsed if isinstance(row, dict)]
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        index = 0
        while index < len(text):
            while index < len(text) and text[index].isspace():
                index += 1
            if index >= len(text):
                break
            try:
                row, next_index = decoder.raw_decode(text, index)
            except json.JSONDecodeError:
                index += 1
                continue
            if isinstance(row, dict):
                rows.append(row)
            index = next_index

    audio_paths: list[str] = []
    seen: set[str] = set()
    for row in rows:
        audio_path = str(row.get("audio_filepath", "") or "").strip()
        if audio_path and audio_path not in seen:
            seen.add(audio_path)
            audio_paths.append(audio_path)
    return audio_paths


def resolve_audio_paths(
    *,
    root_audio_dir: str | None,
    audio_manifest: str | None,
) -> list[str]:
    """Resolve a backend's shared directory-or-manifest input contract."""
    if audio_manifest:
        audio_paths = discover_wavs_from_manifest(audio_manifest)
        if not audio_paths:
            raise SystemExit(f"No valid audio_filepath rows found in: {audio_manifest}")
        return audio_paths

    if not root_audio_dir:
        raise SystemExit("Provide --root_audio_dir or --audio_manifest.")
    audio_paths = discover_wavs(root_audio_dir)
    if not audio_paths:
        raise SystemExit(f"No .wav files found under: {root_audio_dir}")
    return audio_paths


def load_audio_and_resample(
    path: str,
    target_sr: int,
) -> tuple[np.ndarray, int, float, bool]:
    """Load mono float32 audio and resample it to the requested sample rate."""
    if target_sr <= 0:
        raise ValueError(f"invalid target sample rate: {target_sr}")

    audio, source_sr = sf.read(path, always_2d=False)
    if source_sr <= 0:
        raise ValueError(f"invalid source sample rate: {source_sr}")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise ValueError("audio is empty")
    duration = float(len(audio) / source_sr)

    was_resampled = source_sr != target_sr
    if was_resampled:
        audio = librosa.resample(audio, orig_sr=source_sr, target_sr=target_sr)

    return np.asarray(audio, dtype=np.float32), target_sr, duration, was_resampled
