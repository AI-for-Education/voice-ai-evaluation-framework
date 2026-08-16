"""Shared contracts for framework-specific ASR inference backends."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class TranscriptionResult:
    """One framework-neutral ASR result row."""

    audio_filepath: str
    duration: float
    pred_text: str
    error: str | None = None
    raw_pred_text: str | None = None

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        if row["error"] is None:
            row.pop("error")
        if row["raw_pred_text"] is None:
            row.pop("raw_pred_text")
        row["duration"] = round(float(row["duration"]), 3)
        return row


@runtime_checkable
class ASRBackend(Protocol):
    """Small interface implemented by every inference-family adapter."""

    def transcribe_batch(self, audio_paths: Sequence[str]) -> list[TranscriptionResult]:
        """Transcribe paths in order and return exactly one result per path."""

    def metadata(self) -> dict[str, Any]:
        """Return serializable runtime metadata for the current backend."""

    def close(self) -> None:
        """Release model resources held by the backend."""
