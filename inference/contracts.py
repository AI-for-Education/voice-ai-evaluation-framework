"""Shared contracts for inference-library adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class TranscriptionResult:
    """One inference-library-neutral ASR result row."""

    audio_filepath: str
    duration: float
    pred_text: str
    error: str | None = None

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        if row["error"] is None:
            row.pop("error")
        row["duration"] = round(float(row["duration"]), 3)
        return row


@runtime_checkable
class ASRBackend(Protocol):
    """Implementation interface shared by every inference adapter."""

    def transcribe_batch(self, audio_paths: Sequence[str]) -> list[TranscriptionResult]:
        """Transcribe paths in order and return exactly one result per path."""

    def metadata(self) -> dict[str, Any]:
        """Return serializable adapter and execution metadata.

        Adapters with file-based artifacts should include
        ``selected_artifact_files`` as model-directory-relative paths so the
        shared provenance layer identifies only files used by the run.
        """

    def close(self) -> None:
        """Release model resources held by the backend."""
