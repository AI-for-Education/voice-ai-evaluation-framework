#!/usr/bin/env python3
"""Backward-compatible entrypoint for the legacy NeMo inference API."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Sequence


_LEGACY_MODEL_PREFIX = ("nemo_inference", "models")


def remap_legacy_model_path(model_path: str) -> str:
    """Map a missing old package-relative model path to its new model store."""
    original = Path(model_path)
    if original.exists():
        return str(original)

    # Normalize separators without treating a Windows path as POSIX. Only the
    # exact former package/model prefix is eligible for compatibility remapping.
    normalized = PurePosixPath(model_path.replace("\\", "/"))
    if normalized.parts[:2] != _LEGACY_MODEL_PREFIX:
        return model_path

    suffix = normalized.parts[2:]
    if not suffix:
        return model_path
    replacement = Path(__file__).resolve().parent / "inference" / "nemo" / "models"
    replacement = replacement.joinpath(*suffix)
    return str(replacement)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the complete legacy CLI while keeping heavy NeMo imports lazy."""
    from inference.nemo.infer import legacy_main

    legacy_main(argv, model_path_mapper=remap_legacy_model_path)


if __name__ == "__main__":
    main()
