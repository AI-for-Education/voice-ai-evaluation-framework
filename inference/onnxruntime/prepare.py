#!/usr/bin/env python3
"""Resume-safe preparation of the Exp41 FP32 and INT8 ONNX bundle."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from inference.onnxruntime.artifacts import (
    EXP41_SOURCE_NAME,
    FP32_MODEL_NAME,
    INT8_MODEL_NAME,
    ArtifactError,
    export_nemo_to_fp32_onnx,
    quantize_fp32_to_int8,
    verify_artifact,
)

DEFAULT_SOURCE = Path("/nemo-models") / EXP41_SOURCE_NAME
DEFAULT_OUTPUT = Path("/models/swahili-exp41-ctc")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the internal Exp41 NeMo checkpoint to FP32 ONNX and derive "
            "a mobile-target INT8 ONNX artifact"
        )
    )
    parser.add_argument("--source_nemo", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--fp32_only",
        action="store_true",
        help="Prepare and verify FP32 ONNX without creating INT8 ONNX",
    )
    return parser.parse_args(argv)


def prepare(
    source_nemo: str | Path,
    output_dir: str | Path,
    *,
    fp32_only: bool = False,
) -> dict[str, Path]:
    """Create missing stages and verify completed stages before skipping them."""
    source = Path(source_nemo)
    destination = Path(output_dir)
    fp32_path = destination / FP32_MODEL_NAME
    int8_path = destination / INT8_MODEL_NAME

    if destination.exists():
        print(
            f"[INFO] Existing ONNX bundle found; verifying before resume: {destination}"
        )
        verify_artifact(fp32_path)
        print(f"[SKIP] Verified complete FP32 artifact: {fp32_path}")
    else:
        print(f"[INFO] Exporting verified Exp41 checkpoint: {source}")
        export_nemo_to_fp32_onnx(source, destination)
        print(f"[DONE] FP32 ONNX: {fp32_path}")

    result = {"fp32": fp32_path}
    if fp32_only:
        return result

    if int8_path.exists():
        verify_artifact(int8_path)
        print(f"[SKIP] Verified complete INT8 artifact: {int8_path}")
    else:
        print(f"[INFO] Quantizing FP32 ONNX to mobile-target INT8: {fp32_path}")
        quantize_fp32_to_int8(destination)
        print(f"[DONE] INT8 ONNX: {int8_path}")
    result["int8"] = int8_path
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        prepared = prepare(
            args.source_nemo,
            args.output_dir,
            fp32_only=args.fp32_only,
        )
    except ArtifactError as exc:
        raise SystemExit(f"Unable to prepare Exp41 ONNX artifacts: {exc}") from exc

    print("[SUMMARY] Prepared and verified artifacts:")
    for label, path in prepared.items():
        print(f"  {label}: {path}")
    print(
        "[INFO] These are deployment variants of Exp41, not separately trained models."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
