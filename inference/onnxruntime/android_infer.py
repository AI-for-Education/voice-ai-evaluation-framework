#!/usr/bin/env python3
"""Run the packaged Android INT8 pipeline on PC with controlled semantics."""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Sequence
from pathlib import Path

from inference.onnxruntime.android_backend import AndroidParityCtcBackend
from inference.profile import ProfileError, load_profile, resolve_model_path
from inference.runner import run_backend


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Android-parity validation of the packaged Exp41 INT8 ONNX model"
    )
    parser.add_argument("--model_config", required=True)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--root_audio_dir", default=None)
    input_group.add_argument("--audio_manifest", default=None)
    parser.add_argument("--output_root", default="input_output_data/output")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--smoke_test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    if args.batch_size != 1:
        raise SystemExit("Android parity requires --batch_size 1")
    if args.num_threads != 1:
        raise SystemExit("Android parity requires --num_threads 1")

    try:
        print(f"[INFO] Loading Android-parity profile: {args.model_config}")
        profile = load_profile(args.model_config, expected_framework="onnxruntime")
        model_path = resolve_model_path(profile)
        print(f"[INFO] Loading pinned Android model: {model_path}")
        with warnings.catch_warnings(record=True) as startup_warnings:
            warnings.simplefilter("always")
            backend = AndroidParityCtcBackend(profile, model_path, num_threads=1)
    except (ProfileError, RuntimeError, OSError) as exc:
        raise SystemExit(
            f"Unable to initialize Android-parity ONNX inference: {exc}"
        ) from exc

    return run_backend(
        backend=backend,
        profile=profile,
        profile_path=args.model_config,
        model_path=str(model_path),
        root_audio_dir=args.root_audio_dir,
        audio_manifest=args.audio_manifest,
        output_root=args.output_root,
        batch_size=1,
        smoke_test=args.smoke_test,
        startup_warnings=startup_warnings,
    )


if __name__ == "__main__":
    main()
