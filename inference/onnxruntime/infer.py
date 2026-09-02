#!/usr/bin/env python3
"""Profile-driven PC inference for mobile-target ONNX artifacts."""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Sequence
from pathlib import Path

from inference.onnxruntime.android_backend import AndroidParityCtcBackend
from inference.onnxruntime.backend import OnnxRuntimeCtcBackend
from inference.profile import ProfileError, load_profile, resolve_model_path
from inference.runner import run_backend


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PC validation of Exp41 FP32/INT8 ONNX with greedy CTC"
    )
    parser.add_argument(
        "--inference_profile",
        dest="inference_profile",
        required=True,
        help="Inference profile YAML",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--root_audio_dir", default=None)
    input_group.add_argument("--audio_manifest", default=None)
    parser.add_argument("--output_root", default="input_output_data/output")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_threads", type=int, default=2)
    parser.add_argument("--smoke_test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    if args.batch_size < 1:
        raise SystemExit("--batch_size must be at least 1")
    if args.num_threads < 1:
        raise SystemExit("--num_threads must be at least 1")

    try:
        print(f"[INFO] Loading inference profile: {args.inference_profile}")
        profile = load_profile(
            args.inference_profile,
            expected_inference_library="onnxruntime",
        )
        model_path = resolve_model_path(profile)
        if profile.adapter == "android_ctc" and (
            args.batch_size != 1 or args.num_threads != 1
        ):
            raise ProfileError(
                "Controlled Android preprocessing requires --batch_size 1 "
                "and --num_threads 1"
            )
        print(f"[INFO] Loading controlled model: {model_path}")
        with warnings.catch_warnings(record=True) as startup_warnings:
            warnings.simplefilter("always")
            if profile.adapter == "android_ctc":
                backend = AndroidParityCtcBackend(
                    profile,
                    model_path,
                    num_threads=1,
                    expected_ort_version=None,
                )
            else:
                backend = OnnxRuntimeCtcBackend(
                    profile,
                    model_path,
                    num_threads=args.num_threads,
                )
    except (ProfileError, RuntimeError, OSError) as exc:
        raise SystemExit(f"Unable to initialize ONNX Runtime inference: {exc}") from exc

    return run_backend(
        backend=backend,
        profile=profile,
        profile_path=args.inference_profile,
        model_path=str(model_path),
        root_audio_dir=args.root_audio_dir,
        audio_manifest=args.audio_manifest,
        output_root=args.output_root,
        batch_size=args.batch_size,
        smoke_test=args.smoke_test,
        startup_warnings=startup_warnings,
    )


if __name__ == "__main__":
    main()
