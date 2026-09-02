#!/usr/bin/env python3
"""Profile-driven local Sherpa-ONNX inference entrypoint."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Sequence

from inference.profile import ProfileError, load_profile, resolve_model_path
from inference.runner import run_backend
from inference.sherpa_onnx.backend import SherpaOnnxOnlineTransducerBackend


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline profile-driven Sherpa-ONNX transcription"
    )
    parser.add_argument(
        "--inference_profile",
        "--model_config",
        dest="inference_profile",
        required=True,
        help="Inference profile YAML (--model_config is deprecated)",
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
            expected_inference_library="sherpa_onnx",
        )
        model_path = resolve_model_path(profile)
        print(f"[INFO] Loading local model: {model_path}")
        with warnings.catch_warnings(record=True) as startup_warnings:
            warnings.simplefilter("always")
            backend = SherpaOnnxOnlineTransducerBackend(
                profile,
                model_path,
                num_threads=args.num_threads,
            )
    except (ProfileError, RuntimeError, OSError) as exc:
        raise SystemExit(f"Unable to initialize Sherpa-ONNX inference: {exc}") from exc

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
