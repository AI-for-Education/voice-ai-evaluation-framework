#!/usr/bin/env python3
"""Profile-driven native TorchScript streaming-transducer inference."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Sequence

from inference.profile import ProfileError, load_profile, resolve_model_path
from inference.runner import run_backend
from inference.torch.backend import TorchScriptStreamingTransducerBackend


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline profile-driven native TorchScript transcription"
    )
    parser.add_argument(
        "--inference_profile",
        required=True,
        help="Inference profile YAML",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--root_audio_dir", default=None)
    input_group.add_argument("--audio_manifest", default=None)
    parser.add_argument("--output_root", default="input_output_data/output")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_threads", type=int, default=2)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--fail_on_error", action="store_true")
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
            expected_inference_library="torch",
        )
        model_path = resolve_model_path(profile)
        print(f"[INFO] Loading local model: {model_path}")
        with warnings.catch_warnings(record=True) as startup_warnings:
            warnings.simplefilter("always")
            backend = TorchScriptStreamingTransducerBackend(
                profile,
                model_path,
                num_threads=args.num_threads,
            )
    except (ProfileError, RuntimeError, OSError) as exc:
        raise SystemExit(
            f"Unable to initialize native TorchScript inference: {exc}"
        ) from exc

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
        fail_on_error=args.fail_on_error,
        startup_warnings=startup_warnings,
    )


if __name__ == "__main__":
    main()
