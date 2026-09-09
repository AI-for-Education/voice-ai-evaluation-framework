#!/usr/bin/env python3
"""Offline profile-driven Omnilingual ASR inference entrypoint."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Sequence

from inference.omnilingual.backend import OmnilingualBackend
from inference.omnilingual.cache import configure_model_cache, prepared_model_identity
from inference.profile import ProfileError, load_profile
from inference.runner import run_backend


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline Omnilingual ASR inference")
    parser.add_argument("--inference_profile", required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--root_audio_dir", default=None)
    inputs.add_argument("--audio_manifest", default=None)
    parser.add_argument("--output_root", default="input_output_data/output")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument(
        "--resume_run",
        help="Existing .<run>.in_progress directory to resume",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    if args.batch_size < 1:
        raise SystemExit("--batch_size must be at least 1")
    try:
        profile = load_profile(
            args.inference_profile, expected_inference_library="omnilingual"
        )
        cache_path = configure_model_cache(profile, create=False)
        model_identity = prepared_model_identity(profile, cache_path)
        with warnings.catch_warnings(record=True) as startup_warnings:
            warnings.simplefilter("always")
            backend = OmnilingualBackend(profile, batch_size=args.batch_size)
    except (ProfileError, RuntimeError, OSError) as exc:
        raise SystemExit(f"Unable to initialize Omnilingual inference: {exc}") from exc

    try:
        return run_backend(
            backend=backend,
            profile=profile,
            profile_path=args.inference_profile,
            model_path=str(cache_path),
            model_identity=model_identity,
            root_audio_dir=args.root_audio_dir,
            audio_manifest=args.audio_manifest,
            output_root=args.output_root,
            batch_size=args.batch_size,
            smoke_test=args.smoke_test,
            startup_warnings=startup_warnings,
            checkpoint=True,
            resume_run=args.resume_run,
        )
    except Exception as exc:
        raise SystemExit(f"Omnilingual inference failed: {exc}") from exc
    finally:
        backend.close()


if __name__ == "__main__":
    main()
