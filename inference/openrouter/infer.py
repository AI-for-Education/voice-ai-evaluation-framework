#!/usr/bin/env python3
"""Profile-driven OpenRouter ASR inference."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from inference.openrouter.backend import OpenRouterASRBackend, OpenRouterFatalError
from inference.profile import ProfileError, load_profile
from inference.runner import run_backend


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "OpenRouter ASR through explicit global-or-EU routing with "
            "fail-closed ZDR enforcement"
        )
    )
    parser.add_argument(
        "--inference_profile",
        required=True,
        help="Tracked OpenRouter inference profile YAML",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--root_audio_dir", help="Directory containing pre-segmented WAV files"
    )
    input_group.add_argument(
        "--audio_manifest", help="Manifest containing ordered audio_filepath values"
    )
    parser.add_argument("--output_root", default="input_output_data/output")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Maximum concurrent one-audio requests (default: 4)",
    )
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
            args.inference_profile,
            expected_inference_library="openrouter",
        )
        backend = OpenRouterASRBackend(profile)
        return run_backend(
            backend=backend,
            profile=profile,
            profile_path=args.inference_profile,
            model_path=None,
            model_identity=backend.model_identity(),
            root_audio_dir=args.root_audio_dir,
            audio_manifest=args.audio_manifest,
            output_root=args.output_root,
            batch_size=args.batch_size,
            smoke_test=args.smoke_test,
            checkpoint=True,
            resume_run=args.resume_run,
        )
    except (ProfileError, OpenRouterFatalError, RuntimeError, OSError, ValueError) as exc:
        raise SystemExit(f"OpenRouter inference failed: {exc}") from exc


if __name__ == "__main__":
    main()
