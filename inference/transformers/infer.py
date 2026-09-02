#!/usr/bin/env python3
"""Profile-driven offline Hugging Face ASR inference."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Sequence

from inference.profile import ProfileError, load_profile, resolve_model_path
from inference.runner import run_backend
from inference.transformers.factory import create_backend


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline profile-driven Transformers ASR inference for pre-segmented WAV files"
        )
    )
    parser.add_argument(
        "--inference_profile",
        "--model_config",
        dest="inference_profile",
        required=True,
        help="Tracked Transformers inference profile YAML (--model_config is deprecated)",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--root_audio_dir",
        help="Directory containing pre-segmented WAV files (searched recursively)",
    )
    input_group.add_argument(
        "--audio_manifest",
        help="Manifest containing ordered audio_filepath values to transcribe",
    )
    parser.add_argument(
        "--output_root",
        default="input_output_data/output",
        help=(
            "Output base; results are written below "
            "transcripts/<inference_setup_id>_<UTC timestamp> "
            "(default: input_output_data/output)"
        ),
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help=(
            "Write below "
            "smoke_tests/transcripts/<inference_setup_id>_<UTC timestamp>"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    if args.batch_size < 1:
        raise SystemExit("--batch_size must be at least 1")
    try:
        print(f"[INFO] Loading inference profile: {args.inference_profile}")
        profile = load_profile(
            args.inference_profile,
            expected_inference_library="transformers",
        )
        model_path = resolve_model_path(profile)
        print(f"[INFO] Loading local model: {model_path}")
        with warnings.catch_warnings(record=True) as startup_warnings:
            warnings.simplefilter("always")
            backend = create_backend(profile, model_path)
    except (ProfileError, RuntimeError, OSError) as exc:
        raise SystemExit(f"Unable to initialize Transformers inference: {exc}") from exc

    print(
        f"[INFO] Inference setup: {profile.inference_setup_id} "
        f"({profile.adapter})"
    )
    print(f"[INFO] Local model: {model_path}")
    try:
        output_manifest = run_backend(
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
    except Exception as exc:
        raise SystemExit(f"Transformers inference failed: {exc}") from exc
    finally:
        # run_backend closes the backend after a normal run. This second call is
        # intentionally idempotent and covers input-resolution failures too.
        backend.close()

    print(f"[INFO] Wrote transcriptions to: {output_manifest}")
    return output_manifest


if __name__ == "__main__":
    main()
