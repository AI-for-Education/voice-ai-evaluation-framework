#!/usr/bin/env python3
"""Profile-driven offline Hugging Face ASR inference."""

from __future__ import annotations

import argparse
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
        "--model_config",
        required=True,
        help="Tracked Transformers model profile YAML",
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
            "Output base; results are written below transcripts/<model>_<UTC timestamp> "
            "(default: input_output_data/output)"
        ),
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Write below smoke_tests/transcripts/<model>_<UTC timestamp>",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    if args.batch_size < 1:
        raise SystemExit("--batch_size must be at least 1")
    try:
        print(f"[INFO] Loading profile: {args.model_config}")
        profile = load_profile(args.model_config, expected_framework="transformers")
        model_path = resolve_model_path(profile)
        print(f"[INFO] Loading local model: {model_path}")
        backend = create_backend(profile, model_path)
    except (ProfileError, RuntimeError, OSError) as exc:
        raise SystemExit(f"Unable to initialize Transformers inference: {exc}") from exc

    print(f"[INFO] Profile: {profile.id} ({profile.adapter})")
    print(f"[INFO] Local model: {model_path}")
    try:
        output_manifest = run_backend(
            backend=backend,
            profile=profile,
            profile_path=args.model_config,
            model_path=str(model_path),
            root_audio_dir=args.root_audio_dir,
            audio_manifest=args.audio_manifest,
            output_root=args.output_root,
            batch_size=args.batch_size,
            smoke_test=args.smoke_test,
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
