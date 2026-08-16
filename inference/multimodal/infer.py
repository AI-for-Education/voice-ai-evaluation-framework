#!/usr/bin/env python3
"""Profile-driven local multimodal audio-to-text inference entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from inference.multimodal.factory import create_backend
from inference.profile import ProfileError, load_profile, resolve_model_path
from inference.runner import run_backend


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline profile-driven multimodal audio-to-text inference"
    )
    parser.add_argument("--model_config", required=True)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--root_audio_dir", default=None)
    input_group.add_argument("--audio_manifest", default=None)
    parser.add_argument("--output_root", default="input_output_data/output")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Runner grouping size; multimodal adapters process files sequentially",
    )
    parser.add_argument("--smoke_test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    if args.batch_size < 1:
        raise SystemExit("--batch_size must be at least 1")

    try:
        print(f"[INFO] Loading profile: {args.model_config}")
        profile = load_profile(args.model_config, expected_framework="multimodal")
        model_path = resolve_model_path(profile)
        print(f"[INFO] Loading local model: {model_path}")
        backend = create_backend(profile, model_path)
    except (ProfileError, RuntimeError, OSError) as exc:
        raise SystemExit(f"Unable to initialize multimodal inference: {exc}") from exc

    try:
        return run_backend(
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
        raise SystemExit(f"Multimodal inference failed: {exc}") from exc
    finally:
        backend.close()


if __name__ == "__main__":
    main()
