#!/usr/bin/env python3
"""Download and validate one Omnilingual Fairseq2 asset cache."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare an Omnilingual model cache")
    parser.add_argument("--inference_profile", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    import torch

    from inference.omnilingual.cache import configure_model_cache, write_prepared_marker
    from inference.profile import ProfileError, load_profile

    args = parse_args(argv)
    try:
        profile = load_profile(
            args.inference_profile, expected_inference_library="omnilingual"
        )
        cache_path = configure_model_cache(profile, create=True)
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Preparing the 3B model requires a CUDA BF16 GPU")

        # Import only after FAIRSEQ2_CACHE_DIR points at the model-specific cache.
        from inference.omnilingual.backend import load_owner_pipeline

        pipeline = load_owner_pipeline(profile, torch.device("cuda:0"))
        del pipeline
        torch.cuda.empty_cache()
        marker = write_prepared_marker(profile, cache_path)
    except (ProfileError, RuntimeError, OSError) as exc:
        raise SystemExit(f"Unable to prepare Omnilingual model: {exc}") from exc

    print(f"[INFO] Prepared Omnilingual cache: {cache_path}")
    print(f"[INFO] Preparation marker: {marker}")
    return marker


if __name__ == "__main__":
    main()
