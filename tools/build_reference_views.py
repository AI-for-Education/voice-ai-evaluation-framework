#!/usr/bin/env python3
"""Build reusable orthographic and exact-system IPA reference views."""

from __future__ import annotations

import argparse
from pathlib import Path

from egra_eval2.reference.g2p import SUPPORTED_G2P_TOOLS
from egra_eval2.reference.views import ReferenceViewError, build_reference_views


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build dataset-adjacent orthographic and IPA reference views."
    )
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument(
        "--manifest_in",
        required=True,
        help="Existing reference JSONL containing audio_filepath, can_text, and ref_text.",
    )
    parser.add_argument("--language", default="swh")
    parser.add_argument(
        "--g2p_tool",
        "--g2p-tool",
        required=True,
        choices=SUPPORTED_G2P_TOOLS,
        help="Exact G2P tool whose IPA reference view should be built.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Override <dataset_root>/_derived/reference_views.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing view after explicit review.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        paths = build_reference_views(
            dataset_root=args.dataset_root,
            manifest_in=args.manifest_in,
            language=args.language,
            g2p_tool=args.g2p_tool,
            output_dir=args.output_dir,
            force=args.force,
        )
    except ReferenceViewError as exc:
        raise SystemExit(str(exc)) from exc

    action = "Reused" if paths.reused else "Generated"
    print(f"{action} reference views: {Path(paths.output_dir)}")
    print(f"Orthographic: {paths.orthographic}")
    print(f"IPA: {paths.ipa}")
    print(f"G2P system: {paths.g2p_system.system_id}")
    print(f"Metadata: {paths.metadata}")


if __name__ == "__main__":
    main()
