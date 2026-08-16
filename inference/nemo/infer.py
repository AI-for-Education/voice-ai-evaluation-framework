#!/usr/bin/env python3
"""Profile-driven and legacy NeMo ASR command-line entrypoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, List, Sequence

from egra_eval2.dataset_layout import DatasetLayoutError, resolve_dataset_paths
from egra_eval2.segmenter import segment_wav_from_textgrid
from inference.common import load_audio_and_resample, resolve_audio_paths
from inference.nemo.backend import (
    ASRModel,
    DEFAULT_TMP,
    NemoBackend,
    build_override_cfg,
    configure_decoding_strategy,
    configure_runtime,
    require_nemo_runtime,
    transcribe_batches,
    write_wav,
)
from inference.profile import ProfileError, load_profile, resolve_model_path
from inference.runner import run_backend


TARGET_SR = 16000


def parse_profile_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the new framework launcher contract."""
    parser = argparse.ArgumentParser(description="Offline profile-driven NeMo transcription")
    parser.add_argument(
        "--model_config",
        required=True,
        help="Path to a tracked NeMo model profile YAML file.",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--root_audio_dir",
        default=None,
        help="Audio root to search recursively for WAV files.",
    )
    input_group.add_argument(
        "--audio_manifest",
        default=None,
        help="Manifest containing ordered audio_filepath values.",
    )
    parser.add_argument(
        "--output_root",
        default="input_output_data/output",
        help=(
            "Output base; results are written below transcripts/<model>_<UTC timestamp> "
            "(default: input_output_data/output)."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="NeMo transcribe batch size (default: 32).",
    )
    parser.add_argument(
        "--tmp_dir",
        default=str(DEFAULT_TMP),
        help="Directory for temporary resampled audio.",
    )
    parser.add_argument(
        "--cpu_workers",
        type=int,
        default=0,
        help="NeMo workers and CPU thread hint when CUDA is unavailable.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Retain temporary prepared audio for inspection.",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Write below smoke_tests/transcripts/<model>_<UTC timestamp>.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    """Run NeMo through a validated, portable model profile."""
    args = parse_profile_args(argv)
    if args.batch_size < 1:
        raise SystemExit("--batch_size must be at least 1")

    try:
        print(f"[INFO] Loading profile: {args.model_config}")
        profile = load_profile(args.model_config, expected_framework="nemo")
        model_path = resolve_model_path(profile)
        print(f"[INFO] Loading local model: {model_path}")
        backend = NemoBackend(
            profile=profile,
            model_path=model_path,
            batch_size=args.batch_size,
            cpu_workers=args.cpu_workers,
            tmp_dir=args.tmp_dir,
            debug=args.debug,
        )
    except (ProfileError, RuntimeError, OSError) as exc:
        raise SystemExit(f"Unable to initialize NeMo inference: {exc}") from exc
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


# ---------------------------------------------------------------------------
# Legacy CLI retained exclusively for root infer.py.
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the complete legacy NeMo command contract."""
    parser = argparse.ArgumentParser(description="Offline NeMo transcription helper")
    parser.add_argument("--model", required=True, help="Path to .nemo model")
    parser.add_argument(
        "--dataset_root",
        default=None,
        help="Legacy dataset root used to discover 0_Audio and 2_TextGrid directories.",
    )
    parser.add_argument(
        "--dataset_annotator",
        default=None,
        help="Annotator folder inside 2_TextGrid to use.",
    )
    parser.add_argument(
        "--root_audio_dir",
        default=None,
        help="Explicit audio root (overrides dataset discovery).",
    )
    parser.add_argument(
        "--audio_manifest",
        "--manifest_in",
        dest="audio_manifest",
        default=None,
        help=(
            "Segment manifest containing the ordered audio_filepath values to "
            "transcribe; --manifest_in remains a backward-compatible alias."
        ),
    )
    parser.add_argument(
        "--textgrid_dir",
        default=None,
        help="Explicit TextGrid directory (overrides dataset discovery).",
    )
    parser.add_argument("--tier_name", default="child")
    parser.add_argument(
        "--output_root",
        required=True,
        help="Directory where transcriptions.jsonl will be written.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for NeMo transcribe() override_cfg (default: 32).",
    )
    parser.add_argument("--tmp_dir", default=str(DEFAULT_TMP))
    parser.add_argument(
        "--cpu_workers",
        type=int,
        default=0,
        help="num_workers for transcribe override_cfg when GPU is unavailable.",
    )
    parser.add_argument(
        "--decoder_type",
        default="ctc",
        choices=["ctc", "rnnt"],
        help="Decoding strategy to use (default: ctc).",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def resolve_io_paths(args: argparse.Namespace) -> None:
    """Resolve the legacy dataset layout and output paths in place."""
    if args.dataset_root:
        try:
            layout = resolve_dataset_paths(args.dataset_root, annotator=args.dataset_annotator)
        except DatasetLayoutError as exc:
            raise SystemExit(str(exc)) from exc

        args.root_audio_dir = args.root_audio_dir or str(layout.audio_root)
        args.textgrid_dir = args.textgrid_dir or str(layout.textgrid_root)
    elif not args.root_audio_dir and not args.audio_manifest:
        raise SystemExit(
            "Provide --root_audio_dir or --audio_manifest. "
            "Legacy dataset discovery is available through --dataset_root."
        )

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    args.output_manifest = str(out_root / "transcriptions.jsonl")

    Path(args.tmp_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_manifest).parent.mkdir(parents=True, exist_ok=True)


def legacy_main(
    argv: Sequence[str] | None = None,
    model_path_mapper: Callable[[str], str | Path] | None = None,
) -> None:
    """Run the original NeMo workflow, including TextGrid segmentation."""
    args = parse_args(argv)
    from tqdm import tqdm

    if model_path_mapper is not None:
        args.model = str(model_path_mapper(args.model))
    resolve_io_paths(args)

    audio_paths = resolve_audio_paths(
        root_audio_dir=args.root_audio_dir,
        audio_manifest=args.audio_manifest,
    )

    require_nemo_runtime()
    assert ASRModel is not None
    device, num_workers = configure_runtime(args.cpu_workers)

    model = ASRModel.restore_from(args.model, map_location=device)
    model.to(device).eval()

    configure_decoding_strategy(model, decoder_type=args.decoder_type)
    override_cfg = build_override_cfg(
        model=model,
        batch_size=args.batch_size,
        num_workers=num_workers,
    )

    print("[INFO] Using override_cfg:")
    print(f"       batch_size={override_cfg.batch_size}")
    print(f"       num_workers={override_cfg.num_workers}")
    print(f"       return_hypotheses={override_cfg.return_hypotheses}")
    print(f"       channel_selector={override_cfg.channel_selector}")
    print(f"       augmentor={override_cfg.augmentor}")
    print(f"       text_field={override_cfg.text_field}")
    print(f"       lang_field={override_cfg.lang_field}")
    print(f"       timestamps={override_cfg.timestamps}")

    with open(args.output_manifest, "w", encoding="utf-8") as output:
        for wav_path in tqdm(audio_paths, desc="Files"):
            wav_path = str(wav_path)
            try:
                audio, sample_rate, duration, was_resampled = load_audio_and_resample(
                    wav_path,
                    TARGET_SR,
                )
            except Exception as exc:
                output.write(
                    json.dumps(
                        {
                            "audio_filepath": wav_path,
                            "duration": 0.0,
                            "pred_text": "",
                            "error": f"failed_to_read_audio: {exc}",
                        }
                    )
                    + "\n"
                )
                continue

            segment_files: List[str] = []
            cleanup_paths: List[Path] = []

            segment_paths = []
            if args.textgrid_dir:
                segment_paths = segment_wav_from_textgrid(
                    wav_path=wav_path,
                    textgrid_root=args.textgrid_dir,
                    segments_out_root=args.tmp_dir,
                )
            if segment_paths:
                segment_files = [str(path) for path in segment_paths]
                cleanup_paths = list(segment_paths)
                if args.debug:
                    print(
                        f"[DEBUG] Segmented {wav_path} into "
                        f"{len(segment_files)} file(s)."
                    )

            if not segment_files:
                if was_resampled:
                    segment_path = Path(args.tmp_dir) / f"{Path(wav_path).stem}_full16k.wav"
                    write_wav(str(segment_path), audio, sample_rate)
                    segment_files = [str(segment_path)]
                    cleanup_paths.append(segment_path)
                else:
                    segment_files = [wav_path]

            predictions = transcribe_batches(
                model=model,
                files=segment_files,
                override_cfg=override_cfg,
            )
            combined = " ".join(
                prediction.strip()
                for prediction in predictions
                if prediction and prediction.strip()
            ).strip()

            output.write(
                json.dumps(
                    {
                        "audio_filepath": wav_path,
                        "duration": round(duration, 3),
                        "pred_text": combined,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            if not args.debug:
                for tmp_path in cleanup_paths:
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except Exception:
                        pass


if __name__ == "__main__":
    main()
