#!/usr/bin/env python3

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from nemo.collections.asr.models import ASRModel
from tqdm import tqdm

from egra_eval.data.dataset_layout import DatasetLayoutError, resolve_dataset_paths
from egra_eval.pipeline.segmenter import segment_wav_from_textgrid

TARGET_SR = 16000
DEFAULT_TMP = "nemo_inference/tmp"


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def load_audio_and_resample(path: str, target_sr: int = TARGET_SR) -> Tuple[np.ndarray, int, bool]:
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    was_resampled = False
    if sr != target_sr:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr
        was_resampled = True

    return audio.astype(np.float32), sr, was_resampled


def write_wav(path: str, audio: np.ndarray, sr: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr)


def seconds_to_samples(s: float, sr: int) -> int:
    return int(round(s * sr))


# ---------------------------------------------------------------------------
# NeMo transcriber
# ---------------------------------------------------------------------------
def transcribe_batches(model, files: list[str], batch_size: int = 16, num_workers: int | None = None) -> list[str]:
   
    hyps = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(files), batch_size):
            batch = files[i : i + batch_size]
            preds = model.transcribe(batch, batch_size=batch_size, num_workers=num_workers)

            # Handle possible object outputs with `.text` attributes
            if preds and hasattr(preds[0], "text"):
                preds = [p.text for p in preds]

            hyps.extend(preds)
    return hyps

# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def discover_wavs(root_dir: str) -> List[str]:
    root = Path(root_dir)
    return [str(p) for p in sorted(root.rglob("*.wav"))]


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline NeMo transcription helper")
    parser.add_argument("--model", required=True, help="Path to .nemo model")
    parser.add_argument("--dataset_root", required=True, help="Dataset root containing 0_Audio and 2_TextGrid directories.")
    parser.add_argument("--dataset_annotator", default=None, help="Annotator folder inside 2_TextGrid to use.")
    parser.add_argument("--root_audio_dir", default=None, help="Explicit audio root (overrides dataset discovery).")
    parser.add_argument("--textgrid_dir", default=None, help="Explicit TextGrid directory (overrides dataset discovery).")
    parser.add_argument("--tier_name", default="child")
    parser.add_argument(
        "--output_root",
        required=True,
        help="Directory where transcriptions.jsonl will be written.",
    )
    parser.add_argument("--batch_size", type=int, default=os.cpu_count() or 1, help="Batch size (defaults to number of CPU cores)")
    parser.add_argument("--tmp_dir", default=DEFAULT_TMP)
    parser.add_argument("--cpu_workers", type=int, default=None, help="CPU workers when GPU is unavailable (default: all cores).")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def resolve_io_paths(args: argparse.Namespace) -> None:
    """Update CLI args in-place when dataset_root/output_root are supplied."""

    try:
        layout = resolve_dataset_paths(args.dataset_root, annotator=args.dataset_annotator)
    except DatasetLayoutError as exc:
        raise SystemExit(str(exc)) from exc

    args.root_audio_dir = args.root_audio_dir or str(layout.audio_root)
    args.textgrid_dir = args.textgrid_dir or str(layout.textgrid_root)

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    args.output_manifest = str(out_root / "transcriptions.jsonl")

    Path(args.tmp_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_manifest).parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    resolve_io_paths(args)

    audio_paths = discover_wavs(args.root_audio_dir)
    if not audio_paths:
        raise SystemExit(f"No .wav files found under: {args.root_audio_dir}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        print("[INFO] GPU detected. Using CUDA for inference.")
        num_workers = None
    else:
        workers = args.cpu_workers or os.cpu_count() or 1
        print(f"[INFO] GPU not available. Using CPU with {workers} worker(s).")
        try:
            torch.set_num_threads(workers)
        except Exception:
            pass
        try:
            torch.set_num_interop_threads(min(4, workers))
        except Exception:
            pass
        os.environ.setdefault("OMP_NUM_THREADS", str(workers))
        os.environ.setdefault("MKL_NUM_THREADS", str(workers))
        num_workers = workers

    model = ASRModel.restore_from(args.model, map_location=device)
    model.to(device).eval()

    with open(args.output_manifest, "w", encoding="utf-8") as out:
        for wav_path in tqdm(audio_paths, desc="Files"):
            wav_path = str(wav_path)
            try:
                audio, sr, was_resampled = load_audio_and_resample(wav_path, TARGET_SR)
            except Exception as exc:
                out.write(json.dumps({
                    "audio_filepath": wav_path,
                    "duration": 0.0,
                    "pred_text": "",
                    "error": f"failed_to_read_audio: {exc}"
                }) + "\n")
                continue

            duration = float(len(audio) / sr)

            segment_files: List[str] = []
            cleanup_paths: List[Path] = []

            # Segmentation logic is delegated to egra_eval.pipeline.segmenter.
            # infer.py only consumes the produced segment file list.
            seg_paths = segment_wav_from_textgrid(
                wav_path=wav_path,
                textgrid_root=args.textgrid_dir,
                segments_out_root=args.tmp_dir,
            )
            if seg_paths:
                segment_files = [str(p) for p in seg_paths]
                cleanup_paths = list(seg_paths)
                if args.debug:
                    print(f"[DEBUG] Segmented {wav_path} into {len(segment_files)} file(s).")

            if not segment_files:
                if was_resampled:
                    seg_path = Path(args.tmp_dir) / f"{Path(wav_path).stem}_full16k.wav"
                    write_wav(str(seg_path), audio, sr)
                    segment_files = [str(seg_path)]
                    cleanup_paths.append(seg_path)
                else:
                    segment_files = [wav_path]

            predictions = transcribe_batches(
                model,
                segment_files,
                batch_size=args.batch_size,
                num_workers=num_workers,
            )
            combined = " ".join([p.strip() for p in predictions if p and p.strip()]).strip()

            out.write(json.dumps({
                "audio_filepath": wav_path,
                "duration": round(duration, 3),
                "pred_text": combined,
            }, ensure_ascii=False) + "\n")

            if not args.debug:
                for tmp_path in cleanup_paths:
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except Exception:
                        pass


if __name__ == "__main__":
    main()
