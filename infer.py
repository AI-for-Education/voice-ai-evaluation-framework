#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from nemo.collections.asr.models import (
    ASRModel,
    EncDecCTCModel,
    EncDecHybridRNNTCTCModel,
    EncDecRNNTModel,
)
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig
from tqdm import tqdm

from egra_eval.data.dataset_layout import DatasetLayoutError, resolve_dataset_paths
from egra_eval.pipeline.segmenter import segment_wav_from_textgrid

TARGET_SR = 8000
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


# ---------------------------------------------------------------------------
# NeMo helpers
# ---------------------------------------------------------------------------

def configure_model_for_ctc(model) -> None:
    """
    Force CTC decoding where supported.
    """
    if isinstance(model, EncDecCTCModel):
        print("[INFO] Loaded model is EncDecCTCModel. CTC decoding is already active.")
        return

    if isinstance(model, EncDecHybridRNNTCTCModel):
        ctc_cfg = CTCDecodingConfig()
        model.change_decoding_strategy(ctc_cfg, decoder_type="ctc")
        cur_decoder = getattr(model, "cur_decoder", None)
        print(f"[INFO] Loaded hybrid RNNT/CTC model. Forced decoder_type='ctc'. Current decoder: {cur_decoder}")
        return

    if isinstance(model, EncDecRNNTModel):
        raise SystemExit(
            "Loaded model is EncDecRNNTModel (RNNT-only). It cannot be forced to decode with CTC."
        )

    print(f"[WARN] Unknown ASR model class: {type(model)}. CTC forcing was not applied.")


def build_override_cfg(model, batch_size: int, num_workers: int | None):
    """
    Build NeMo transcribe override config.
    """
    override_cfg = model.get_transcribe_config()
    override_cfg.batch_size = batch_size
    override_cfg.num_workers = 0 if num_workers is None else num_workers
    override_cfg.return_hypotheses = False
    override_cfg.channel_selector = None
    override_cfg.augmentor = None
    override_cfg.text_field = "text"
    override_cfg.lang_field = "lang"
    override_cfg.timestamps = None
    return override_cfg


def transcribe_batches(model, files: list[str], override_cfg) -> list[str]:
    hyps: list[str] = []
    model.eval()

    batch_size = int(override_cfg.batch_size)
    with torch.no_grad():
        for i in range(0, len(files), batch_size):
            batch = files[i : i + batch_size]
            preds = model.transcribe(
                audio=batch,
                override_config=override_cfg,
            )

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


def discover_wavs_from_manifest(manifest_path: str) -> List[str]:
    """
    Load audio file paths from a manifest file.
    Supports:
      - JSONL (one object per line)
      - concatenated JSON objects
      - JSON array of objects
    """
    path = Path(manifest_path)
    if not path.is_file():
        raise SystemExit(f"--manifest_in file not found: {path}")

    text = path.read_text(encoding="utf-8")
    rows: list[dict] = []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            rows = [parsed]
        elif isinstance(parsed, list):
            rows = [x for x in parsed if isinstance(x, dict)]
    except json.JSONDecodeError:
        dec = json.JSONDecoder()
        i = 0
        n = len(text)
        while i < n:
            while i < n and text[i].isspace():
                i += 1
            if i >= n:
                break
            try:
                obj, j = dec.raw_decode(text, i)
            except json.JSONDecodeError:
                i += 1
                continue
            if isinstance(obj, dict):
                rows.append(obj)
            i = j

    audio_paths: List[str] = []
    seen = set()
    for obj in rows:
        ap = str(obj.get("audio_filepath", "") or "").strip()
        if not ap or ap in seen:
            continue
        seen.add(ap)
        audio_paths.append(ap)
    return audio_paths


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline NeMo transcription helper")
    parser.add_argument("--model", required=True, help="Path to .nemo model")
    parser.add_argument("--dataset_root", required=True, help="Dataset root containing 0_Audio and 2_TextGrid directories.")
    parser.add_argument("--dataset_annotator", default=None, help="Annotator folder inside 2_TextGrid to use.")
    parser.add_argument("--root_audio_dir", default=None, help="Explicit audio root (overrides dataset discovery).")
    parser.add_argument("--manifest_in", default=None, help="Optional manifest file to read input audio_filepath list from.")
    parser.add_argument("--textgrid_dir", default=None, help="Explicit TextGrid directory (overrides dataset discovery).")
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
        help="Batch size for NeMo transcribe() override_cfg (NeMo script default is 32).",
    )
    parser.add_argument("--tmp_dir", default=DEFAULT_TMP)
    parser.add_argument(
        "--cpu_workers",
        type=int,
        default=0,
        help="num_workers for transcribe override_cfg when GPU is unavailable (NeMo script default is 0).",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def resolve_io_paths(args: argparse.Namespace) -> None:
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

    if args.manifest_in:
        audio_paths = discover_wavs_from_manifest(args.manifest_in)
    else:
        audio_paths = discover_wavs(args.root_audio_dir)

    if not audio_paths:
        if args.manifest_in:
            raise SystemExit(f"No valid audio_filepath rows found in manifest: {args.manifest_in}")
        raise SystemExit(f"No .wav files found under: {args.root_audio_dir}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        print("[INFO] GPU detected. Using CUDA for inference.")
        num_workers = 0
    else:
        workers = args.cpu_workers
        print(f"[INFO] GPU not available. Using CPU with torch threads={workers}.")
        try:
            torch.set_num_threads(max(1, workers))
        except Exception:
            pass
        try:
            torch.set_num_interop_threads(min(4, max(1, workers)))
        except Exception:
            pass
        os.environ.setdefault("OMP_NUM_THREADS", str(max(1, workers)))
        os.environ.setdefault("MKL_NUM_THREADS", str(max(1, workers)))
        num_workers = workers

    model = ASRModel.restore_from(args.model, map_location=device)
    model.to(device).eval()

    configure_model_for_ctc(model)

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
                model=model,
                files=segment_files,
                override_cfg=override_cfg,
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
