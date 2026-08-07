#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

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

from egra_eval2.dataset_layout import DatasetLayoutError, resolve_dataset_paths
from egra_eval2.segmenter import segment_wav_from_textgrid
from inference_common import load_audio_and_resample, resolve_audio_paths

TARGET_SR = 16000
DEFAULT_TMP = "nemo_inference/tmp"


def write_wav(path: str, audio: np.ndarray, sr: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr)


# ---------------------------------------------------------------------------
# NeMo helpers
# ---------------------------------------------------------------------------

def configure_decoding_strategy(model, decoder_type: str = "ctc") -> None:
    if isinstance(model, EncDecCTCModel):
        if decoder_type == "rnnt":
            raise SystemExit("Loaded model is EncDecCTCModel (CTC-only). It cannot decode with RNNT.")
        print("[INFO] Loaded model is EncDecCTCModel. CTC decoding is already active.")
        return

    if isinstance(model, EncDecHybridRNNTCTCModel):
        ctc_cfg = CTCDecodingConfig()
        model.change_decoding_strategy(ctc_cfg, decoder_type=decoder_type)
        cur_decoder = getattr(model, "cur_decoder", None)
        print(f"[INFO] Loaded hybrid RNNT/CTC model. Set decoder_type='{decoder_type}'. Current decoder: {cur_decoder}")
        return

    if isinstance(model, EncDecRNNTModel):
        if decoder_type == "ctc":
            raise SystemExit(
                "Loaded model is EncDecRNNTModel (RNNT-only). It cannot be forced to decode with CTC."
            )
        print("[INFO] Loaded model is EncDecRNNTModel. RNNT decoding is already active.")
        return

    print(f"[WARN] Unknown ASR model class: {type(model)}. Decoding strategy was not changed.")


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
# CLI / main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline NeMo transcription helper")
    parser.add_argument("--model", required=True, help="Path to .nemo model")
    parser.add_argument(
        "--dataset_root",
        default=None,
        help="Legacy dataset root used to discover 0_Audio and 2_TextGrid directories.",
    )
    parser.add_argument("--dataset_annotator", default=None, help="Annotator folder inside 2_TextGrid to use.")
    parser.add_argument("--root_audio_dir", default=None, help="Explicit audio root (overrides dataset discovery).")
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
    parser.add_argument(
        "--decoder_type",
        default="ctc",
        choices=["ctc", "rnnt"],
        help="Decoding strategy to use (default: ctc).",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def resolve_io_paths(args: argparse.Namespace) -> None:
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


def main() -> None:
    args = parse_args()
    resolve_io_paths(args)

    audio_paths = resolve_audio_paths(
        root_audio_dir=args.root_audio_dir,
        audio_manifest=args.audio_manifest,
    )

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

    with open(args.output_manifest, "w", encoding="utf-8") as out:
        for wav_path in tqdm(audio_paths, desc="Files"):
            wav_path = str(wav_path)
            try:
                audio, sr, duration, was_resampled = load_audio_and_resample(wav_path, TARGET_SR)
            except Exception as exc:
                out.write(json.dumps({
                    "audio_filepath": wav_path,
                    "duration": 0.0,
                    "pred_text": "",
                    "error": f"failed_to_read_audio: {exc}"
                }) + "\n")
                continue

            segment_files: List[str] = []
            cleanup_paths: List[Path] = []

            seg_paths = []
            if args.textgrid_dir:
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
