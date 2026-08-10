#!/usr/bin/env python3
"""Offline Hugging Face CTC inference for pre-segmented audio."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCTC,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

from inference_common import load_audio_and_resample, resolve_audio_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline Hugging Face CTC transcription for pre-segmented WAV files"
    )
    parser.add_argument("--model", required=True, help="Local Hugging Face model directory")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--root_audio_dir",
        help="Directory containing pre-segmented WAV files (searched recursively)",
    )
    input_group.add_argument(
        "--audio_manifest",
        dest="audio_manifest",
        help="Segment manifest containing the ordered audio_filepath values to transcribe",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Directory where transcriptions.jsonl will be written",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def _load_backend(
    model_path: Path,
    device: torch.device,
) -> tuple[Any, Any, int]:
    if not model_path.is_dir():
        raise SystemExit(f"--model directory not found: {model_path}")

    try:
        # Build the plain CTC processor explicitly. Some orthographic checkpoints
        # advertise a language-model processor, which would otherwise require
        # optional pyctcdecode/KenLM packages even for greedy decoding.
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            str(model_path),
            local_files_only=True,
        )
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
        )
        processor = Wav2Vec2Processor(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
        )

        model = AutoModelForCTC.from_pretrained(
            str(model_path),
            local_files_only=True,
        )
    except Exception as exc:
        raise SystemExit(f"Failed to load local Transformers model from {model_path}: {exc}") from exc

    feature_extractor = getattr(processor, "feature_extractor", None)
    sampling_rate = int(getattr(feature_extractor, "sampling_rate", 16000))
    if sampling_rate <= 0:
        raise SystemExit(f"Model processor reported an invalid sampling rate: {sampling_rate}")

    model.to(device).eval()
    return processor, model, sampling_rate


def _transcribe_batch(
    processor: Any,
    model: Any,
    device: torch.device,
    audio_batch: list[np.ndarray],
    sampling_rate: int,
) -> list[str]:
    encoded = processor(
        audio_batch,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True,
    )
    model_inputs = {
        name: value.to(device) if hasattr(value, "to") else value
        for name, value in encoded.items()
    }

    with torch.inference_mode():
        logits = model(**model_inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    decoded_texts = processor.batch_decode(predicted_ids)

    if isinstance(decoded_texts, str):
        decoded_texts = [decoded_texts]

    return [str(text).strip() for text in decoded_texts]


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch_size must be at least 1")

    audio_paths = resolve_audio_paths(
        root_audio_dir=args.root_audio_dir,
        audio_manifest=args.audio_manifest,
    )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output_manifest = output_root / "transcriptions.jsonl"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using inference device: {device}")

    processor, model, sampling_rate = _load_backend(
        Path(args.model),
        device,
    )
    print(f"[INFO] Model processor sampling rate: {sampling_rate} Hz")
    print(f"[INFO] Discovered {len(audio_paths)} segmented WAV file(s)")

    with output_manifest.open("w", encoding="utf-8") as output:
        for start in tqdm(range(0, len(audio_paths), args.batch_size), desc="Batches"):
            batch_paths = audio_paths[start : start + args.batch_size]
            rows: list[dict[str, Any] | None] = [None] * len(batch_paths)
            valid_audio: list[np.ndarray] = []
            valid_indices: list[int] = []

            for index, audio_path in enumerate(batch_paths):
                try:
                    audio, _, duration, _ = load_audio_and_resample(audio_path, sampling_rate)
                except Exception as exc:
                    rows[index] = {
                        "audio_filepath": audio_path,
                        "duration": 0.0,
                        "pred_text": "",
                        "error": f"failed_to_read_audio: {exc}",
                    }
                    continue

                valid_audio.append(audio)
                valid_indices.append(index)
                rows[index] = {
                    "audio_filepath": audio_path,
                    "duration": round(duration, 3),
                    "pred_text": "",
                }

            if valid_audio:
                try:
                    predictions = _transcribe_batch(
                        processor=processor,
                        model=model,
                        device=device,
                        audio_batch=valid_audio,
                        sampling_rate=sampling_rate,
                    )
                except Exception as exc:
                    raise SystemExit(f"Transformers inference failed: {exc}") from exc

                if len(predictions) != len(valid_indices):
                    raise SystemExit(
                        "Transformers inference returned an unexpected number of predictions: "
                        f"expected {len(valid_indices)}, got {len(predictions)}"
                    )

                for row_index, prediction in zip(valid_indices, predictions):
                    assert rows[row_index] is not None
                    rows[row_index]["pred_text"] = prediction

            for row in rows:
                assert row is not None
                output.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[INFO] Wrote transcriptions to: {output_manifest}")


if __name__ == "__main__":
    main()
