#!/usr/bin/env python3
import argparse
import json
import os
import re
import inspect
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
from praatio import textgrid

import torch
from nemo.collections.asr.models import ASRModel

# =====================================
# Constants
# =====================================
TARGET_SR = 16000  # model's training sample rate
DEFAULT_ROOT = "input_output_data/input/audio_and_texgrid"
DEFAULT_OUT = "input_output_data/input/nemo_asr_output/transcriptions.jsonl"
DEFAULT_TMP = "nemo_inference/tmp"


# =====================================
# Audio helpers
# =====================================
def load_audio_and_resample(path: str, target_sr: int = TARGET_SR) -> Tuple[np.ndarray, int, bool]:
    """
    Load audio (mono) and resample if needed.
    """
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  # mono mixdown

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


# =====================================
# TextGrid helpers
# =====================================
def find_passage_textgrid(
    wav_path: str,
    textgrid_dir: Optional[str],
    keyword: str = "passage",
    debug: bool = False,
) -> Optional[Path]:
    """
      Try a .TextGrid with the SAME STEM as the wav. Then search the folder for any *.TextGrid that contains `keyword` (case-insensitive).
    """
    wav = Path(wav_path)
    search_dir = Path(textgrid_dir) if textgrid_dir else wav.parent

    # exact stem match
    exact = search_dir / f"{wav.stem}.TextGrid"
    if exact.exists():
        if debug:
            print(f"[DEBUG] Using TextGrid (exact stem): {exact}")
        return exact

    # keyword match
    pattern = re.compile(rf".*{re.escape(keyword)}.*\.TextGrid$", re.IGNORECASE)
    for candidate in search_dir.glob("*.TextGrid"):
        if pattern.match(candidate.name):
            if debug:
                print(f"[DEBUG] Using TextGrid (keyword match): {candidate}")
            return candidate

    if debug:
        print(f"[DEBUG] No TextGrid found for {wav_path} in {search_dir} (keyword='{keyword}')")
    return None


def _get_tier_case_insensitive(tg: textgrid.Textgrid, tier_name: str):
    for nm in tg.tierNames:
        if nm.lower() == tier_name.lower():
            return tg.getTier(nm)
    return None


def _entries_from_tier(tier) -> List[Tuple[float, float, str]]:
    """
    Normalize Praatio 5.x / 6.x tier entries into a list of (start, end, label).
    """
    entries = getattr(tier, "entries", None)
    if entries is None:
        entries = getattr(tier, "entryList", [])

    out = []
    for e in entries:
        if isinstance(e, (tuple, list)) and len(e) >= 3:
            start, end, lab = e[0], e[1], e[2]
        else:
            start = getattr(e, "start", None)
            end = getattr(e, "end", None)
            lab = getattr(e, "label", "")
        if start is None or end is None:
            continue
        out.append((float(start), float(end), (lab or "").strip()))
    return out


def read_passage_intervals(tg_path: Path, tier_name: str, debug: bool = False) -> List[Tuple[float, float, str]]:
    tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=False, reportingMode="silence")
    tier = _get_tier_case_insensitive(tg, tier_name)
    if tier is None:
        if debug:
            print(f"[DEBUG] Tier '{tier_name}' not found in {tg_path}. Available: {tg.tierNames}")
        return []
    raw = _entries_from_tier(tier)
    labeled = [(s, e, lab) for (s, e, lab) in raw if lab]
    if debug:
        print(f"[DEBUG] {tg_path.name}: {len(raw)} intervals on tier, {len(labeled)} labeled.")
    return labeled


def slice_by_intervals(audio: np.ndarray, sr: int, intervals: List[Tuple[float, float, str]]) -> List[np.ndarray]:
    chunks = []
    for start, end, _ in intervals:
        s = max(0, seconds_to_samples(start, sr))
        e = min(len(audio), seconds_to_samples(end, sr))
        if e > s:
            chunks.append(audio[s:e].copy())
    return chunks


# =====================================
# NeMo transcribe compatibility
# =====================================
def transcribe_compat(model, paths: List[str], batch_size=16):
    """
    Choose the right kwarg for model.transcribe() across NeMo 1.x / 2.x.
    """
    fn = getattr(model, "transcribe", None)
    if fn is None:
        raise RuntimeError("Model has no .transcribe() method")

    sig = inspect.signature(fn)
    params = sig.parameters

    if "paths" in params:
        return fn(paths=paths, batch_size=batch_size)
    if "paths2audio_files" in params:
        return fn(paths2audio_files=paths, batch_size=batch_size)

    try:
        return fn(paths)
    except TypeError:
        return fn(paths2audio_files=paths)


def transcribe_batches(model, files: List[str], batch_size=16) -> List[str]:
    hyps = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            preds = transcribe_compat(model, batch, batch_size=batch_size)
            if preds and hasattr(preds[0], "text"):
                preds = [p.text for p in preds]
            hyps.extend(preds)
    return hyps


# =====================================
# Discovery wavs
# =====================================
def discover_wavs(root_dir: str) -> List[str]:
    root = Path(root_dir)
    files = [str(p) for p in sorted(root.rglob("*.wav"))]
    return files


# =====================================
# Main
# =====================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to .nemo model")

    ap.add_argument("--root_audio_dir", default=DEFAULT_ROOT,
                    help="Root folder to scan for WAV files recursively.")

    ap.add_argument("--textgrid_dir", default=None,
                    help="Optional override: where to look for TextGrids. "
                         "If not set, search next to each WAV.")

    ap.add_argument("--textgrid_keyword", default="passage",
                    help="Fallback keyword to match TextGrids if exact stem is not found.")

    ap.add_argument("--tier_name", default="child", help="Tier name in TextGrid (case-insensitive)")

    ap.add_argument("--output_manifest", default=DEFAULT_OUT,
                    help="NeMo-style output JSONL path.")

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--tmp_dir", default=DEFAULT_TMP,
                    help="Where to write temporary 16k segments.")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    # Discover audio files
    audio_paths = discover_wavs(args.root_audio_dir)
    if not audio_paths:
        raise SystemExit(f"No .wav files found under: {args.root_audio_dir}")

    # Ensure dirs exist
    Path(args.tmp_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_manifest).parent.mkdir(parents=True, exist_ok=True)

    # Device
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using {'GPU' if device_str.startswith('cuda') else 'CPU'} for inference.")

    # Load model
    model = ASRModel.restore_from(args.model, map_location=device_str)
    model.to(device_str).eval()

    # Output
    with open(args.output_manifest, "w", encoding="utf-8") as out:
        for wav_path in tqdm(audio_paths, desc="Files"):
            wav_path = str(wav_path)

            # Load + resample if needed
            try:
                audio, sr, was_resampled = load_audio_and_resample(wav_path, TARGET_SR)
            except Exception as e:
                out.write(json.dumps({
                    "audio_filepath": wav_path,
                    "duration": 0.0,
                    "pred_text": "",
                    "error": f"failed_to_read_audio: {e}"
                }) + "\n")
                continue

            duration = float(len(audio) / sr)

            # Find TextGrid
            tg_path = find_passage_textgrid(
                wav_path,
                args.textgrid_dir,
                keyword=args.textgrid_keyword,
                debug=args.debug,
            )

            # Build list of segment files
            segment_files: List[str] = []
            cleanup_paths: List[Path] = []

            if tg_path is not None:
                intervals = read_passage_intervals(tg_path, args.tier_name, debug=args.debug)
                if intervals:
                    for idx, chunk in enumerate(slice_by_intervals(audio, sr, intervals)):
                        seg_path = Path(args.tmp_dir) / f"{Path(wav_path).stem}_seg{idx:03d}.wav"
                        write_wav(str(seg_path), chunk, sr)
                        if args.debug:
                            s, e, lab = intervals[idx]
                            print(f"[DEBUG] Wrote segment {idx}: {seg_path} [{s:.3f},{e:.3f}] '{lab}'")
                        segment_files.append(str(seg_path))
                        cleanup_paths.append(seg_path)

            # If no intervals or empty tier, use full file
            if not segment_files:
                if was_resampled:
                    seg_path = Path(args.tmp_dir) / f"{Path(wav_path).stem}_full16k.wav"
                    write_wav(str(seg_path), audio, sr)
                    segment_files = [str(seg_path)]
                    cleanup_paths.append(seg_path)
                else:
                    segment_files = [wav_path]

            # Transcribe and combine
            hyps = transcribe_batches(model, segment_files, batch_size=args.batch_size)
            combined_hyp = " ".join([h.strip() for h in hyps if h and h.strip()]).strip()

            out.write(json.dumps({
                "audio_filepath": wav_path,
                "duration": round(duration, 3),
                "pred_text": combined_hyp
            }, ensure_ascii=False) + "\n")

            # Cleanup temp files unless debugging
            if not args.debug:
                for p in cleanup_paths:
                    try:
                        p.unlink(missing_ok=True)
                    except Exception:
                        pass


if __name__ == "__main__":
    main()

