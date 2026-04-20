#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import soundfile as sf


# ----------------------------
# TextGrid parsing (single tier IntervalTier)
# ----------------------------

@dataclass
class TGInterval:
    xmin: float
    xmax: float
    text: str  # raw text from TextGrid


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        return s[1:-1]
    return s


def parse_textgrid_intervals(textgrid_path: str) -> List[TGInterval]:
    """
    Parses a Praat TextGrid (text format) with a single IntervalTier.
    Extracts all 'intervals [k]' blocks with xmin/xmax/text.

    Does NOT filter/drop any interval based on text.
    """
    intervals: List[TGInterval] = []

    cur_xmin: Optional[float] = None
    cur_xmax: Optional[float] = None
    cur_text: Optional[str] = None
    in_interval_block = False

    with open(textgrid_path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()

            if line.startswith("intervals [") and line.endswith("]:"):
                if in_interval_block and cur_xmin is not None and cur_xmax is not None and cur_text is not None:
                    intervals.append(TGInterval(cur_xmin, cur_xmax, cur_text))
                in_interval_block = True
                cur_xmin = None
                cur_xmax = None
                cur_text = None
                continue

            if not in_interval_block:
                continue

            if line.startswith("xmin ="):
                cur_xmin = float(line.split("=", 1)[1].strip())
            elif line.startswith("xmax ="):
                cur_xmax = float(line.split("=", 1)[1].strip())
            elif line.startswith("text ="):
                val = line.split("=", 1)[1].strip()
                cur_text = _strip_quotes(val)

    if in_interval_block and cur_xmin is not None and cur_xmax is not None and cur_text is not None:
        intervals.append(TGInterval(cur_xmin, cur_xmax, cur_text))

    return intervals


# ----------------------------
# TextGrid lookup
# ----------------------------

def find_textgrid_for_wav(textgrid_root: str, wav_path: str) -> Optional[str]:
    wav_base = Path(wav_path).stem
    parts = wav_base.split("_")
    speaker_id = "_".join(parts[:3]) if len(parts) >= 3 else parts[0]

    root = Path(textgrid_root)
    candidate_dir = root / speaker_id

    exact = candidate_dir / f"{wav_base}.TextGrid"
    if exact.is_file():
        return str(exact)

    if candidate_dir.is_dir():
        hits = sorted(candidate_dir.glob(f"{wav_base}*.TextGrid"))
        if hits:
            return str(hits[0])

    hits = sorted(root.rglob(f"{wav_base}.TextGrid"))
    if hits:
        return str(hits[0])

    return None


# ----------------------------
# Output path helpers
# ----------------------------

def ensure_dir(path: str, dry_run: bool) -> None:
    if dry_run:
        return
    os.makedirs(path, exist_ok=True)


def speaker_id_from_wav(wav_path: str) -> str:
    wav_base = Path(wav_path).stem
    parts = wav_base.split("_")
    return "_".join(parts[:3]) if len(parts) >= 3 else parts[0]


def build_segment_path(segments_out_root: str, wav_path: str, segment_idx: int) -> Tuple[str, str]:
    spk = speaker_id_from_wav(wav_path)
    wav_base = Path(wav_path).stem
    out_dir = os.path.join(segments_out_root, spk)
    seg_name = f"{wav_base}_segment{segment_idx}.wav"
    return out_dir, os.path.join(out_dir, seg_name)


def build_original_name_path(segments_out_root: str, wav_path: str) -> Tuple[str, str]:
    spk = speaker_id_from_wav(wav_path)
    out_dir = os.path.join(segments_out_root, spk)
    return out_dir, os.path.join(out_dir, Path(wav_path).name)


# ----------------------------
# Audio segmentation
# ----------------------------

def cut_audio_segments(
    wav_path: str,
    intervals: List[Tuple[float, float]],  # (xmin, xmax)
    segments_out_root: str,
    dry_run: bool,
    keep_original_name_when_single: bool = False,
) -> List[Tuple[int, str, float]]:
    """
    Cuts audio for EVERY interval given (no filtering by label text).

    Returns list of:
      (interval_index_1based, segment_path, duration)

    Note: Degenerate intervals where xmax<=xmin or sample end<=start are skipped
    (no valid audio can be written). Their interval indices will not appear in results.
    """
    audio = None
    sr = None
    if not dry_run:
        audio, sr = sf.read(wav_path)

    results: List[Tuple[int, str, float]] = []
    seg_written_idx = 0
    positive_intervals = [(idx, b) for idx, b in enumerate(intervals, 1) if float(b[1]) > float(b[0])]
    single_positive_interval = keep_original_name_when_single and len(positive_intervals) == 1

    for interval_idx, (xmin, xmax) in enumerate(intervals, 1):
        if float(xmax) <= float(xmin):
            continue

        seg_dur = max(0.0, float(xmax) - float(xmin))

        if not dry_run:
            assert sr is not None and audio is not None
            s = int(round(float(xmin) * sr))
            e = int(round(float(xmax) * sr))
            s = max(0, min(s, len(audio)))
            e = max(0, min(e, len(audio)))
            if e <= s:
                continue

        seg_written_idx += 1
        if single_positive_interval:
            out_dir, seg_path = build_original_name_path(segments_out_root, wav_path)
        else:
            out_dir, seg_path = build_segment_path(segments_out_root, wav_path, seg_written_idx)
        ensure_dir(out_dir, dry_run)

        if not dry_run:
            sf.write(seg_path, audio[s:e], sr)

        results.append((interval_idx, seg_path, seg_dur))

    return results


# ----------------------------
# Manifest processing (SEGMENT EVERYTHING, TEXT PER INTERVAL)
# ----------------------------

def process_manifest(
    jsonl_path: str,
    textgrid_root: str,
    segments_out_root: str,
    manifest_out_path: str,
    dry_run: bool,
) -> None:
    """
    Reads an input manifest and writes an output manifest to manifest_out_path.

    Behavior (strict segment-only):
      - If missing wav / missing TG / empty TG / invalid interval(s): skip row.
      - For any valid TextGrid with valid intervals: write one line per generated segment.
      - Single-interval files keep the original filename (no *_segment1 suffix).
    """
    in_path = Path(jsonl_path)
    out_path = Path(manifest_out_path)
    ensure_dir(str(out_path.parent), dry_run)

    new_items: List[dict] = []
    skipped_missing_wav = 0
    skipped_missing_tg = 0
    skipped_empty_tg = 0
    skipped_no_audio_segment = 0
    dropped_duplicate_audio = 0

    with open(in_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)

            wav_path = item.get("audio_filepath", "")
            if not wav_path or not Path(wav_path).is_file():
                skipped_missing_wav += 1
                continue

            tg_path = find_textgrid_for_wav(textgrid_root, wav_path)
            if not tg_path or not Path(tg_path).is_file():
                skipped_missing_tg += 1
                continue

            raw_intervals = parse_textgrid_intervals(tg_path)
            if len(raw_intervals) == 0:
                skipped_empty_tg += 1
                continue

            bounds: List[Tuple[float, float]] = [(itv.xmin, itv.xmax) for itv in raw_intervals]
            texts: List[str] = [itv.text for itv in raw_intervals]
            segs = cut_audio_segments(
                wav_path=wav_path,
                intervals=bounds,
                segments_out_root=segments_out_root,
                dry_run=dry_run,
                keep_original_name_when_single=True,
            )

            if not segs:
                skipped_no_audio_segment += 1
                continue

            for interval_idx, seg_path, seg_dur in segs:
                seg_item = dict(item)
                seg_item["audio_filepath"] = seg_path
                seg_item["duration"] = seg_dur

                # text from the corresponding interval
                # interval_idx is 1-based
                if 1 <= interval_idx <= len(texts):
                    interval_text = texts[interval_idx - 1]
                    # Keep reference text aligned to this specific segment only.
                    seg_item["ref_text"] = interval_text
                    seg_item["text"] = interval_text
                else:
                    seg_item["ref_text"] = ""
                    seg_item["text"] = ""

                new_items.append(seg_item)

    deduped_items: List[dict] = []
    seen_audio_paths: set[str] = set()
    for it in new_items:
        audio_path = str(it.get("audio_filepath", "")).strip()
        if not audio_path:
            continue
        if audio_path in seen_audio_paths:
            dropped_duplicate_audio += 1
            continue
        seen_audio_paths.add(audio_path)
        deduped_items.append(it)

    if not dry_run:
        with open(out_path, "w", encoding="utf-8") as out_f:
            for it in deduped_items:
                out_f.write(json.dumps(it, ensure_ascii=False) + "\n")

    total_skipped = skipped_missing_wav + skipped_missing_tg + skipped_empty_tg + skipped_no_audio_segment
    print(
        "SEGMENT SUMMARY:",
        f"written={len(deduped_items)}",
        f"skipped_missing_wav={skipped_missing_wav}",
        f"skipped_missing_tg={skipped_missing_tg}",
        f"skipped_empty_tg={skipped_empty_tg}",
        f"skipped_no_audio_segment={skipped_no_audio_segment}",
        f"dropped_duplicate_audio={dropped_duplicate_audio}",
        f"total_skipped={total_skipped}",
    )


# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Segment WAV files using ALL TextGrid intervals and write a *_segments.jsonl manifest. "
            "The text for each segment is set to the corresponding TextGrid interval label."
        )
    )
    ap.add_argument("--textgrid-root", required=True, help="Path to the 2_TextGrid directory")
    ap.add_argument("--manifest-in", required=True, help="Single input manifest path (JSONL or concatenated JSON).")
    ap.add_argument("--manifest-out", required=True, help="Single output manifest path.")
    ap.add_argument(
        "--segments-out-root",
        required=True,
        help="Directory where segmented WAV files are written (organized by speaker_id).",
    )

    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write manifests/segments; only print what would be done.",
    )
    args = ap.parse_args()

    in_path = Path(args.manifest_in)
    out_path = Path(args.manifest_out)
    if args.dry_run:
        print(f"[DRY] {in_path} -> {out_path}")
    else:
        print(f"PROC: {in_path.name} -> {out_path.name}")
    process_manifest(
        jsonl_path=str(in_path),
        textgrid_root=args.textgrid_root,
        segments_out_root=args.segments_out_root,
        manifest_out_path=str(out_path),
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        print(f"OK:   {in_path.name} -> {out_path.name}")

    print("\n[DRY-RUN DONE]" if args.dry_run else "\n[DONE]")


if __name__ == "__main__":
    main()
