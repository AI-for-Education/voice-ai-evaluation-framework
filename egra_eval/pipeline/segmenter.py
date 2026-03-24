from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import soundfile as sf


@dataclass
class TGInterval:
    xmin: float
    xmax: float
    text: str


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        return s[1:-1]
    return s


def parse_textgrid_intervals(textgrid_path: str) -> List[TGInterval]:
    """
    Parse all intervals [k] blocks from a Praat TextGrid text file.
    No filtering by label text is applied.
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
                cur_text = _strip_quotes(line.split("=", 1)[1].strip())

    if in_interval_block and cur_xmin is not None and cur_xmax is not None and cur_text is not None:
        intervals.append(TGInterval(cur_xmin, cur_xmax, cur_text))

    return intervals


def speaker_id_from_wav(wav_path: str) -> str:
    wav_base = Path(wav_path).stem
    parts = wav_base.split("_")
    return "_".join(parts[:3]) if len(parts) >= 3 else parts[0]


def find_textgrid_for_wav(textgrid_root: str, wav_path: str) -> Optional[Path]:
    wav_base = Path(wav_path).stem
    spk = speaker_id_from_wav(wav_path)

    root = Path(textgrid_root)
    candidate_dir = root / spk

    exact = candidate_dir / f"{wav_base}.TextGrid"
    if exact.is_file():
        return exact

    if candidate_dir.is_dir():
        hits = sorted(candidate_dir.glob(f"{wav_base}*.TextGrid"))
        if hits:
            return hits[0]

    hits = sorted(root.rglob(f"{wav_base}.TextGrid"))
    if hits:
        return hits[0]

    hits = sorted(root.rglob(f"{wav_base}.textgrid"))
    if hits:
        return hits[0]

    return None


def _build_segment_path(segments_out_root: str, wav_path: str, segment_idx: int) -> Path:
    spk = speaker_id_from_wav(wav_path)
    wav_base = Path(wav_path).stem
    return Path(segments_out_root) / spk / f"{wav_base}_segment{segment_idx}.wav"


def cut_audio_segments(
    wav_path: str,
    intervals: List[Tuple[float, float]],
    segments_out_root: str,
) -> List[Path]:
    """
    Cut wav into one file per valid interval (xmax > xmin and sample range non-empty).
    Returns created segment file paths.
    """
    audio, sr = sf.read(wav_path)
    out_paths: List[Path] = []
    seg_written_idx = 0

    for xmin, xmax in intervals:
        if float(xmax) <= float(xmin):
            continue

        s = int(round(float(xmin) * sr))
        e = int(round(float(xmax) * sr))
        s = max(0, min(s, len(audio)))
        e = max(0, min(e, len(audio)))
        if e <= s:
            continue

        seg_written_idx += 1
        seg_path = _build_segment_path(segments_out_root, wav_path, seg_written_idx)
        seg_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(seg_path), audio[s:e], sr)
        out_paths.append(seg_path)

    return out_paths


def segment_wav_from_textgrid(
    wav_path: str,
    textgrid_root: Optional[str],
    segments_out_root: str,
) -> List[Path]:
    """
    Segment wav by TextGrid intervals, following segment_manifests.py rules:
    - no segmentation if TextGrid missing
    - no segmentation if 0 or 1 intervals
    - segment for >=2 intervals
    """
    if not textgrid_root:
        return []
    tg_path = find_textgrid_for_wav(textgrid_root, wav_path)
    if tg_path is None or not tg_path.is_file():
        return []
    intervals = parse_textgrid_intervals(str(tg_path))
    if len(intervals) < 2:
        return []
    bounds = [(itv.xmin, itv.xmax) for itv in intervals]
    return cut_audio_segments(wav_path, bounds, segments_out_root)
