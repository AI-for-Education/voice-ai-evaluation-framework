from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DatasetPaths:
    root: Path
    audio_root: Path
    textgrid_root: Path
    canonical_csv: Path
    metadata_csv: Path
    annotator: Optional[str] = None


class DatasetLayoutError(ValueError):
    pass


def _pick_one(paths: list[Path], description: str) -> Path:
    if not paths:
        raise DatasetLayoutError(f"Could not find {description} in dataset")
    return sorted(paths)[0]


def _find_csv(root: Path, pattern: str, description: str) -> Path:
    matches = list(root.rglob(pattern))
    return _pick_one(matches, description)


def _find_directory(root: Path, name: str) -> Path:
    matches = [p for p in root.rglob(name) if p.is_dir()]
    return _pick_one(matches, f"directory '{name}'")


def resolve_dataset_paths(dataset_root: str | Path, annotator: Optional[str] = None) -> DatasetPaths:
    root = Path(dataset_root).expanduser().resolve()
    if not root.exists():
        raise DatasetLayoutError(f"Dataset root '{root}' does not exist")

    canonical_csv = _find_csv(root, "Student_Full_Canonical_EGRA*.csv", "Student Full Canonical CSV")
    metadata_csv = _find_csv(root, "Student_MetaData_EGRA*.csv", "Student MetaData CSV")

    # locate 0_IAR hub directory if present; otherwise rely on specific subdirectories
    try:
        iar_dir = _find_directory(root, "0_IAR")
    except DatasetLayoutError:
        iar_dir = None

    audio_root = _find_directory(iar_dir if iar_dir else root, "0_Audio")
    textgrid_parent = _find_directory(iar_dir if iar_dir else root, "2_TextGrid")

    chosen_textgrid_dir = textgrid_parent
    chosen_annotator: Optional[str] = None
    if annotator:
        candidate = textgrid_parent / annotator
        if candidate.is_dir():
            chosen_textgrid_dir = candidate
            chosen_annotator = annotator
        else:
            raise DatasetLayoutError(
                f"Annotator '{annotator}' not found under '{textgrid_parent}'."
            )

    return DatasetPaths(
        root=root,
        audio_root=audio_root,
        textgrid_root=chosen_textgrid_dir,
        canonical_csv=canonical_csv,
        metadata_csv=metadata_csv,
        annotator=chosen_annotator,
    )
