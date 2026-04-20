from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict


NON_ASCII_RE = re.compile(r"[^\x20-\x7E]")
LT_RE = re.compile(r"\s*<\s*")
GT_RE = re.compile(r"\s*>\s*")
TAG_RE = re.compile(r"<[^>]*>")
# Keep apostrophes (both ASCII ' and Unicode ’) because they are phonologically
# meaningful in this dataset (e.g., n'go, ng'u).
PUNCT_AND_DIGITS_RE = re.compile(r"[,\.\d\?\!\`\"\-]")
SPACES_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Clean manifest text fields.

    Training-style aggressive cleaning:
      - remove bracket-like tags/tokens
      - remove punctuation and digits
      - lowercase + collapse whitespace
    """
    if not isinstance(text, str) or not text:
        return ""

    text = NON_ASCII_RE.sub("", text)
    text = LT_RE.sub(" <", text)
    text = GT_RE.sub("> ", text)
    text = TAG_RE.sub(" ", text)
    tokens = [tok for tok in text.split() if ("<" not in tok and ">" not in tok)]
    text = " ".join(tokens)

    text = PUNCT_AND_DIGITS_RE.sub("", text)

    return SPACES_RE.sub(" ", text).strip().lower()


def normalize_manifest_record(
    obj: Dict[str, Any],
    *,
    text_input_key: str = "ref_text",
    text_output_key: str = "ref_text",
    lowercase_can_text: bool = True,
) -> Dict[str, Any]:
    out = dict(obj)
    source = out.get(text_input_key, "")
    out[text_output_key] = clean_text(source)
    if text_output_key != "text":
        out.pop("text", None)
    if lowercase_can_text and isinstance(out.get("can_text"), str):
        out["can_text"] = out["can_text"].lower()
    return out


def clean_manifest_jsonl(
    *,
    input_path: str | Path,
    output_path: str | Path,
    drop_empty_text: bool = False,
    text_input_key: str = "ref_text",
    text_output_key: str = "ref_text",
) -> dict[str, Any]:
    in_path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_lines": 0,
        "bad_json_skipped": 0,
        "written_lines": 0,
        "dropped_empty_text": 0,
        "output_path": str(out_path),
    }

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            s = line.strip()
            if not s:
                continue
            stats["total_lines"] += 1
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                stats["bad_json_skipped"] += 1
                continue

            if not isinstance(obj, dict):
                stats["bad_json_skipped"] += 1
                continue

            normalized = normalize_manifest_record(
                obj,
                text_input_key=text_input_key,
                text_output_key=text_output_key,
            )

            if drop_empty_text and not normalized.get(text_output_key, ""):
                stats["dropped_empty_text"] += 1
                continue

            fout.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            stats["written_lines"] += 1

    return stats
