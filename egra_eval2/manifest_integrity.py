from __future__ import annotations

from collections.abc import Iterable
from typing import Any


class ReferenceIntegrityError(ValueError):
    """Raised when reference-side text contains a known decoding artifact."""


# These are the same UTF-8 replacement character after zero, one, or two common
# rounds of mojibake. Do not enforce ASCII: phoneme references legitimately use IPA.
REFERENCE_CORRUPTION_MARKERS = (
    "\ufffd",
    "\u00ef\u00bf\u00bd",
    "\u00c3\u00af\u00c2\u00bf\u00c2\u00bd",
)


def validate_reference_rows(
    rows: Iterable[dict[str, Any]],
    *,
    text_fields: tuple[str, ...],
    audio_field: str,
    source: str,
) -> None:
    """Reject known encoding corruption in REF/CAN fields, never in model HYP."""
    for row_number, row in enumerate(rows, start=1):
        audio = str(row.get(audio_field, "") or "").strip() or "<unknown>"
        for field in text_fields:
            value = row.get(field, "")
            text = "" if value is None else str(value)
            if any(marker in text for marker in REFERENCE_CORRUPTION_MARKERS):
                raise ReferenceIntegrityError(
                    f"Reference integrity check failed in {source}: field={field!r}, "
                    f"row={row_number}, audio={audio!r} contains a known encoding "
                    "corruption marker. Correct the shared reference source before "
                    "building or scoring this manifest."
                )
