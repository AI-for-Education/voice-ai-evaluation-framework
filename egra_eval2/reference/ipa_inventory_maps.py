"""Reviewed mappings between declared IPA inventories."""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from typing import Callable, Literal, Mapping


class IPAInventoryMapError(ValueError):
    """Raised when an IPA inventory route is absent, incomplete, or invalid."""


@dataclass(frozen=True)
class IPAInventoryAdapter:
    """One auditable source-to-target IPA conversion decision."""

    source: str
    target: str
    status: Literal["draft", "approved"]
    source_tokens: tuple[str, ...]
    replacements: Mapping[str, str]
    unresolved: Mapping[str, str]

    @property
    def key(self) -> tuple[str, str]:
        return self.source, self.target


# BookBot CTC and Sherpa Zipformer expose this same 36-unit vocabulary.
#
# Review evidence (checked 2026-08-13):
# - IPA chart: r/ɾ and implosive/pulmonic stops are generally distinct.
#   https://www.internationalphoneticassociation.org/IPAcharts/common_files/pdfs/pdfs_IPA_charts_E/IPA_unitipa_tt.pdf
# - Unicode documents ʧ as an IPA affricate ligature and distinguishes g/ɡ.
#   https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-7/
#   https://www.unicode.org/charts/nameslist/n_0250.html
# - Polomé describes Swahili /r/ as a trill or tap and voiced stops as
#   non-implosive after a tautosyllabic nasal (pp. 56-59, 225).
#   https://files.eric.ed.gov/fulltext/ED012888.pdf
# - BookBot/gruut writes Swahili ny as n+j (for example, nyuma -> nju.mɑ),
#   while its vocabulary has a separate ⁿɗ͡ʒ token for orthographic nj.
#   https://github.com/bookbot-hive/word_info_sw
#   https://huggingface.co/bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-alphabets-phonemes-bookbot/commit/3c33819141ca8f82391e32a617f10f1cef692deb
# - Africa G2P is the project reference convention: ch -> ʧ, ny -> ɲ,
#   mb -> ᵐb, nd -> ⁿd, ng -> ᵑɡ, and nj -> ⁿdʒ.
#   https://github.com/AfriSpeech/africa-g2p/blob/main/src/africa_g2p/languages/swh.json
#
# These are approved only for broad Swahili phoneme scoring between the two
# named inventories. They must not be reused as language-independent IPA rules.
BOOKBOT_GRUUT_TO_AFRICA_G2P = IPAInventoryAdapter(
    source="bookbot_gruut_sw_v1",
    target="africa_g2p_swh_ipa_v1",
    status="approved",
    source_tokens=(
        "ⁿɗ͡ʒ",
        "t͡ʃ",
        "ᵐɓ",
        "ᵑg",
        "ᶬv",
        "ⁿz",
        "ⁿɗ",
        "ð",
        "ŋ",
        "ɑ",
        "ɓ",
        "ɔ",
        "ɗ",
        "ɛ",
        "ɠ",
        "ɣ",
        "ɾ",
        "ʃ",
        "ʄ",
        "θ",
        "f",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "p",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "z",
    ),
    replacements={
        # Longest-match sequence: BookBot/gruut's n+j represents Swahili /ɲ/.
        "nj": "ɲ",
        # Notation/code-point canonicalization.
        "t͡ʃ": "ʧ",
        "ᵑg": "ᵑɡ",
        # Swahili phoneme-level equivalences, not general IPA equivalences.
        "ɾ": "r",
        "ᵐɓ": "ᵐb",
        "ⁿɗ": "ⁿd",
        "ⁿɗ͡ʒ": "ⁿdʒ",
    },
    unresolved={},
)


IPA_INVENTORY_ADAPTERS: dict[tuple[str, str], IPAInventoryAdapter] = {
    BOOKBOT_GRUUT_TO_AFRICA_G2P.key: BOOKBOT_GRUUT_TO_AFRICA_G2P,
}


def _normalize_ipa(text: object) -> str:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    return " ".join(unicodedata.normalize("NFC", str(text)).split())


def build_ipa_aligner(source: str, target: str) -> Callable[[object], str]:
    """Return a longest-match aligner only for an approved inventory route."""
    if source == target:
        return _normalize_ipa

    adapter = IPA_INVENTORY_ADAPTERS.get((source, target))
    if adapter is None:
        raise IPAInventoryMapError(
            f"No IPA inventory adapter is registered: {source} -> {target}"
        )
    if adapter.status != "approved" or adapter.unresolved:
        decisions = ", ".join(adapter.unresolved) or "status review"
        raise IPAInventoryMapError(
            f"IPA inventory adapter is not approved: {source} -> {target}; "
            f"unresolved: {decisions}"
        )
    if not adapter.source_tokens or "" in adapter.source_tokens:
        raise IPAInventoryMapError(
            f"IPA inventory adapter has an invalid source vocabulary: {source}"
        )

    # Replacement keys may span multiple source tokens (for example n+j -> ɲ).
    # Trying them together with atomic source tokens makes those decisions
    # longest-match while retaining fail-closed validation of unknown symbols.
    tokens = sorted(
        set(adapter.source_tokens) | set(adapter.replacements),
        key=len,
        reverse=True,
    )

    def align(text: object) -> str:
        normalized = _normalize_ipa(text)
        aligned: list[str] = []
        for word in normalized.split():
            offset = 0
            while offset < len(word):
                token = next(
                    (value for value in tokens if word.startswith(value, offset)),
                    None,
                )
                if token is None:
                    raise IPAInventoryMapError(
                        f"Unknown {source} IPA symbol at offset {offset}: {word!r}"
                    )
                aligned.append(adapter.replacements.get(token, token))
                offset += len(token)
        return " ".join(aligned)

    return align
