"""Reviewed mappings between declared IPA inventories."""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass, field
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
    token_delimited: bool = False
    sequence_replacements: Mapping[tuple[str, ...], str] = field(
        default_factory=dict
    )
    target_tokens: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()

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
    evidence=(
        "https://files.eric.ed.gov/fulltext/ED012888.pdf",
        "https://github.com/bookbot-hive/word_info_sw",
        "https://huggingface.co/bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-alphabets-phonemes-bookbot/commit/3c33819141ca8f82391e32a617f10f1cef692deb",
        "https://github.com/AfriSpeech/africa-g2p/blob/main/src/africa_g2p/languages/swh.json",
    ),
)

# babygruut's pinned Swahili lexicon uses the same atomic BookBot/gruut phone
# convention. Keep a distinct target identity so its scores can never be mixed
# with Africa G2P, while still tokenizing native model output deterministically.
BOOKBOT_GRUUT_TO_BABYGRUUT = IPAInventoryAdapter(
    source="bookbot_gruut_sw_v1",
    target="babygruut_sw_ipa_v1",
    status="approved",
    source_tokens=BOOKBOT_GRUUT_TO_AFRICA_G2P.source_tokens,
    replacements={},
    unresolved={},
    evidence=BOOKBOT_GRUUT_TO_AFRICA_G2P.evidence,
)


# XLS-R eSpeak conversion evidence (checked 2026-09-05):
# - Meta's published dictionary is the canonical 388-token model vocabulary.
#   https://dl.fbaipublicfiles.com/fairseq/wav2vec/zero_shot/espeak_dict.txt
# - The model paper defines ``tr2tgt`` as a many-to-one mapping from each
#   training phone to the closest target phone by PanPhon feature distance.
#   https://arxiv.org/html/2109.11680#S2.SS2
#   https://github.com/dmort27/panphon
# - eSpeak's Swahili rules establish the source phone sequences used for the
#   language-specific nasal clusters and affricates.
#   https://github.com/espeak-ng/espeak-ng/blob/master/dictsource/sw_rules
# - Africa G2P and BookBot/gruut establish the two exact target conventions.
#   https://github.com/AfriSpeech/africa-g2p/blob/main/src/africa_g2p/languages/swh.json
#   https://huggingface.co/bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-alphabets-phonemes-bookbot/commit/3c33819141ca8f82391e32a617f10f1cef692deb
# - Polomé documents the relevant Swahili five-vowel system, rhotic variation,
#   and the behavior of voiced stops after tautosyllabic nasals (pp. 56-59).
#   https://files.eric.ed.gov/fulltext/ED012888.pdf
#
# The base table is a conservative tr2tgt-style map: one observed model token
# becomes one reviewed Swahili target phone. PanPhon proximity provides the
# candidates; Swahili phonology and target-specific conventions resolve cases
# where blindly choosing the numerical nearest neighbor would lose meaning.
# The source-token list is exactly the 95-token set emitted by the complete
# 7,617-item EGRA run; any other model token fails closed and must be reviewed.
FACEBOOK_ESPEAK_EGRA_TO_AFRICA = {
    **dict.fromkeys(
        (
            "a",
            "aː",
            "a5",
            "æ",
            "ai5",
            "aɪ",
            "aʊ",
            "ɐ",
            "ɑ",
            "ɑ̃",
            "ɑː",
            "ɑ5",
            "ɑu5",
        ),
        "ɑ",
    ),
    **dict.fromkeys(
        ("e", "eː", "ei5", "eɪ", "ə", "ə5", "əɜ", "ɛ", "ɚ", "ɜː"),
        "ɛ",
    ),
    **dict.fromkeys(("i", "i.5", "iː", "i5", "iɛ5", "ɪ", "ɨ"), "i"),
    **dict.fromkeys(
        ("o", "ø", "oː", "o5", "oɪ", "onɡ5", "ou5", "oʊ", "ɔ", "ɔ̃"),
        "ɔ",
    ),
    **dict.fromkeys(
        ("u", "uː", "u5", "uo5", "ʉ", "ʊ", "ʌ", "y", "y5"),
        "u",
    ),
    "b": "ɓ",
    "ç": "x",
    "ɕ": "ʃ",
    "d": "ɗ",
    "ð": "ð",
    "dʒ": "ʄ",
    "f": "f",
    "ɡ": "ɠ",
    "ɣ": "ɣ",
    "h": "h",
    "ħ": "x",
    "j": "j",
    "ɟ": "ʄ",
    "k": "k",
    "kh": "k",
    "l": "l",
    "ʎ": "l",
    "m": "m",
    "n": "n",
    "ɲ": "ɲ",
    "ŋ": "ɲ",
    "p": "p",
    "q": "k",
    **dict.fromkeys(("r", "ɹ", "ɾ", "ʁ"), "r"),
    **dict.fromkeys(("s", "s."), "s"),
    "ʃ": "ʃ",
    **dict.fromkeys(("t", "th", "ts", "ts."), "t"),
    **dict.fromkeys(("tɕ", "tɕh", "tʃ"), "ʧ"),
    "v": "v",
    "w": "w",
    "x": "x",
    "z": "z",
    "ʒ": "ʃ",
    "ʔ": "h",
    "ʕ": "ɣ",
    "β": "v",
    "θ": "θ",
}

AFRICA_G2P_SW_TARGET_TOKENS = (
    "ɑ", "ɓ", "ð", "ɗ", "ɛ", "f", "ɠ", "ɣ", "h", "i", "j", "ʄ", "k",
    "l", "m", "ᵐb", "ᶬv", "n", "ⁿd", "ⁿdʒ", "ⁿz", "ɲ", "ᵑɡ", "ɔ",
    "p", "r", "s", "ʃ", "t", "ʧ", "u", "v", "w", "x", "z", "θ",
)
BABYGRUUT_SW_TARGET_TOKENS = (
    "ɑ", "ɓ", "ð", "ɗ", "ɛ", "f", "ɠ", "ɣ", "h", "i", "j", "ʄ", "k",
    "l", "m", "ᵐɓ", "ᶬv", "n", "ⁿɗ", "ⁿɗ͡ʒ", "ⁿz", "ᵑg", "ɔ", "p",
    "ɾ", "s", "ʃ", "t", "t͡ʃ", "u", "v", "w", "x", "z", "θ",
)

FACEBOOK_ESPEAK_MAPPING_EVIDENCE = (
    "https://dl.fbaipublicfiles.com/fairseq/wav2vec/zero_shot/espeak_dict.txt",
    "https://arxiv.org/html/2109.11680#S2.SS2",
    "https://github.com/dmort27/panphon",
    "https://github.com/espeak-ng/espeak-ng/blob/master/dictsource/sw_rules",
    "https://github.com/AfriSpeech/africa-g2p/blob/main/src/africa_g2p/languages/swh.json",
    "https://huggingface.co/bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-alphabets-phonemes-bookbot/commit/3c33819141ca8f82391e32a617f10f1cef692deb",
    "https://files.eric.ed.gov/fulltext/ED012888.pdf",
)

XLS_R_ESPEAK_TO_AFRICA_G2P = IPAInventoryAdapter(
    source="facebook_espeak_cv_ft_vocab_v1",
    target="africa_g2p_swh_ipa_v1",
    status="approved",
    source_tokens=tuple(FACEBOOK_ESPEAK_EGRA_TO_AFRICA),
    replacements=FACEBOOK_ESPEAK_EGRA_TO_AFRICA,
    unresolved={},
    token_delimited=True,
    sequence_replacements={
        ("m", "b"): "ᵐb",
        ("m", "v"): "ᶬv",
        ("n", "d"): "ⁿd",
        ("n", "z"): "ⁿz",
        ("n", "dʒ"): "ⁿdʒ",
        ("n", "ɟ"): "ⁿdʒ",
        ("ɲ", "ɟ"): "ⁿdʒ",
        ("ŋ", "ɡ"): "ᵑɡ",
    },
    target_tokens=AFRICA_G2P_SW_TARGET_TOKENS,
    evidence=FACEBOOK_ESPEAK_MAPPING_EVIDENCE,
)

FACEBOOK_ESPEAK_EGRA_TO_BABYGRUUT = {
    token: {
        "r": "ɾ",
        "ʧ": "t͡ʃ",
        "ɲ": "n j",
    }.get(target, target)
    for token, target in FACEBOOK_ESPEAK_EGRA_TO_AFRICA.items()
}
# babygruut has no standalone velar-nasal unit; keep an isolated source /ŋ/ as
# the broader nasal /n/. The Swahili /ŋɡ/ sequence is handled atomically below.
FACEBOOK_ESPEAK_EGRA_TO_BABYGRUUT["ŋ"] = "n"

XLS_R_ESPEAK_TO_BABYGRUUT = IPAInventoryAdapter(
    source="facebook_espeak_cv_ft_vocab_v1",
    target="babygruut_sw_ipa_v1",
    status="approved",
    source_tokens=tuple(FACEBOOK_ESPEAK_EGRA_TO_BABYGRUUT),
    replacements=FACEBOOK_ESPEAK_EGRA_TO_BABYGRUUT,
    unresolved={},
    token_delimited=True,
    sequence_replacements={
        ("m", "b"): "ᵐɓ",
        ("m", "v"): "ᶬv",
        ("n", "d"): "ⁿɗ",
        ("n", "z"): "ⁿz",
        ("n", "dʒ"): "ⁿɗ͡ʒ",
        ("n", "ɟ"): "ⁿɗ͡ʒ",
        ("ɲ", "ɟ"): "ⁿɗ͡ʒ",
        ("ŋ", "ɡ"): "ᵑg",
    },
    target_tokens=BABYGRUUT_SW_TARGET_TOKENS,
    evidence=FACEBOOK_ESPEAK_MAPPING_EVIDENCE,
)


IPA_INVENTORY_ADAPTERS: dict[tuple[str, str], IPAInventoryAdapter] = {
    BOOKBOT_GRUUT_TO_AFRICA_G2P.key: BOOKBOT_GRUUT_TO_AFRICA_G2P,
    BOOKBOT_GRUUT_TO_BABYGRUUT.key: BOOKBOT_GRUUT_TO_BABYGRUUT,
    XLS_R_ESPEAK_TO_AFRICA_G2P.key: XLS_R_ESPEAK_TO_AFRICA_G2P,
    XLS_R_ESPEAK_TO_BABYGRUUT.key: XLS_R_ESPEAK_TO_BABYGRUUT,
}


def _approved_adapter(source: str, target: str) -> IPAInventoryAdapter:
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
    return adapter


def describe_ipa_inventory_route(source: str, target: str) -> str:
    """Describe whether native IPA is unchanged or converted for scoring."""
    if source == target:
        return f"native IPA {source} — no phoneme conversion"
    adapter = _approved_adapter(source, target)
    if not adapter.replacements:
        return (
            f"native IPA {source} — already compatible with {target} "
            "(no phoneme conversion)"
        )
    return f"native IPA {source} -> {target}"


def ipa_inventory_route_evidence(source: str, target: str) -> tuple[str, ...]:
    """Return the review sources attached to an approved conversion route."""
    if source == target:
        return ()
    return _approved_adapter(source, target).evidence


def _normalize_ipa(text: object) -> str:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    return " ".join(unicodedata.normalize("NFC", str(text)).split())


def build_ipa_aligner(source: str, target: str) -> Callable[[object], str]:
    """Return a longest-match aligner only for an approved inventory route."""
    if source == target:
        return _normalize_ipa

    adapter = _approved_adapter(source, target)
    if not adapter.source_tokens or "" in adapter.source_tokens:
        raise IPAInventoryMapError(
            f"IPA inventory adapter has an invalid source vocabulary: {source}"
        )

    if adapter.token_delimited:
        source_tokens = set(adapter.source_tokens)
        replacement_tokens = set(adapter.replacements)
        missing = sorted(source_tokens - replacement_tokens)
        extra = sorted(replacement_tokens - source_tokens)
        if missing or extra:
            details = []
            if missing:
                details.append(f"unmapped source token(s): {', '.join(missing[:5])}")
            if extra:
                details.append(f"unknown replacement key(s): {', '.join(extra[:5])}")
            raise IPAInventoryMapError(
                f"IPA inventory adapter has incomplete token coverage: {source} -> "
                f"{target}; {'; '.join(details)}"
            )
        sequence_replacements = sorted(
            adapter.sequence_replacements.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        )
        for sequence, _ in sequence_replacements:
            if not sequence or any(token not in source_tokens for token in sequence):
                raise IPAInventoryMapError(
                    "IPA inventory adapter has an invalid source-token sequence: "
                    f"{source} -> {target}; {sequence!r}"
                )
        target_tokens = set(adapter.target_tokens)
        if not target_tokens:
            raise IPAInventoryMapError(
                f"IPA inventory adapter has no target vocabulary: {source} -> {target}"
            )
        emitted = list(adapter.replacements.values()) + list(
            adapter.sequence_replacements.values()
        )
        invalid_outputs = sorted(
            {
                token
                for value in emitted
                for token in value.split()
                if token not in target_tokens
            }
        )
        if invalid_outputs:
            raise IPAInventoryMapError(
                f"IPA inventory adapter emits unknown {target} token(s): "
                f"{', '.join(invalid_outputs)}"
            )

        def align_token_delimited(text: object) -> str:
            normalized = _normalize_ipa(text)
            input_tokens = normalized.split()
            unknown = next(
                (token for token in input_tokens if token not in source_tokens),
                None,
            )
            if unknown is not None:
                raise IPAInventoryMapError(
                    f"Unknown {source} IPA token: {unknown!r}"
                )

            aligned: list[str] = []
            offset = 0
            while offset < len(input_tokens):
                matched = next(
                    (
                        (sequence, replacement)
                        for sequence, replacement in sequence_replacements
                        if tuple(input_tokens[offset : offset + len(sequence)])
                        == sequence
                    ),
                    None,
                )
                if matched is not None:
                    sequence, replacement = matched
                    aligned.extend(replacement.split())
                    offset += len(sequence)
                else:
                    aligned.extend(adapter.replacements[input_tokens[offset]].split())
                    offset += 1
            return " ".join(aligned)

        return align_token_delimited

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
