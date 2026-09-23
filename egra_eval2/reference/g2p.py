"""Pinned, provenance-rich G2P systems used to build IPA scoring views."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from inference.provenance import sha256_file


BABYGRUUT_SOURCE_REVISION = "85eadf1743455ba57ccb1baccf39485dc8628c33"
BABYGRUUT_VERSION = "0.0.7.post4+g85eadf174"
GRUUT_LANG_SW_VERSION = "2.0.1"
AFRICA_G2P_VERSION = "0.1.0"
SUPPORTED_G2P_TOOLS = ("africa_g2p", "babygruut")


class G2PSystemError(ValueError):
    """Raised when a configured G2P system cannot be identified or executed."""


PhonemizeBatch = Callable[[Sequence[str]], list[str]]


def canonical_identity_sha256(identity: Mapping[str, Any]) -> str:
    """Hash a G2P identity using a stable JSON representation."""
    payload = json.dumps(
        dict(identity),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _safe_component(value: str) -> str:
    component = re.sub(r"[^a-z0-9_-]+", "-", value.strip().lower()).strip("-")
    if not component:
        raise G2PSystemError("G2P identity component must not be empty")
    return component


def system_id_for_identity(identity: Mapping[str, Any]) -> str:
    """Return the filesystem-safe exact-system identifier for an identity."""
    tool_id = _safe_component(str(identity.get("tool_id", "")))
    language = _safe_component(str(identity.get("language", "")))
    return f"{tool_id}-{language}-{canonical_identity_sha256(identity)[:12]}"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError as exc:
        raise G2PSystemError(
            f"Required G2P package is not installed: {name}"
        ) from exc


@dataclass(frozen=True)
class G2PSystem:
    """One exact G2P implementation, resource set, and configuration."""

    tool_id: str
    display_name: str
    language: str
    inventory: str
    identity: Mapping[str, Any]
    phonemize: PhonemizeBatch

    @property
    def identity_sha256(self) -> str:
        return canonical_identity_sha256(self.identity)

    @property
    def system_id(self) -> str:
        return system_id_for_identity(self.identity)

    def metadata(self) -> dict[str, Any]:
        return {
            "system_id": self.system_id,
            "identity_sha256": self.identity_sha256,
            "tool_id": self.tool_id,
            "display_name": self.display_name,
            "language": self.language,
            "inventory": self.inventory,
            "identity": dict(self.identity),
        }


def make_test_system(
    *,
    tool_id: str = "africa_g2p",
    display_name: str = "Africa G2P",
    language: str = "swh",
    inventory: str = "africa_g2p_swh_ipa_v1",
    version_value: str = "test-version",
    phonemize: PhonemizeBatch,
) -> G2PSystem:
    """Build a deterministic dependency-free system for unit tests/callers."""
    identity = {
        "schema_version": 1,
        "tool_id": tool_id,
        "language": language,
        "inventory": inventory,
        "packages": {tool_id.replace("_", "-"): version_value},
        "resources": {"test_fixture": version_value},
        "configuration": {"mode": "test"},
    }
    return G2PSystem(
        tool_id=tool_id,
        display_name=display_name,
        language=language,
        inventory=inventory,
        identity=identity,
        phonemize=phonemize,
    )


def _africa_phonemizer(language: str) -> PhonemizeBatch:
    pipeline: Any | None = None

    def phonemize(texts: Sequence[str]) -> list[str]:
        nonlocal pipeline
        try:
            from africa_g2p import AfricaPipeline
        except ImportError as exc:
            raise G2PSystemError(
                "Africa G2P is not installed. Rebuild the evaluation image."
            ) from exc

        if pipeline is None:
            pipeline = AfricaPipeline(
                lang=language,
                output="ipa",
                unknown="passthrough",
            )
        values = pipeline.run(list(texts), sep=" ")
        if not isinstance(values, list) or len(values) != len(texts):
            raise G2PSystemError(
                "Africa G2P must return one IPA value for every input text"
            )
        return [str(value) for value in values]

    return phonemize


def _resolve_africa_g2p() -> G2PSystem:
    package_version = _package_version("africa-g2p")
    if package_version != AFRICA_G2P_VERSION:
        raise G2PSystemError(
            "Africa G2P version does not match the pinned evaluation system: "
            f"expected {AFRICA_G2P_VERSION}, found {package_version}"
        )
    try:
        language_data = files("africa_g2p").joinpath("languages", "swh.json")
        language_sha256 = _sha256_bytes(language_data.read_bytes())
    except (FileNotFoundError, ModuleNotFoundError, OSError) as exc:
        raise G2PSystemError(
            "Africa G2P Swahili language data could not be identified"
        ) from exc

    identity = {
        "schema_version": 1,
        "tool_id": "africa_g2p",
        "language": "swh",
        "inventory": "africa_g2p_swh_ipa_v1",
        "packages": {"africa-g2p": package_version},
        "resources": {"languages/swh.json": language_sha256},
        "configuration": {
            "output": "ipa",
            "unknown": "passthrough",
            "separator": " ",
        },
    }
    return G2PSystem(
        tool_id="africa_g2p",
        display_name="Africa G2P",
        language="swh",
        inventory="africa_g2p_swh_ipa_v1",
        identity=identity,
        phonemize=_africa_phonemizer("swh"),
    )


def _babygruut_phonemizer() -> PhonemizeBatch:
    cache: dict[str, str] = {}
    processor: Any | None = None

    def initialize_processor() -> Any:
        nonlocal processor
        if processor is not None:
            return processor
        try:
            from gruut import TextProcessor
        except ImportError as exc:
            raise G2PSystemError(
                "babygruut is not installed. Rebuild the evaluation image."
            ) from exc

        processor = TextProcessor(
            default_lang="sw",
            model_prefix="",
            turso_config=None,
        )
        settings = processor.get_settings("sw")
        # babygruut's lexicon and CRF calls are deterministic per word/role. The
        # upstream SQLite layer caches successful lookups, but not misses or CRF
        # guesses; memoizing both avoids repeating identical work in long ASR
        # hallucinations without changing the phoneme sequence.
        if settings.lookup_phonemes is not None:
            settings.lookup_phonemes = lru_cache(maxsize=None)(
                settings.lookup_phonemes
            )
        if settings.guess_phonemes is not None:
            settings.guess_phonemes = lru_cache(maxsize=None)(
                settings.guess_phonemes
            )
        return processor

    def phonemize(texts: Sequence[str]) -> list[str]:
        text_processor = initialize_processor()
        results: list[str] = []
        for text in texts:
            source_text = str(text)
            cached = cache.get(source_text)
            if cached is not None:
                results.append(cached)
                continue
            phones: list[str] = []
            graph, root = text_processor(
                source_text,
                lang="sw",
                ssml=False,
                verbalize_numbers=False,
            )
            for sentence in text_processor.sentences(
                graph,
                root,
                major_breaks=False,
                minor_breaks=False,
                punctuations=False,
                explicit_lang=False,
                phonemes=True,
                break_phonemes=False,
                pos=False,
            ):
                for word in sentence:
                    if not word.is_spoken:
                        continue
                    if not word.phonemes:
                        raise G2PSystemError(
                            "babygruut produced no phonemes for spoken token: "
                            f"{word.text!r}"
                        )
                    phones.extend(str(phone) for phone in word.phonemes)
            value = " ".join(
                unicodedata.normalize("NFC", phone) for phone in phones
            )
            cache[source_text] = value
            results.append(value)
        if len(results) != len(texts):
            raise G2PSystemError(
                "babygruut must return one IPA value for every input text"
            )
        return results

    return phonemize


def _resolve_babygruut() -> G2PSystem:
    core_version = _package_version("babygruut")
    language_version = _package_version("gruut-lang-sw")
    if (
        core_version != BABYGRUUT_VERSION
        or language_version != GRUUT_LANG_SW_VERSION
    ):
        raise G2PSystemError(
            "babygruut package versions do not match the pinned source revision: "
            f"expected {BABYGRUUT_VERSION}/{GRUUT_LANG_SW_VERSION}, found "
            f"{core_version}/{language_version}"
        )
    try:
        from gruut_lang_sw import get_lang_dir
    except ImportError as exc:
        raise G2PSystemError(
            "babygruut Swahili resources are not installed. Rebuild the evaluation image."
        ) from exc

    language_dir = Path(get_lang_dir())
    lexicon_path = language_dir / "lexicon.db"
    model_path = language_dir / "g2p" / "model.crf"
    if not lexicon_path.is_file() or not model_path.is_file():
        raise G2PSystemError(
            "babygruut Swahili lexicon or CRF model is missing"
        )

    identity = {
        "schema_version": 1,
        "tool_id": "babygruut",
        "language": "sw",
        "inventory": "babygruut_sw_ipa_v1",
        "source_revision": BABYGRUUT_SOURCE_REVISION,
        "packages": {
            "babygruut": core_version,
            "gruut-lang-sw": language_version,
        },
        "resources": {
            "lexicon.db": sha256_file(lexicon_path),
            "g2p/model.crf": sha256_file(model_path),
        },
        "configuration": {
            "source": "bundled_sqlite_then_crf",
            "turso": False,
            "espeak": False,
            "spoken_words_only": True,
            "include_breaks": False,
            "include_punctuation": False,
            "pos": False,
            "verbalize_numbers": False,
        },
    }
    return G2PSystem(
        tool_id="babygruut",
        display_name="babygruut",
        language="sw",
        inventory="babygruut_sw_ipa_v1",
        identity=identity,
        phonemize=_babygruut_phonemizer(),
    )


G2P_BACKENDS: Mapping[str, Callable[[], G2PSystem]] = {
    "africa_g2p": _resolve_africa_g2p,
    "babygruut": _resolve_babygruut,
}


def resolve_g2p_system(tool_id: str, *, language: str = "swh") -> G2PSystem:
    """Resolve a supported tool to the exact installed G2P system."""
    normalized_tool = tool_id.strip().lower()
    normalized_language = language.strip().lower()
    if normalized_tool not in SUPPORTED_G2P_TOOLS:
        raise G2PSystemError(
            f"Unsupported G2P tool: {tool_id}. Choose from: "
            + ", ".join(SUPPORTED_G2P_TOOLS)
        )
    if normalized_language not in {"sw", "swh"}:
        raise G2PSystemError(
            f"G2P tool {normalized_tool} currently supports only Swahili"
        )
    return G2P_BACKENDS[normalized_tool]()
