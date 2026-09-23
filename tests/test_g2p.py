from __future__ import annotations

import hashlib
import sys
import types
import unicodedata
from pathlib import Path
from types import SimpleNamespace

import pytest

from egra_eval2.reference import g2p


def test_registry_and_identity_are_stable_and_configuration_sensitive() -> None:
    assert set(g2p.G2P_BACKENDS) == {"africa_g2p", "babygruut"}
    identity = {
        "tool_id": "babygruut",
        "language": "sw",
        "configuration": {"turso": False},
    }
    same_identity = {
        "configuration": {"turso": False},
        "language": "sw",
        "tool_id": "babygruut",
    }
    changed = {**identity, "configuration": {"turso": True}}

    assert g2p.canonical_identity_sha256(identity) == (
        g2p.canonical_identity_sha256(same_identity)
    )
    assert g2p.system_id_for_identity(identity) == (
        g2p.system_id_for_identity(same_identity)
    )
    assert g2p.system_id_for_identity(identity) != g2p.system_id_for_identity(changed)


def test_babygruut_identity_hashes_local_lexicon_and_crf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    language_dir = tmp_path / "sw"
    (language_dir / "g2p").mkdir(parents=True)
    (language_dir / "lexicon.db").write_bytes(b"lexicon")
    (language_dir / "g2p" / "model.crf").write_bytes(b"model")
    monkeypatch.setitem(
        sys.modules,
        "gruut_lang_sw",
        SimpleNamespace(get_lang_dir=lambda: language_dir),
    )
    monkeypatch.setattr(
        g2p,
        "_package_version",
        lambda name: {
            "babygruut": g2p.BABYGRUUT_VERSION,
            "gruut-lang-sw": g2p.GRUUT_LANG_SW_VERSION,
        }[name],
    )

    system = g2p._resolve_babygruut()

    assert system.identity["source_revision"] == g2p.BABYGRUUT_SOURCE_REVISION
    assert system.identity["resources"] == {
        "lexicon.db": hashlib.sha256(b"lexicon").hexdigest(),
        "g2p/model.crf": hashlib.sha256(b"model").hexdigest(),
    }
    assert system.identity["configuration"]["turso"] is False
    assert system.identity["configuration"]["source"] == "bundled_sqlite_then_crf"
    assert system.identity["configuration"]["verbalize_numbers"] is False


def test_babygruut_flattens_only_spoken_phones_and_disables_remote_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process_calls: list[tuple[str, dict]] = []
    sentence_calls: list[dict] = []
    lookup_calls: list[tuple[str, str | None]] = []
    guess_calls: list[tuple[str, str | None]] = []

    def lookup(word: str, role: str | None = None):
        lookup_calls.append((word, role))
        return None

    def guess(word: str, role: str | None = None):
        guess_calls.append((word, role))
        return [word]

    settings = SimpleNamespace(
        lookup_phonemes=lookup,
        guess_phonemes=guess,
    )

    class TextProcessor:
        def __init__(self, **kwargs):
            assert kwargs == {
                "default_lang": "sw",
                "model_prefix": "",
                "turso_config": None,
            }

        def get_settings(self, lang: str):
            assert lang == "sw"
            return settings

        def __call__(self, text: str, **kwargs):
            process_calls.append((text, kwargs))
            # Exercise the exact per-word caches independently of the fake
            # sentence output.
            settings.lookup_phonemes(text, None)
            settings.lookup_phonemes(text, None)
            settings.guess_phonemes(text, None)
            settings.guess_phonemes(text, None)
            return text, object()

        def sentences(self, graph, root, **kwargs):
            sentence_calls.append(kwargs)
            if not graph:
                return []
            spoken = SimpleNamespace(
                text=graph,
                is_spoken=True,
                phonemes=["a\u0303", "t͡ʃ"] if graph != "oovword" else ["ɔ", "v"],
            )
            punctuation = SimpleNamespace(text="!", is_spoken=False, phonemes=["‖"])
            return [[spoken, punctuation]]

    monkeypatch.setitem(
        sys.modules, "gruut", SimpleNamespace(TextProcessor=TextProcessor)
    )
    phonemize = g2p._babygruut_phonemizer()

    assert phonemize(["", "Habari!", "oovword", "Habari!"]) == [
        "",
        f"{unicodedata.normalize('NFC', 'ã')} t͡ʃ",
        "ɔ v",
        f"{unicodedata.normalize('NFC', 'ã')} t͡ʃ",
    ]
    assert [text for text, _ in process_calls].count("Habari!") == 1
    assert len(lookup_calls) == 3
    assert len(guess_calls) == 3
    for _, kwargs in process_calls:
        assert kwargs["lang"] == "sw"
        assert kwargs["ssml"] is False
        assert kwargs["verbalize_numbers"] is False
    for kwargs in sentence_calls:
        assert kwargs["major_breaks"] is False
        assert kwargs["minor_breaks"] is False
        assert kwargs["punctuations"] is False
        assert kwargs["break_phonemes"] is False
        assert kwargs["pos"] is False


def test_babygruut_fails_when_a_spoken_token_has_no_pronunciation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    word = SimpleNamespace(text="missing", is_spoken=True, phonemes=[])

    class TextProcessor:
        def __init__(self, **kwargs):
            pass

        def get_settings(self, lang: str):
            return SimpleNamespace(lookup_phonemes=None, guess_phonemes=None)

        def __call__(self, text: str, **kwargs):
            return object(), object()

        def sentences(self, *args, **kwargs):
            return [[word]]

    monkeypatch.setitem(
        sys.modules, "gruut", SimpleNamespace(TextProcessor=TextProcessor)
    )

    with pytest.raises(g2p.G2PSystemError, match="no phonemes"):
        g2p._babygruut_phonemizer()(["missing"])


def test_africa_g2p_enforces_one_result_per_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Pipeline:
        def __init__(self, **kwargs):
            pass

        def run(self, values, **kwargs):
            return ["only one"]

    monkeypatch.setitem(
        sys.modules, "africa_g2p", types.SimpleNamespace(AfricaPipeline=Pipeline)
    )
    with pytest.raises(g2p.G2PSystemError, match="one IPA value"):
        g2p._africa_phonemizer("swh")(["one", "two"])
