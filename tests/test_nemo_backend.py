from __future__ import annotations

from pathlib import Path

import numpy as np

from inference.nemo import backend as nemo_backend
from inference.nemo.backend import NemoBackend
from inference.profile import parse_profile


def _profile():
    return parse_profile(
        {
            "id": "nemo-characterization",
            "framework": "nemo",
            "adapter": "nemo",
            "artifact": "model.nemo",
            "decoding": {"strategy": "ctc"},
        }
    )


def test_nemo_backend_preserves_order_and_writes_read_errors(
    monkeypatch, tmp_path: Path
) -> None:
    backend = NemoBackend.__new__(NemoBackend)
    backend.profile = _profile()
    backend.model = object()
    backend.override_cfg = object()
    backend.tmp_dir = tmp_path
    backend.debug = False

    good_paths = {"first.wav": 1.25, "second.wav": 2.5}

    def fake_load(path: str, target_sr: int):
        if path == "unreadable.wav":
            raise RuntimeError("cannot decode")
        return np.zeros(16, dtype=np.float32), target_sr, good_paths[path], False

    monkeypatch.setattr(nemo_backend, "load_audio_and_resample", fake_load)
    monkeypatch.setattr(
        nemo_backend,
        "transcribe_batches",
        lambda **kwargs: [" KWANZA ", "pili"],
    )

    rows = backend.transcribe_batch(["first.wav", "unreadable.wav", "second.wav"])

    assert [row.audio_filepath for row in rows] == [
        "first.wav",
        "unreadable.wav",
        "second.wav",
    ]
    assert [row.pred_text for row in rows] == ["KWANZA", "", "pili"]
    assert rows[0].duration == 1.25
    assert rows[1].duration == 0.0
    assert rows[1].error == "failed_to_read_audio: cannot decode"
    assert rows[2].duration == 2.5


def test_nemo_backend_turns_batch_failure_into_one_error_per_input(
    monkeypatch,
    tmp_path: Path,
) -> None:
    backend = NemoBackend.__new__(NemoBackend)
    backend.profile = _profile()
    backend.model = object()
    backend.override_cfg = object()
    backend.tmp_dir = tmp_path
    backend.debug = False

    monkeypatch.setattr(
        nemo_backend,
        "load_audio_and_resample",
        lambda path, target_sr: (
            np.zeros(16, dtype=np.float32),
            target_sr,
            1.0,
            False,
        ),
    )

    def fail_transcription(**kwargs):
        raise RuntimeError("model failed")

    monkeypatch.setattr(nemo_backend, "transcribe_batches", fail_transcription)

    rows = backend.transcribe_batch(["one.wav", "two.wav"])

    assert len(rows) == 2
    assert all(row.pred_text == "" for row in rows)
    assert all(row.error == "inference_failed: model failed" for row in rows)


def test_nemo_metadata_records_effective_decoder_and_override_config() -> None:
    backend = NemoBackend.__new__(NemoBackend)
    backend.profile = _profile()
    backend._model_class = "EncDecCTCModel"
    backend.device = "cuda:0"
    backend._effective_decoder_type = "ctc"
    backend._effective_decoding_config = {"strategy": "greedy_batch"}
    backend._transcribe_override_config = {"batch_size": 4, "num_workers": 0}
    backend.num_workers = 0
    backend.tmp_dir = Path("/tmp/nemo")
    backend.debug = False

    metadata = backend.metadata()

    assert metadata["effective_decoder_type"] == "ctc"
