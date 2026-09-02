from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from inference.torch import backend as torch_backend
from inference.torch.backend import (
    TorchScriptStreamingTransducerBackend,
    _load_token_table,
)


class _FakeFbank:
    def __init__(self) -> None:
        self.samples = 0

    @property
    def num_frames_ready(self) -> int:
        return self.samples // 160

    def accept_waveform(self, sample_rate: int, samples: list[float]) -> None:
        assert sample_rate == 16000
        self.samples += len(samples)

    def get_frame(self, _index: int) -> np.ndarray:
        return np.zeros(80, dtype=np.float32)


class _FakeEncoder:
    chunk_size = 32
    pad_length = 13

    def __init__(self) -> None:
        self.calls = 0

    def get_init_states(self, *, batch_size: int, device: torch.device) -> list:
        assert batch_size == 1
        assert device.type == "cpu"
        return []

    def __call__(self, **kwargs):
        self.calls += 1
        assert kwargs["features"].shape == (1, 77, 80)
        assert kwargs["feature_lengths"].tolist() == [77]
        return torch.zeros((1, 1, 2)), torch.tensor([1]), kwargs["states"]


class _FakeBeamDecoder:
    def __call__(self, decoder_input: torch.Tensor, need_pad: bool) -> torch.Tensor:
        assert decoder_input.dtype == torch.int64
        assert need_pad is False
        last_token = decoder_input[:, -1].to(torch.float32)
        return torch.stack((last_token, torch.ones_like(last_token)), dim=-1).unsqueeze(1)


class _FakeBeamJoiner:
    def encoder_proj(self, encoder_out: torch.Tensor) -> torch.Tensor:
        assert encoder_out.ndim == 2
        return encoder_out

    def decoder_proj(self, decoder_out: torch.Tensor) -> torch.Tensor:
        assert decoder_out.ndim == 4
        return decoder_out

    def __call__(
        self,
        encoder_out: torch.Tensor,
        decoder_out: torch.Tensor,
        project_input: bool,
    ) -> torch.Tensor:
        assert encoder_out.ndim == 4
        assert decoder_out.ndim == 4
        assert project_input is False
        last_tokens = decoder_out[:, 0, 0, 0].to(torch.int64)
        logits = torch.full(
            (last_tokens.numel(), 1, 1, 3),
            -8.0,
            dtype=torch.float32,
        )
        for index, token in enumerate(last_tokens.tolist()):
            if token == 0:
                logits[index, 0, 0] = torch.tensor([4.0, 5.0, -8.0])
            elif token == 1:
                logits[index, 0, 0] = torch.tensor([5.0, -8.0, -8.0])
            else:
                logits[index, 0, 0] = torch.tensor([5.0, -8.0, -8.0])
        return logits


def _fake_streaming_backend() -> TorchScriptStreamingTransducerBackend:
    backend = object.__new__(TorchScriptStreamingTransducerBackend)
    backend._torch = torch
    backend.device = torch.device("cpu")
    backend.encoder = _FakeEncoder()
    backend.chunk_length = 64
    backend.pad_length = 13
    backend.feature_window = 77
    backend.context_size = 2
    backend.blank_id = 0
    backend.decoding_method = "greedy_search"
    backend.token_table = {0: "<blk>", 1: "ɑ"}
    backend.tokens_file = Path("tokens.txt")
    backend.decoded_windows = 0
    backend.short_input_padding_files = 0
    backend.short_input_padding_samples = 0
    backend._new_fbank = _FakeFbank
    backend._decode_greedy_chunk = lambda *_args: ([0, 0, 1], None)
    return backend


def test_load_token_table_preserves_ipa_symbols(tmp_path: Path) -> None:
    path = tmp_path / "tokens.txt"
    path.write_text("<eps> 0\n<UNK> 1\nt͡ʃ 2\nɑ 3\n", encoding="utf-8")

    table, unknown_id = _load_token_table(path)

    assert table == {0: "<eps>", 1: "<UNK>", 2: "t͡ʃ", 3: "ɑ"}
    assert unknown_id == 1


def test_load_token_table_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "tokens.txt"
    path.write_text("<eps> 0\na 0\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Duplicate token id 0"):
        _load_token_table(path)


def test_tokens_to_text_fails_closed_for_unmapped_decoder_id() -> None:
    backend = _fake_streaming_backend()

    with pytest.raises(RuntimeError, match="token id 9"):
        backend._tokens_to_text([9])


def test_short_input_is_padded_to_one_encoder_window() -> None:
    backend = _fake_streaming_backend()

    text = backend._transcribe_audio(np.zeros(1600, dtype=np.float32))

    assert text == "ɑ"
    assert backend.encoder.calls == 1
    assert backend.decoded_windows == 1
    assert backend.short_input_padding_files == 1
    assert backend.short_input_padding_samples == 8000


def test_normal_input_preserves_owner_no_residual_flush_policy() -> None:
    backend = _fake_streaming_backend()

    backend._transcribe_audio(np.zeros(16000, dtype=np.float32))

    assert backend.encoder.calls == 1
    assert backend.short_input_padding_files == 0
    assert backend.short_input_padding_samples == 0


def test_modified_beam_preserves_hypotheses_across_chunks_and_merges_duplicates() -> None:
    backend = _fake_streaming_backend()
    backend.decoding_method = "modified_beam_search"
    backend.max_active_paths = 4
    backend.decoder = _FakeBeamDecoder()
    backend.joiner = _FakeBeamJoiner()
    encoder_out = torch.zeros((1, 2), dtype=torch.float32)

    first = backend._decode_modified_beam_chunk(encoder_out, None)
    duplicate_key = (-1, 0, 1)
    assert duplicate_key in first

    second = backend._decode_modified_beam_chunk(encoder_out, first)

    assert duplicate_key in second
    assert len(second) <= 4
    assert all(tokens[:2] == (-1, 0) for tokens in second)
    assert torch.isfinite(second[duplicate_key])


def test_transcription_error_preserves_loaded_audio_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _fake_streaming_backend()
    backend.resampled_inputs = 0
    monkeypatch.setattr(
        torch_backend,
        "load_audio_and_resample",
        lambda *_args: (np.zeros(8, dtype=np.float32), 16000, 1.25, False),
    )
    backend._transcribe_audio = lambda _audio: (_ for _ in ()).throw(
        RuntimeError("decoder failed")
    )

    result = backend.transcribe_batch(["sample.wav"])[0]

    assert result.duration == 1.25
    assert result.error == "RuntimeError: decoder failed"
