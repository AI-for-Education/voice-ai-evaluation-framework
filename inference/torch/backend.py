"""Native streaming Zipformer inference from BookBot's TorchScript export."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from inference.common import load_audio_and_resample
from inference.contracts import TranscriptionResult
from inference.profile import InferenceProfile, ProfileError


SAMPLE_RATE = 16000
FEATURE_BINS = 80
TAIL_PADDING_SECONDS = 0.3
AUDIO_CHUNK_SECONDS = 0.25
LEFT_CONTEXT_FRAMES = 128
MODEL_RELATIVE_PATH = Path("exp-causal/jit_script_chunk_32_left_128.pt")
TOKENS_RELATIVE_PATH = Path("data/lang_phone/tokens.txt")


def _load_token_table(path: Path) -> tuple[dict[int, str], int | None]:
    symbols: dict[int, str] = {}
    unknown_id: int | None = None
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        symbol, separator, raw_id = line.rpartition(" ")
        if not separator or not symbol:
            raise RuntimeError(f"Malformed token row {line_number} in {path}")
        try:
            token_id = int(raw_id)
        except ValueError as exc:
            raise RuntimeError(
                f"Malformed token id on row {line_number} in {path}"
            ) from exc
        if token_id in symbols:
            raise RuntimeError(f"Duplicate token id {token_id} in {path}")
        symbols[token_id] = symbol
        if symbol in {"<UNK>", "<unk>"}:
            unknown_id = token_id
    if not symbols:
        raise RuntimeError(f"No tokens found in {path}")
    return symbols, unknown_id


class TorchScriptStreamingTransducerBackend:
    """File transcription through BookBot's native streaming JIT export."""

    def __init__(
        self,
        profile: InferenceProfile,
        model_path: str | Path,
        *,
        device: str | None = None,
        num_threads: int = 2,
    ) -> None:
        if (
            profile.inference_library != "torch"
            or profile.adapter != "streaming_transducer"
        ):
            raise ProfileError(
                "TorchScriptStreamingTransducerBackend requires a "
                "torch/streaming_transducer profile"
            )
        if profile.decoding.strategy not in {
            "greedy_search",
            "modified_beam_search",
        }:
            raise ProfileError(
                "The native transducer adapter requires greedy_search or "
                "modified_beam_search decoding"
            )
        if num_threads < 1:
            raise ProfileError("Torch num_threads must be at least 1")

        self.profile = profile
        self.model_path = Path(model_path)
        if not self.model_path.is_dir():
            raise ProfileError(f"Torch model directory not found: {self.model_path}")
        self.model_file = self.model_path / MODEL_RELATIVE_PATH
        self.tokens_file = self.model_path / TOKENS_RELATIVE_PATH
        if not self.model_file.is_file():
            raise RuntimeError(f"TorchScript model is missing: {self.model_file}")
        if not self.tokens_file.is_file():
            raise RuntimeError(f"Token table is missing: {self.tokens_file}")

        try:
            import kaldi_native_fbank
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Native TorchScript inference requires torch and "
                "kaldi-native-fbank in the image"
            ) from exc

        self._fbank_module = kaldi_native_fbank
        self._torch = torch
        self.num_threads = num_threads
        torch.set_num_threads(num_threads)
        self.device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.decoding_method = profile.decoding.strategy
        search_config = profile.decoding.transducer_search_kwargs
        self.max_active_paths = (
            search_config.max_active_paths if search_config is not None else None
        )
        self.token_table, self.unknown_id = _load_token_table(self.tokens_file)
        try:
            self.model = torch.jit.load(str(self.model_file), map_location=self.device)
            self.model.eval()
            self.model.to(self.device)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load local TorchScript model from {self.model_file}: {exc}"
            ) from exc

        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        self.joiner = self.model.joiner
        self.chunk_length = int(self.encoder.chunk_size) * 2
        self.pad_length = int(self.encoder.pad_length)
        self.feature_window = self.chunk_length + self.pad_length
        self.context_size = int(self.decoder.context_size)
        self.blank_id = int(self.decoder.blank_id)
        self.resampled_inputs = 0
        self.decoded_windows = 0
        self.short_input_padding_files = 0
        self.short_input_padding_samples = 0

    def _new_fbank(self) -> Any:
        opts = self._fbank_module.FbankOptions()
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = False
        opts.frame_opts.samp_freq = SAMPLE_RATE
        opts.mel_opts.num_bins = FEATURE_BINS
        opts.mel_opts.high_freq = -400
        return self._fbank_module.OnlineFbank(opts)

    def _decode_greedy_chunk(
        self,
        encoder_out: Any,
        hypothesis: list[int] | None,
        decoder_out: Any | None,
    ) -> tuple[list[int], Any]:
        torch = self._torch
        if decoder_out is None:
            hypothesis = [self.blank_id] * self.context_size
            decoder_input = torch.tensor(
                hypothesis,
                dtype=torch.int32,
                device=self.device,
            ).unsqueeze(0)
            decoder_out = self.decoder(decoder_input, False).squeeze(1)
        assert hypothesis is not None
        for frame_index in range(int(encoder_out.size(0))):
            current_encoder_out = encoder_out[frame_index : frame_index + 1]
            logits = self.joiner(current_encoder_out, decoder_out, True).squeeze(0)
            token = int(logits.argmax(dim=-1).item())
            if token != self.blank_id:
                hypothesis.append(token)
                decoder_input = torch.tensor(
                    hypothesis[-self.context_size :],
                    dtype=torch.int32,
                    device=self.device,
                ).unsqueeze(0)
                decoder_out = self.decoder(decoder_input, False).squeeze(1)
        return hypothesis, decoder_out

    def _decode_modified_beam_chunk(
        self,
        encoder_out: Any,
        hypotheses: dict[tuple[int, ...], Any] | None,
    ) -> dict[tuple[int, ...], Any]:
        torch = self._torch
        if self.max_active_paths is None:
            raise RuntimeError("Modified beam search is missing max_active_paths")
        if hypotheses is None:
            initial = tuple(
                [-1] * (self.context_size - 1) + [self.blank_id]
            )
            hypotheses = {
                initial: torch.zeros((), dtype=torch.float32, device=self.device)
            }

        projected_encoder_out = self.joiner.encoder_proj(encoder_out)
        for frame_index in range(int(encoder_out.size(0))):
            active = list(hypotheses.items())
            decoder_input = torch.tensor(
                [list(tokens[-self.context_size :]) for tokens, _ in active],
                dtype=torch.int64,
                device=self.device,
            )
            decoder_out = self.decoder(decoder_input, False).unsqueeze(1)
            decoder_out = self.joiner.decoder_proj(decoder_out)
            current_encoder_out = projected_encoder_out[
                frame_index : frame_index + 1
            ]
            current_encoder_out = current_encoder_out.expand(len(active), -1)
            current_encoder_out = current_encoder_out.unsqueeze(1).unsqueeze(1)
            logits = self.joiner(current_encoder_out, decoder_out, False)
            logits = logits.reshape(len(active), -1)
            log_probs = torch.log_softmax(logits, dim=-1)
            prior_scores = torch.stack([score for _, score in active]).reshape(-1, 1)
            combined = (log_probs + prior_scores).reshape(-1)
            path_count = min(self.max_active_paths, int(combined.numel()))
            top_scores, top_indexes = torch.topk(combined, k=path_count)
            vocabulary_size = int(log_probs.size(-1))
            next_hypotheses: dict[tuple[int, ...], Any] = {}
            for rank in range(path_count):
                flat_index = int(top_indexes[rank].item())
                hypothesis_index = flat_index // vocabulary_size
                token = flat_index % vocabulary_size
                previous_tokens = active[hypothesis_index][0]
                if token == self.blank_id:
                    next_tokens = previous_tokens
                else:
                    next_tokens = (*previous_tokens, token)
                score = top_scores[rank]
                if next_tokens in next_hypotheses:
                    next_hypotheses[next_tokens] = torch.logaddexp(
                        next_hypotheses[next_tokens], score
                    )
                else:
                    next_hypotheses[next_tokens] = score
            hypotheses = next_hypotheses
        return hypotheses

    def _tokens_to_text(self, token_ids: Sequence[int]) -> str:
        pieces: list[str] = []
        for raw_token_id in token_ids:
            token_id = int(raw_token_id)
            if token_id not in self.token_table:
                raise RuntimeError(
                    f"Decoder emitted token id {token_id}, which is absent from "
                    f"{self.tokens_file}"
                )
            pieces.append(self.token_table[token_id])
        return "".join(pieces).replace("▁", " ").strip()

    def _transcribe_audio(self, audio: np.ndarray) -> str:
        torch = self._torch
        fbank = self._new_fbank()
        tail = np.zeros(int(TAIL_PADDING_SECONDS * SAMPLE_RATE), dtype=np.float32)
        wave = np.concatenate((audio, tail))
        samples_per_chunk = int(AUDIO_CHUNK_SECONDS * SAMPLE_RATE)
        num_processed_frames = 0
        states = self.encoder.get_init_states(batch_size=1, device=self.device)
        greedy_hypothesis: list[int] | None = None
        greedy_decoder_out: Any | None = None
        beam_hypotheses: dict[tuple[int, ...], Any] | None = None
        decoded_windows = 0
        extra_short_padding_samples = 0

        def decode_ready_windows() -> None:
            nonlocal num_processed_frames
            nonlocal states
            nonlocal greedy_hypothesis
            nonlocal greedy_decoder_out
            nonlocal beam_hypotheses
            nonlocal decoded_windows

            while (
                int(fbank.num_frames_ready) - num_processed_frames
                >= self.feature_window
            ):
                features = np.stack(
                    [
                        np.asarray(
                            fbank.get_frame(num_processed_frames + offset),
                            dtype=np.float32,
                        )
                        for offset in range(self.feature_window)
                    ]
                )
                feature_tensor = (
                    torch.from_numpy(features).to(self.device).unsqueeze(0)
                )
                feature_lengths = torch.tensor(
                    [self.feature_window],
                    dtype=torch.int32,
                    device=self.device,
                )
                encoder_out, _out_lens, states = self.encoder(
                    features=feature_tensor,
                    feature_lengths=feature_lengths,
                    states=states,
                )
                num_processed_frames += self.chunk_length
                decoded_windows += 1
                encoder_out = encoder_out.squeeze(0)
                if self.decoding_method == "greedy_search":
                    greedy_hypothesis, greedy_decoder_out = (
                        self._decode_greedy_chunk(
                            encoder_out,
                            greedy_hypothesis,
                            greedy_decoder_out,
                        )
                    )
                else:
                    beam_hypotheses = self._decode_modified_beam_chunk(
                        encoder_out,
                        beam_hypotheses,
                    )

        with torch.no_grad():
            for start in range(0, int(wave.size), samples_per_chunk):
                samples = wave[start : start + samples_per_chunk]
                fbank.accept_waveform(SAMPLE_RATE, samples.tolist())
                decode_ready_windows()

            # The owner script appends 0.3 seconds and does not generally flush
            # a partial final window. Three frozen benchmark segments are too
            # short to produce even one window under that policy, so pad only
            # that edge case to the first complete window instead of silently
            # returning a successful empty hypothesis.
            while decoded_windows == 0 and (
                int(fbank.num_frames_ready) - num_processed_frames
                < self.feature_window
            ):
                padding = np.zeros(samples_per_chunk, dtype=np.float32)
                fbank.accept_waveform(SAMPLE_RATE, padding.tolist())
                extra_short_padding_samples += int(padding.size)
            decode_ready_windows()

        if decoded_windows == 0:
            raise RuntimeError("No complete encoder window could be constructed")
        self.decoded_windows += decoded_windows
        if extra_short_padding_samples:
            self.short_input_padding_files += 1
            self.short_input_padding_samples += extra_short_padding_samples

        if self.decoding_method == "greedy_search":
            if greedy_hypothesis is None:
                return ""
            return self._tokens_to_text(greedy_hypothesis[self.context_size :])
        if not beam_hypotheses:
            return ""
        best_tokens, _best_score = max(
            beam_hypotheses.items(),
            key=lambda item: float((item[1] / len(item[0])).item()),
        )
        return self._tokens_to_text(best_tokens[self.context_size :])

    def transcribe_batch(self, audio_paths: Sequence[str]) -> list[TranscriptionResult]:
        rows: list[TranscriptionResult] = []
        for raw_path in audio_paths:
            audio_path = str(raw_path)
            duration = 0.0
            try:
                audio, _, duration, was_resampled = load_audio_and_resample(
                    audio_path,
                    SAMPLE_RATE,
                )
                self.resampled_inputs += int(was_resampled)
                text = self._transcribe_audio(audio)
                rows.append(
                    TranscriptionResult(
                        audio_filepath=audio_path,
                        duration=duration,
                        pred_text=text,
                    )
                )
            except Exception as exc:
                rows.append(
                    TranscriptionResult(
                        audio_filepath=audio_path,
                        duration=duration,
                        pred_text="",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        return rows

    def metadata(self) -> dict[str, Any]:
        try:
            fbank_version = importlib.metadata.version("kaldi-native-fbank")
        except importlib.metadata.PackageNotFoundError:
            fbank_version = str(
                getattr(self._fbank_module, "__version__", "unknown")
            )
        payload: dict[str, Any] = {
            "inference_library": "torch",
            "adapter": "streaming_transducer",
            "device": str(self.device),
            "sampling_rate": SAMPLE_RATE,
            "decoding_strategy": self.decoding_method,
            "artifact_format": "torchscript",
            "artifact_precision": "fp32",
            "torch_version": str(self._torch.__version__),
            "requested_torch_dtype": self.profile.loader.torch_dtype,
            "effective_torch_dtype": "float32",
            "kaldi_native_fbank_version": fbank_version,
            "num_threads": self.num_threads,
            "model_file": str(MODEL_RELATIVE_PATH.as_posix()),
            "tokens_file": str(TOKENS_RELATIVE_PATH.as_posix()),
            "selected_artifact_files": [
                str(MODEL_RELATIVE_PATH.as_posix()),
                str(TOKENS_RELATIVE_PATH.as_posix()),
            ],
            "token_table_size": len(self.token_table),
            "unknown_token_id": self.unknown_id,
            "feature_extractor": {
                "implementation": "kaldi_native_fbank.OnlineFbank",
                "sample_rate_hz": SAMPLE_RATE,
                "feature_bins": FEATURE_BINS,
                "dither": 0,
                "snip_edges": False,
                "high_freq": -400,
            },
            "streaming_geometry": {
                "encoder_chunk_size": int(self.encoder.chunk_size),
                "chunk_length": self.chunk_length,
                "left_context_frames": LEFT_CONTEXT_FRAMES,
                "pad_length": self.pad_length,
                "feature_window": self.feature_window,
                "tail_padding_seconds": TAIL_PADDING_SECONDS,
                "audio_feed_chunk_seconds": AUDIO_CHUNK_SECONDS,
                "finalization_policy": (
                    "owner_tail_no_general_residual_flush_with_"
                    "minimum_one_window_short_input_padding"
                ),
            },
            "resampled_inputs": self.resampled_inputs,
            "decoded_windows": self.decoded_windows,
            "short_input_padding_files": self.short_input_padding_files,
            "short_input_padding_samples": self.short_input_padding_samples,
        }
        if self.max_active_paths is not None:
            payload["max_active_paths"] = self.max_active_paths
        return payload

    def close(self) -> None:
        self.model = None
        self.encoder = None
        self.decoder = None
        self.joiner = None
