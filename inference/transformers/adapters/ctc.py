"""Greedy and packaged-LM CTC inference for local Hugging Face checkpoints."""

from __future__ import annotations

import importlib
import importlib.metadata
import multiprocessing
import os
from pathlib import Path
from typing import Any, Sequence

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from transformers import (
    AutoModelForCTC,
    AutoProcessor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

from inference.common import load_audio_and_resample
from inference.contracts import TranscriptionResult
from inference.profile import CTCLMConfig, InferenceProfile, ProfileError


_LM_REQUIRED_FILES = (
    "alphabet.json",
    "config.json",
    "preprocessor_config.json",
    "vocab.json",
    "language_model/5gram_mms.bin",
    "language_model/unigrams.txt",
    "language_model/attrs.json",
)


def _select_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _loader_dtype(
    requested: str,
    device: torch.device,
) -> tuple[torch.dtype | str, str | None]:
    """Return a safe from_pretrained dtype and an optional fallback reason."""
    if device.type != "cuda":
        fallback = (
            None
            if requested in {"auto", "float32"}
            else (
                f"{requested} was promoted to float32 because inference is running on CPU"
            )
        )
        return torch.float32, fallback

    if requested == "auto":
        return "auto", None
    if requested == "float16":
        return torch.float16, None
    if requested == "bfloat16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, None
        return torch.float32, "bfloat16 was promoted to float32 on this CUDA device"
    return torch.float32, None


def _model_dtype(model: Any) -> torch.dtype:
    try:
        return next(model.parameters()).dtype
    except (AttributeError, StopIteration):
        return torch.float32


def _move_model_inputs(
    encoded: Any,
    *,
    device: torch.device,
    floating_dtype: torch.dtype,
) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    for name, value in encoded.items():
        if not hasattr(value, "to"):
            inputs[name] = value
        elif torch.is_floating_point(value):
            inputs[name] = value.to(device=device, dtype=floating_dtype)
        else:
            inputs[name] = value.to(device=device)
    return inputs


def _distribution_version(name: str, module: Any) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return str(getattr(module, "__version__", "unknown"))


def _create_decoder_pool(workers: int) -> Any:
    if "fork" not in multiprocessing.get_all_start_methods():
        raise RuntimeError(
            "CTC LM decoder_workers > 0 requires a Linux/Unix fork multiprocessing "
            "context; use the Docker inference image or set decoder_workers to 0"
        )
    return multiprocessing.get_context("fork").Pool(processes=workers)


class TransformersCTCBackend:
    """Offline Transformers backend with isolated greedy and LM decode paths."""

    def __init__(
        self,
        profile: InferenceProfile,
        model_path: str | Path,
        *,
        device: torch.device | None = None,
    ) -> None:
        if profile.inference_library != "transformers" or profile.adapter != "ctc":
            raise ProfileError(
                "TransformersCTCBackend requires a transformers/ctc profile"
            )
        if profile.decoding.strategy not in {"greedy", "beam_search"}:
            raise ProfileError(
                "The CTC adapter supports greedy or beam_search decoding"
            )

        self.profile = profile
        self.model_path = Path(model_path)
        self.model = None
        self.processor = None
        self._decoder_pool = None
        self._lm_processor_class = None
        self._pyctcdecode_version = None
        self._kenlm_version = None
        self._language_model_path = None
        self._language_model_size_bytes = None
        if not self.model_path.is_dir():
            raise ProfileError(
                f"Transformers model directory not found: {self.model_path}"
            )
        if self._is_mms_profile() and not profile.language:
            raise ProfileError("MMS profiles require a language adapter code")
        if self._uses_lm():
            self._prepare_lm_decoder_resources()

        self.device = device or _select_device()
        dtype_argument, self._dtype_fallback = _loader_dtype(
            profile.loader.torch_dtype,
            self.device,
        )

        try:
            self.processor = self._load_processor()
            if self._uses_lm():
                self._validate_lm_processor()
                self._configure_decoder_pool()
            self.model = AutoModelForCTC.from_pretrained(
                str(self.model_path),
                local_files_only=profile.loader.local_files_only,
                trust_remote_code=profile.loader.trust_remote_code,
                torch_dtype=dtype_argument,
            )
            self._configure_language_adapter()
            if self._uses_lm():
                self._validate_lm_vocabulary()
            self.model.to(self.device).eval()
        except Exception as exc:
            self.close()
            raise RuntimeError(
                f"Failed to load local Transformers CTC model from {self.model_path}: {exc}"
            ) from exc

        feature_extractor = getattr(self.processor, "feature_extractor", None)
        self.sampling_rate = int(getattr(feature_extractor, "sampling_rate", 16000))
        if self.sampling_rate <= 0:
            self.close()
            raise RuntimeError(
                f"Model processor reported an invalid sampling rate: {self.sampling_rate}"
            )
        self._model_class = type(self.model).__name__
        self._processor_class = type(self.processor).__name__
        if self._uses_lm():
            config = self._lm_config()
            self._decoder_call = {
                "api": "Wav2Vec2ProcessorWithLM.batch_decode",
                "arguments": {
                    "beam_width": config.beam_width,
                    "alpha": config.alpha,
                    "beta": config.beta,
                    "unk_score_offset": config.unk_score_offset,
                    "lm_score_boundary": config.lm_score_boundary,
                    "n_best": config.n_best,
                    "pool": "persistent_process_pool" if self._decoder_pool else None,
                    "num_processes": None if self._decoder_pool else 1,
                },
            }
        else:
            self._decoder_call = {
                "api": "torch.argmax + processor.batch_decode",
                "arguments": {"argmax_dim": -1},
            }
        self._effective_dtype = str(_model_dtype(self.model)).removeprefix("torch.")

    def _load_processor(self) -> Any:
        common_kwargs = {
            "local_files_only": self.profile.loader.local_files_only,
            "trust_remote_code": self.profile.loader.trust_remote_code,
        }
        if self.profile.loader.processor_mode == "auto":
            return AutoProcessor.from_pretrained(str(self.model_path), **common_kwargs)
        if self.profile.loader.processor_mode == "wav2vec2_with_lm":
            assert self._lm_processor_class is not None
            return self._lm_processor_class.from_pretrained(
                str(self.model_path),
                **common_kwargs,
            )

        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            str(self.model_path),
            local_files_only=self.profile.loader.local_files_only,
        )
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
            str(self.model_path),
            local_files_only=self.profile.loader.local_files_only,
        )
        return Wav2Vec2Processor(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
        )

    def _uses_lm(self) -> bool:
        return self.profile.decoding.strategy == "beam_search"

    def _lm_config(self) -> CTCLMConfig:
        config = self.profile.decoding.ctc_lm_kwargs
        if config is None:
            raise ProfileError("CTC beam_search decoding requires ctc_lm_kwargs")
        return config

    def _prepare_lm_decoder_resources(self) -> None:
        missing = [
            relative
            for relative in _LM_REQUIRED_FILES
            if not (self.model_path / relative).is_file()
        ]
        if missing:
            raise RuntimeError(
                "BookBot CTC LM decoder files are missing from the local snapshot: "
                + ", ".join(missing)
            )

        try:
            pyctcdecode_module = importlib.import_module("pyctcdecode")
            kenlm_module = importlib.import_module("kenlm")
        except ModuleNotFoundError as exc:
            package = exc.name or "pyctcdecode/kenlm"
            raise RuntimeError(
                "CTC beam_search decoding requires pyctcdecode==0.5.0 and "
                f"kenlm==0.3.0; missing package: {package}"
            ) from exc

        transformers_module = importlib.import_module("transformers")
        processor_class = getattr(
            transformers_module,
            "Wav2Vec2ProcessorWithLM",
            None,
        )
        if processor_class is None:
            raise RuntimeError(
                "The installed Transformers package does not provide "
                "Wav2Vec2ProcessorWithLM"
            )

        self._lm_processor_class = processor_class
        self._pyctcdecode_version = _distribution_version(
            "pyctcdecode", pyctcdecode_module
        )
        self._kenlm_version = _distribution_version("kenlm", kenlm_module)
        self._language_model_path = self.model_path / "language_model/5gram_mms.bin"
        self._language_model_size_bytes = self._language_model_path.stat().st_size

    def _validate_lm_processor(self) -> None:
        if self._lm_processor_class is None or not isinstance(
            self.processor, self._lm_processor_class
        ):
            raise RuntimeError(
                "BookBot beam_search profile did not load an LM-aware "
                "Wav2Vec2ProcessorWithLM processor"
            )
        decoder = getattr(self.processor, "decoder", None)
        labels = getattr(getattr(decoder, "_alphabet", None), "labels", None)
        if decoder is None or labels is None:
            raise RuntimeError(
                "The LM-aware processor does not expose a pyctcdecode decoder alphabet"
            )

    def _configure_decoder_pool(self) -> None:
        workers = self._lm_config().decoder_workers
        if workers > 0:
            self._decoder_pool = _create_decoder_pool(workers)

    def _normalized_acoustic_labels(self) -> list[str]:
        from pyctcdecode.alphabet import BLANK_TOKEN_PTN, UNK_TOKEN, UNK_TOKEN_PTN

        tokenizer = getattr(self.processor, "tokenizer", None)
        get_vocab = getattr(tokenizer, "get_vocab", None)
        if not callable(get_vocab):
            raise RuntimeError("The LM-aware processor tokenizer has no vocabulary")
        vocab = get_vocab()
        if not isinstance(vocab, dict) or any(
            type(index) is not int for index in vocab.values()
        ):
            raise RuntimeError("The LM-aware processor tokenizer vocabulary is invalid")
        ordered = sorted(vocab.items(), key=lambda item: item[1])
        if [index for _, index in ordered] != list(range(len(ordered))):
            raise RuntimeError(
                "The acoustic tokenizer IDs are not contiguous from zero"
            )

        labels: list[str] = []
        word_delimiter = getattr(tokenizer, "word_delimiter_token", None)
        for token, _ in ordered:
            normalized = token
            if BLANK_TOKEN_PTN.match(token):
                normalized = ""
            if token == word_delimiter:
                normalized = " "
            if UNK_TOKEN_PTN.match(token):
                normalized = UNK_TOKEN
            labels.append(normalized)
        return labels

    def _validate_lm_vocabulary(self) -> None:
        acoustic_labels = self._normalized_acoustic_labels()
        decoder_labels = list(self.processor.decoder._alphabet.labels)
        model_vocab_size = getattr(
            getattr(self.model, "config", None),
            "vocab_size",
            None,
        )
        if type(model_vocab_size) is not int:
            raise RuntimeError("The acoustic model does not declare config.vocab_size")
        if model_vocab_size != len(acoustic_labels):
            raise RuntimeError(
                "Acoustic model/tokenizer vocabulary length mismatch: "
                f"model={model_vocab_size}, tokenizer={len(acoustic_labels)}"
            )
        if acoustic_labels != decoder_labels:
            mismatch = next(
                (
                    index
                    for index, (acoustic, decoder) in enumerate(
                        zip(acoustic_labels, decoder_labels)
                    )
                    if acoustic != decoder
                ),
                min(len(acoustic_labels), len(decoder_labels)),
            )
            raise RuntimeError(
                "Acoustic tokenizer and decoder alphabet must match in length and "
                f"order (acoustic={len(acoustic_labels)}, decoder={len(decoder_labels)}, "
                f"first mismatch index={mismatch})"
            )

    def _is_mms_profile(self) -> bool:
        identity = (
            f"{self.profile.inference_setup_id}/{self.profile.artifact}"
        ).lower()
        return "mms" in identity

    def _configure_language_adapter(self) -> None:
        if not self._is_mms_profile():
            return
        assert self.profile.language is not None

        tokenizer = getattr(self.processor, "tokenizer", None)
        set_target_lang = getattr(tokenizer, "set_target_lang", None)
        load_adapter = getattr(self.model, "load_adapter", None)
        if not callable(set_target_lang) or not callable(load_adapter):
            raise RuntimeError(
                "MMS checkpoint does not expose tokenizer.set_target_lang/model.load_adapter"
            )
        set_target_lang(self.profile.language)
        load_adapter(
            self.profile.language,
            local_files_only=self.profile.loader.local_files_only,
        )

    def _decode(self, audio_batch: list[np.ndarray]) -> list[str]:
        encoded = self.processor(
            audio_batch,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        model_inputs = _move_model_inputs(
            encoded,
            device=self.device,
            floating_dtype=_model_dtype(self.model),
        )
        with torch.inference_mode():
            logits = self.model(**model_inputs).logits
        if self._uses_lm():
            return self._decode_with_lm(logits)
        return self._decode_greedy(logits)

    def _decode_greedy(self, logits: Any) -> list[str]:
        predicted_ids = torch.argmax(logits, dim=-1)
        decoded_texts = self.processor.batch_decode(predicted_ids)
        if isinstance(decoded_texts, str):
            decoded_texts = [decoded_texts]
        return [str(text).strip() for text in decoded_texts]

    def _decode_with_lm(self, logits: Any) -> list[str]:
        config = self._lm_config()
        logits_numpy = logits.detach().to(device="cpu", dtype=torch.float32).numpy()
        decode_kwargs: dict[str, Any] = {
            "beam_width": config.beam_width,
            "alpha": config.alpha,
            "beta": config.beta,
            "unk_score_offset": config.unk_score_offset,
            "lm_score_boundary": config.lm_score_boundary,
            "n_best": config.n_best,
        }
        if self._decoder_pool is not None:
            decode_kwargs["pool"] = self._decoder_pool
        else:
            # Wav2Vec2ProcessorWithLM otherwise creates a full CPU-count pool for
            # every call on Linux. A single process keeps decoder_workers=0 bounded.
            decode_kwargs["num_processes"] = 1
        decoded = self.processor.batch_decode(logits_numpy, **decode_kwargs)
        decoded_texts = getattr(decoded, "text", None)
        if isinstance(decoded_texts, str):
            decoded_texts = [decoded_texts]
        if not isinstance(decoded_texts, (list, tuple)):
            raise RuntimeError("LM-aware processor returned no decoded text")

        top_hypotheses: list[str] = []
        for hypotheses in decoded_texts:
            if isinstance(hypotheses, (list, tuple)):
                if not hypotheses:
                    raise RuntimeError(
                        "LM-aware processor returned an empty hypothesis list"
                    )
                hypotheses = hypotheses[0]
            top_hypotheses.append(str(hypotheses).strip())
        return top_hypotheses

    def transcribe_batch(self, audio_paths: Sequence[str]) -> list[TranscriptionResult]:
        rows: list[TranscriptionResult | None] = [None] * len(audio_paths)
        valid_audio: list[np.ndarray] = []
        valid_indices: list[int] = []
        durations: dict[int, float] = {}

        for index, raw_path in enumerate(audio_paths):
            audio_path = str(raw_path)
            try:
                audio, _, duration, _ = load_audio_and_resample(
                    audio_path,
                    self.sampling_rate,
                )
            except Exception as exc:
                rows[index] = TranscriptionResult(
                    audio_filepath=audio_path,
                    duration=0.0,
                    pred_text="",
                    error=f"failed_to_read_audio: {exc}",
                )
                continue
            valid_audio.append(audio)
            valid_indices.append(index)
            durations[index] = duration

        if valid_audio:
            try:
                predictions = self._decode(valid_audio)
                if len(predictions) != len(valid_indices):
                    raise RuntimeError(
                        "Transformers CTC inference returned an unexpected number of "
                        f"predictions: expected {len(valid_indices)}, got {len(predictions)}"
                    )
            except Exception as exc:
                for index in valid_indices:
                    rows[index] = TranscriptionResult(
                        audio_filepath=str(audio_paths[index]),
                        duration=durations[index],
                        pred_text="",
                        error=f"inference_failed: {exc}",
                    )
            else:
                for index, prediction in zip(valid_indices, predictions):
                    rows[index] = TranscriptionResult(
                        audio_filepath=str(audio_paths[index]),
                        duration=durations[index],
                        pred_text=prediction,
                    )

        if any(row is None for row in rows):
            raise RuntimeError(
                "Transformers CTC backend failed to produce every result row"
            )
        return [row for row in rows if row is not None]

    def metadata(self) -> dict[str, Any]:
        metadata = {
            "inference_library": "transformers",
            "adapter": "ctc",
            "strategy": self.profile.decoding.strategy,
            "decoder_call": self._decoder_call,
            "model_class": self._model_class,
            "processor_class": self._processor_class,
            "device": str(self.device),
            "sampling_rate": self.sampling_rate,
            "processor_mode": self.profile.loader.processor_mode,
            "requested_torch_dtype": self.profile.loader.torch_dtype,
            "effective_torch_dtype": self._effective_dtype,
            "dtype_fallback": self._dtype_fallback,
            "language_adapter": self.profile.language
            if self._is_mms_profile()
            else None,
        }
        if self._uses_lm():
            config = self._lm_config()
            metadata.update(
                {
                    "decoder_class": type(self.processor.decoder).__name__,
                    "language_model_path": str(self._language_model_path),
                    "language_model_size_bytes": self._language_model_size_bytes,
                    "beam_width": config.beam_width,
                    "alpha": config.alpha,
                    "beta": config.beta,
                    "unk_score_offset": config.unk_score_offset,
                    "lm_score_boundary": config.lm_score_boundary,
                    "n_best": config.n_best,
                    "decoder_workers": config.decoder_workers,
                    "pyctcdecode_version": self._pyctcdecode_version,
                    "kenlm_version": self._kenlm_version,
                }
            )
        return metadata

    def close(self) -> None:
        decoder_pool = getattr(self, "_decoder_pool", None)
        self._decoder_pool = None
        if decoder_pool is not None:
            decoder_pool.close()
            decoder_pool.join()
        self.model = None
        self.processor = None
        if getattr(self, "device", None) is not None and self.device.type == "cuda":
            torch.cuda.empty_cache()
