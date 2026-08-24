"""NeMo implementation of the shared offline ASR backend contract."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf

from inference.common import load_audio_and_resample
from inference.contracts import TranscriptionResult
from inference.profile import ModelProfile
from inference.provenance import json_safe

try:  # Keep CLI help and profile validation usable outside the NeMo image.
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised only in slim environments
    torch = None  # type: ignore[assignment]

try:  # NeMo is intentionally installed only in the inference image.
    from nemo.collections.asr.models import (
        ASRModel,
        EncDecCTCModel,
        EncDecHybridRNNTCTCModel,
        EncDecRNNTModel,
    )
    from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig
except ModuleNotFoundError:  # pragma: no cover - exercised only in slim environments
    ASRModel = None  # type: ignore[assignment]
    EncDecCTCModel = None  # type: ignore[assignment]
    EncDecHybridRNNTCTCModel = None  # type: ignore[assignment]
    EncDecRNNTModel = None  # type: ignore[assignment]
    CTCDecodingConfig = None  # type: ignore[assignment]


TARGET_SR = 16000
DEFAULT_TMP = Path(__file__).resolve().parent / "tmp"


def require_nemo_runtime() -> None:
    """Fail with an actionable message when invoked outside the NeMo image."""
    if torch is None or ASRModel is None:
        raise RuntimeError(
            "NeMo inference requires torch and nemo_toolkit[asr]. "
            "Run it through the NeMo inference container."
        )


def write_wav(path: str, audio: np.ndarray, sr: int) -> None:
    """Write a prepared waveform, creating its parent directory if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr)


def configure_decoding_strategy(model: Any, decoder_type: str = "ctc") -> None:
    """Preserve the existing CTC/RNNT/hybrid decoder selection semantics."""
    require_nemo_runtime()

    if EncDecCTCModel is not None and isinstance(model, EncDecCTCModel):
        if decoder_type == "rnnt":
            raise SystemExit(
                "Loaded model is EncDecCTCModel (CTC-only). It cannot decode with RNNT."
            )
        print("[INFO] Loaded model is EncDecCTCModel. CTC decoding is already active.")
        return

    if EncDecHybridRNNTCTCModel is not None and isinstance(
        model, EncDecHybridRNNTCTCModel
    ):
        if CTCDecodingConfig is None:  # Defensive; covered by require_nemo_runtime().
            raise RuntimeError("NeMo CTC decoding support is unavailable")
        ctc_cfg = CTCDecodingConfig()
        model.change_decoding_strategy(ctc_cfg, decoder_type=decoder_type)
        cur_decoder = getattr(model, "cur_decoder", None)
        print(
            "[INFO] Loaded hybrid RNNT/CTC model. "
            f"Set decoder_type='{decoder_type}'. Current decoder: {cur_decoder}"
        )
        return

    if EncDecRNNTModel is not None and isinstance(model, EncDecRNNTModel):
        if decoder_type == "ctc":
            raise SystemExit(
                "Loaded model is EncDecRNNTModel (RNNT-only). "
                "It cannot be forced to decode with CTC."
            )
        print(
            "[INFO] Loaded model is EncDecRNNTModel. RNNT decoding is already active."
        )
        return

    print(
        f"[WARN] Unknown ASR model class: {type(model)}. "
        "Decoding strategy was not changed."
    )


def build_override_cfg(model: Any, batch_size: int, num_workers: int | None):
    """Build the same NeMo transcribe override configuration as the legacy CLI."""
    override_cfg = model.get_transcribe_config()
    override_cfg.batch_size = batch_size
    override_cfg.num_workers = 0 if num_workers is None else num_workers
    override_cfg.return_hypotheses = False
    override_cfg.channel_selector = None
    override_cfg.augmentor = None
    override_cfg.text_field = "text"
    override_cfg.lang_field = "lang"
    override_cfg.timestamps = None
    return override_cfg


def transcribe_batches(model: Any, files: list[str], override_cfg: Any) -> list[str]:
    """Transcribe files in NeMo-sized batches and normalize hypothesis objects."""
    require_nemo_runtime()
    assert torch is not None

    hypotheses: list[str] = []
    model.eval()

    batch_size = int(override_cfg.batch_size)
    with torch.no_grad():
        for start in range(0, len(files), batch_size):
            batch = files[start : start + batch_size]
            predictions = model.transcribe(
                audio=batch,
                override_config=override_cfg,
            )

            if predictions and hasattr(predictions[0], "text"):
                predictions = [prediction.text for prediction in predictions]

            hypotheses.extend(predictions)

    return hypotheses


def configure_runtime(cpu_workers: int) -> tuple[str, int]:
    """Select the legacy CUDA/CPU runtime and thread settings."""
    require_nemo_runtime()
    assert torch is not None

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        print("[INFO] GPU detected. Using CUDA for inference.")
        return device, 0

    workers = cpu_workers
    print(f"[INFO] GPU not available. Using CPU with torch threads={workers}.")
    try:
        torch.set_num_threads(max(1, workers))
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(min(4, max(1, workers)))
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, workers)))
    os.environ.setdefault("MKL_NUM_THREADS", str(max(1, workers)))
    return device, workers


class NemoBackend:
    """Profile-driven NeMo backend producing one result for every input path."""

    def __init__(
        self,
        *,
        profile: ModelProfile,
        model_path: str | Path,
        batch_size: int,
        cpu_workers: int = 0,
        tmp_dir: str | Path = DEFAULT_TMP,
        debug: bool = False,
    ) -> None:
        if profile.framework != "nemo" or profile.adapter != "nemo":
            raise ValueError(
                "NemoBackend requires a profile with framework=nemo and adapter=nemo"
            )
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        require_nemo_runtime()
        assert ASRModel is not None

        self.profile = profile
        self.model_path = Path(model_path)
        self.tmp_dir = Path(tmp_dir)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.debug = debug
        self.device, num_workers = configure_runtime(cpu_workers)
        self.num_workers = num_workers

        self.model = ASRModel.restore_from(
            str(self.model_path), map_location=self.device
        )
        self.model.to(self.device).eval()
        configure_decoding_strategy(
            self.model,
            decoder_type=profile.decoding.strategy,
        )
        self.override_cfg = build_override_cfg(
            model=self.model,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self._model_class = type(self.model).__name__
        self._effective_decoder_type = str(
            getattr(self.model, "cur_decoder", profile.decoding.strategy)
        )
        self._effective_decoding_config = json_safe(
            getattr(getattr(self.model, "cfg", None), "decoding", None)
        )
        self._transcribe_override_config = json_safe(self.override_cfg)

        print("[INFO] Using override_cfg:")
        print(f"       batch_size={self.override_cfg.batch_size}")
        print(f"       num_workers={self.override_cfg.num_workers}")
        print(f"       return_hypotheses={self.override_cfg.return_hypotheses}")
        print(f"       channel_selector={self.override_cfg.channel_selector}")
        print(f"       augmentor={self.override_cfg.augmentor}")
        print(f"       text_field={self.override_cfg.text_field}")
        print(f"       lang_field={self.override_cfg.lang_field}")
        print(f"       timestamps={self.override_cfg.timestamps}")

    def _prepared_path(self, audio_path: str) -> Path:
        digest = hashlib.sha256(audio_path.encode("utf-8")).hexdigest()[:12]
        return self.tmp_dir / f"{Path(audio_path).stem}_{digest}_full16k.wav"

    def transcribe_batch(
        self,
        audio_paths: Sequence[str],
    ) -> list[TranscriptionResult]:
        if self.model is None:
            raise RuntimeError("NemoBackend is closed")

        results: list[TranscriptionResult | None] = [None] * len(audio_paths)
        valid: list[tuple[int, str, str, float]] = []
        cleanup_paths: list[Path] = []

        for index, raw_path in enumerate(audio_paths):
            audio_path = str(raw_path)
            try:
                audio, sample_rate, duration, was_resampled = load_audio_and_resample(
                    audio_path,
                    TARGET_SR,
                )
            except Exception as exc:
                results[index] = TranscriptionResult(
                    audio_filepath=audio_path,
                    duration=0.0,
                    pred_text="",
                    error=f"failed_to_read_audio: {exc}",
                )
                continue

            inference_path = audio_path
            if was_resampled:
                prepared_path = self._prepared_path(audio_path)
                try:
                    write_wav(str(prepared_path), audio, sample_rate)
                except Exception as exc:
                    results[index] = TranscriptionResult(
                        audio_filepath=audio_path,
                        duration=duration,
                        pred_text="",
                        error=f"failed_to_prepare_audio: {exc}",
                    )
                    continue
                inference_path = str(prepared_path)
                cleanup_paths.append(prepared_path)

            valid.append((index, audio_path, inference_path, duration))

        if valid:
            try:
                predictions = transcribe_batches(
                    model=self.model,
                    files=[item[2] for item in valid],
                    override_cfg=self.override_cfg,
                )
                if len(predictions) != len(valid):
                    raise RuntimeError(
                        "NeMo returned an unexpected number of predictions: "
                        f"expected {len(valid)}, got {len(predictions)}"
                    )
            except Exception as exc:
                for index, audio_path, _inference_path, duration in valid:
                    results[index] = TranscriptionResult(
                        audio_filepath=audio_path,
                        duration=duration,
                        pred_text="",
                        error=f"inference_failed: {exc}",
                    )
            else:
                for (index, audio_path, _inference_path, duration), prediction in zip(
                    valid,
                    predictions,
                ):
                    results[index] = TranscriptionResult(
                        audio_filepath=audio_path,
                        duration=duration,
                        pred_text=str(prediction or "").strip(),
                    )
            finally:
                if not self.debug:
                    for tmp_path in cleanup_paths:
                        try:
                            tmp_path.unlink(missing_ok=True)
                        except Exception:
                            pass

        return [
            result
            if result is not None
            else TranscriptionResult(
                audio_filepath=str(audio_paths[index]),
                duration=0.0,
                pred_text="",
                error="inference_failed: backend produced no result",
            )
            for index, result in enumerate(results)
        ]

    def metadata(self) -> dict[str, Any]:
        return {
            "framework": "nemo",
            "adapter": "nemo",
            "model_class": self._model_class,
            "device": self.device,
            "decoder_type": self.profile.decoding.strategy,
            "effective_decoder_type": self._effective_decoder_type,
            "effective_decoding_config": self._effective_decoding_config,
            "transcribe_override_config": self._transcribe_override_config,
            "target_sample_rate": TARGET_SR,
            "num_workers": self.num_workers,
            "tmp_dir": str(self.tmp_dir),
            "debug": self.debug,
        }

    def close(self) -> None:
        model = self.model
        self.model = None
        if model is not None:
            del model
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
