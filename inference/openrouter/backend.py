"""OpenRouter ASR with fail-closed Zero Data Retention routing.

The main path is intentionally linear:

1. validate immutable ZDR policy in the profile and backend;
2. verify the exact model in OpenRouter's live ZDR and audio catalogues;
3. prepare one WAV without changing its bytes;
4. send it to the explicitly selected gateway with mandatory ZDR routing;
5. extract the transcript into the repository's shared result format.

``run_backend`` limits each batch, so ``transcribe_batch`` can safely send the
files in that batch concurrently while preserving their original order.
"""

from __future__ import annotations

import base64
import os
import threading
import time
import wave
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Final, Mapping, Sequence

import requests

from inference.contracts import TranscriptionResult
from inference.profile import InferenceProfile


# Privacy is fixed. Routing is an explicit, validated profile choice.
OPENROUTER_GLOBAL_API_BASE_URL: Final = "https://openrouter.ai/api/v1"
OPENROUTER_EU_API_BASE_URL: Final = "https://eu.openrouter.ai/api/v1"
OPENROUTER_API_BASE_URLS: Final = {
    "global": OPENROUTER_GLOBAL_API_BASE_URL,
    "eu": OPENROUTER_EU_API_BASE_URL,
}
OPENROUTER_ZDR_ENDPOINTS_URL: Final = (
    f"{OPENROUTER_GLOBAL_API_BASE_URL}/endpoints/zdr"
)
OPENROUTER_API_KEY_ENV: Final = "OPENROUTER_API_KEY"
REQUIRED_WORKSPACE_NAME: Final = "QA facility ASR"
REQUIRED_PROVIDER_POLICY: Final = {
    "zdr": True,
    "data_collection": "deny",
}

ENDPOINT_CHAT_AUDIO: Final = "chat_audio"
ENDPOINT_STT: Final = "stt"
ENDPOINT_PATHS: Final = {
    ENDPOINT_CHAT_AUDIO: "/chat/completions",
    ENDPOINT_STT: "/audio/transcriptions",
}

MAX_ATTEMPTS: Final = 5
REQUEST_TIMEOUT_SECONDS: Final = 180
PREFLIGHT_TIMEOUT_SECONDS: Final = 30
RETRYABLE_STATUS_CODES: Final = frozenset(
    {408, 409, 425, 429, 500, 502, 503, 504}
)
FATAL_STATUS_CODES: Final = frozenset({401, 402, 403, 404})


class OpenRouterFatalError(RuntimeError):
    """A run-level failure, such as missing credentials or no eligible route."""


class OpenRouterRequestError(RuntimeError):
    """A failure belonging to one audio file after safe retry handling."""


@dataclass(frozen=True)
class _PreparedWav:
    duration_seconds: float
    base64_data: str


@dataclass
class _RequestStatistics:
    """Thread-safe, non-secret observations for run metadata."""

    attempted_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    retries: int = 0
    models_returned: Counter[str] = field(default_factory=Counter)
    providers_returned: Counter[str] = field(default_factory=Counter)
    usage: Counter[str] = field(default_factory=Counter)
    responses_with_router_metadata: int = 0
    responses_without_router_metadata: int = 0
    router_metadata_samples: list[dict[str, Any]] = field(default_factory=list)
    _lock: Any = field(default_factory=threading.Lock, repr=False)

    def record_attempt(self) -> None:
        with self._lock:
            self.attempted_requests += 1

    def record_retry(self) -> None:
        with self._lock:
            self.retries += 1

    def record_failure(self) -> None:
        with self._lock:
            self.failed_requests += 1

    def record_success(self, response: Mapping[str, Any]) -> None:
        with self._lock:
            self.successful_requests += 1
            model = response.get("model")
            provider = response.get("provider")
            if isinstance(model, str):
                self.models_returned[model] += 1
            if isinstance(provider, str):
                self.providers_returned[provider] += 1

            usage = response.get("usage")
            if isinstance(usage, Mapping):
                for name in (
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "input_tokens",
                    "output_tokens",
                    "seconds",
                    "cost",
                ):
                    value = usage.get(name)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        self.usage[name] += value

            router_metadata = response.get("openrouter_metadata")
            if isinstance(router_metadata, Mapping):
                self.responses_with_router_metadata += 1
                if len(self.router_metadata_samples) < 10:
                    self.router_metadata_samples.append(
                        _sanitize_router_metadata(router_metadata)
                    )
            else:
                self.responses_without_router_metadata += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "attempted_requests": self.attempted_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "retries": self.retries,
                "models_returned": dict(sorted(self.models_returned.items())),
                "providers_returned": dict(
                    sorted(self.providers_returned.items())
                ),
                "usage": dict(sorted(self.usage.items())),
                "routing_observations": {
                    "responses_with_openrouter_metadata": (
                        self.responses_with_router_metadata
                    ),
                    "responses_without_openrouter_metadata": (
                        self.responses_without_router_metadata
                    ),
                    "openrouter_metadata_samples": list(
                        self.router_metadata_samples
                    ),
                    "sample_limit": 10,
                },
            }


def _sanitize_router_metadata(value: Any, *, depth: int = 0) -> Any:
    """Bound router diagnostics while excluding anything resembling a secret."""
    if depth >= 6:
        return "[DEPTH_LIMIT]"
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for raw_key, item in list(value.items())[:100]:
            key = str(raw_key)
            if any(word in key.lower() for word in ("authorization", "api_key", "token")):
                cleaned[key] = "[REDACTED]"
            else:
                cleaned[key] = _sanitize_router_metadata(item, depth=depth + 1)
        return cleaned
    if isinstance(value, (list, tuple)):
        return [
            _sanitize_router_metadata(item, depth=depth + 1)
            for item in list(value)[:100]
        ]
    if isinstance(value, str):
        return value[:500]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:500]


def _prepare_wav(audio_path: str) -> _PreparedWav:
    """Read the original WAV once and derive duration from its header."""
    raw_audio = Path(audio_path).read_bytes()
    with wave.open(BytesIO(raw_audio), "rb") as wav:
        sample_rate = wav.getframerate()
        if sample_rate <= 0:
            raise wave.Error("WAV sample rate must be positive")
        duration = float(wav.getnframes() / sample_rate)
    return _PreparedWav(
        duration_seconds=duration,
        base64_data=base64.b64encode(raw_audio).decode("ascii"),
    )


def _safe_response_detail(response: Any) -> str:
    """Return a short provider error without request headers or audio."""
    try:
        payload = response.json()
    except (TypeError, ValueError):
        payload = None

    if isinstance(payload, Mapping):
        error = payload.get("error")
        if isinstance(error, Mapping):
            detail = error.get("message") or error.get("code")
        else:
            detail = error or payload.get("message")
        if detail is not None:
            return str(detail).replace("\n", " ")[:300]
    return "OpenRouter returned no structured error detail"


class OpenRouterASRBackend:
    """Transcribe audio through an explicit gateway with mandatory ZDR."""

    def __init__(
        self,
        profile: InferenceProfile,
        *,
        api_key: str | None = None,
        get: Callable[..., Any] = requests.get,
        post: Callable[..., Any] = requests.post,
        sleep: Callable[[float], None] = time.sleep,
        max_attempts: int = MAX_ATTEMPTS,
        timeout_seconds: int = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self._validate_profile(profile)
        if max_attempts != MAX_ATTEMPTS:
            raise ValueError(f"OpenRouter retries are fixed at {MAX_ATTEMPTS} attempts")

        key = api_key if api_key is not None else os.environ.get(OPENROUTER_API_KEY_ENV)
        if not isinstance(key, str) or not key.strip():
            raise OpenRouterFatalError(
                f"{OPENROUTER_API_KEY_ENV} for workspace "
                f"'{REQUIRED_WORKSPACE_NAME}' is not available in this process; "
                "start a new terminal or Codex process after setting the user variable"
            )

        self.profile = profile
        self._api_key = key.strip()
        self._get = get
        self._post = post
        self._sleep = sleep
        self._max_attempts = max_attempts
        self._timeout_seconds = timeout_seconds
        self._statistics = _RequestStatistics()
        # This must happen during construction: run_backend has not inspected or
        # encoded any audio yet, so an inconclusive privacy check fails closed.
        self._zdr_preflight = self._preflight_zdr_eligibility()

    @staticmethod
    def _validate_profile(profile: InferenceProfile) -> None:
        """Defend the privacy boundary even if profile parsing is bypassed."""
        if profile.inference_library != "openrouter":
            raise ValueError("OpenRouterASRBackend requires an OpenRouter profile")
        if profile.model is None or profile.api is None or profile.request is None:
            raise ValueError("OpenRouter profile is missing model, api, or request")
        expected_base_url = OPENROUTER_API_BASE_URLS.get(
            profile.api.routing_region
        )
        if expected_base_url is None:
            raise ValueError("OpenRouter routing region must be global or eu")
        if profile.api.base_url != expected_base_url:
            raise ValueError(
                "OpenRouter base URL must exactly match the selected routing region"
            )
        if profile.api.endpoint_kind != profile.adapter:
            raise ValueError("OpenRouter endpoint kind must match the adapter")
        if profile.adapter not in ENDPOINT_PATHS:
            raise ValueError("OpenRouter profile has an unsupported endpoint kind")
        if profile.api.api_key_environment_variable != OPENROUTER_API_KEY_ENV:
            raise ValueError(
                f"OpenRouter credentials must come from {OPENROUTER_API_KEY_ENV}"
            )
        if profile.request.stream is not False:
            raise ValueError("OpenRouter chat requests must disable streaming")
        if not profile.request.provider.zdr:
            raise ValueError("OpenRouter requests must require ZDR")
        if profile.request.provider.data_collection != "deny":
            raise ValueError("OpenRouter requests must deny data collection")

    @property
    def endpoint_url(self) -> str:
        assert self.profile.api is not None
        return f"{self.profile.api.base_url}{ENDPOINT_PATHS[self.profile.adapter]}"

    def _preflight_zdr_eligibility(self) -> dict[str, Any]:
        """Require two current, exact catalogue matches before touching audio."""
        model = self.profile.model
        api = self.profile.api
        assert model is not None and api is not None
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
        }

        zdr_payload = self._preflight_get_json(
            OPENROUTER_ZDR_ENDPOINTS_URL,
            headers=headers,
            label="ZDR endpoint catalogue",
        )
        zdr_rows = zdr_payload.get("data")
        if not isinstance(zdr_rows, list):
            raise OpenRouterFatalError(
                "ZDR preflight failed closed: endpoint catalogue has no data list"
            )
        endpoint_matches = [
            row
            for row in zdr_rows
            if isinstance(row, Mapping) and row.get("model_id") == model.slug
        ]
        if not endpoint_matches:
            raise OpenRouterFatalError(
                "ZDR preflight failed closed: exact model is absent from "
                "OpenRouter's current ZDR endpoint catalogue"
            )

        model_catalogue_url = f"{api.base_url}/models"
        model_params = {"zdr": "true"}
        if self.profile.adapter == ENDPOINT_STT:
            model_params["output_modalities"] = "transcription"
        else:
            model_params["input_modalities"] = "audio"
        model_payload = self._preflight_get_json(
            model_catalogue_url,
            headers=headers,
            params=model_params,
            label=f"{api.routing_region} ZDR audio model catalogue",
        )
        model_rows = model_payload.get("data")
        if not isinstance(model_rows, list):
            raise OpenRouterFatalError(
                "ZDR preflight failed closed: model catalogue has no data list"
            )
        model_matches = [
            row
            for row in model_rows
            if isinstance(row, Mapping) and row.get("id") == model.slug
        ]
        if len(model_matches) != 1:
            raise OpenRouterFatalError(
                "ZDR preflight failed closed: exact model did not appear exactly "
                f"once in the {api.routing_region} ZDR audio catalogue"
            )

        architecture = model_matches[0].get("architecture")
        if not isinstance(architecture, Mapping):
            raise OpenRouterFatalError(
                "ZDR preflight failed closed: exact model has no architecture metadata"
            )
        input_modalities = architecture.get("input_modalities")
        output_modalities = architecture.get("output_modalities")
        if not isinstance(input_modalities, list) or "audio" not in input_modalities:
            raise OpenRouterFatalError(
                "ZDR preflight failed closed: exact model does not declare audio input"
            )
        if self.profile.adapter == ENDPOINT_STT and (
            not isinstance(output_modalities, list)
            or "transcription" not in output_modalities
        ):
            raise OpenRouterFatalError(
                "ZDR preflight failed closed: STT model does not declare "
                "transcription output"
            )

        endpoints = []
        for row in endpoint_matches:
            endpoints.append(
                {
                    key: row[key]
                    for key in ("provider_name", "tag", "status")
                    if key in row
                }
            )
        return {
            "status": "passed",
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "requested_model": model.slug,
            "routing_region": api.routing_region,
            "zdr_endpoint_catalogue": OPENROUTER_ZDR_ENDPOINTS_URL,
            "zdr_endpoint_exact_match_count": len(endpoint_matches),
            "zdr_endpoint_matches": endpoints,
            "model_catalogue": model_catalogue_url,
            "model_catalogue_query": dict(model_params),
            "model_catalogue_exact_match": True,
            "declared_input_modalities": list(input_modalities),
            "declared_output_modalities": (
                list(output_modalities)
                if isinstance(output_modalities, list)
                else []
            ),
            "performed_before_audio_access": True,
        }

    def _preflight_get_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        label: str,
        params: Mapping[str, str] | None = None,
    ) -> Mapping[str, Any]:
        try:
            response = self._get(
                url,
                headers=dict(headers),
                params=dict(params) if params is not None else None,
                timeout=PREFLIGHT_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            raise OpenRouterFatalError(
                f"ZDR preflight failed closed: {label} request failed "
                f"({type(exc).__name__})"
            ) from exc
        status = int(response.status_code)
        if not 200 <= status < 300:
            detail = _safe_response_detail(response).replace(
                self._api_key, "[REDACTED]"
            )
            raise OpenRouterFatalError(
                f"ZDR preflight failed closed: {label} returned HTTP "
                f"{status}: {detail}"
            )
        try:
            payload = response.json()
        except (TypeError, ValueError) as exc:
            raise OpenRouterFatalError(
                f"ZDR preflight failed closed: {label} returned invalid JSON"
            ) from exc
        if not isinstance(payload, Mapping):
            raise OpenRouterFatalError(
                f"ZDR preflight failed closed: {label} returned non-object JSON"
            )
        return payload

    def transcribe_batch(
        self, audio_paths: Sequence[str]
    ) -> list[TranscriptionResult]:
        """Transcribe one runner-bounded batch concurrently and in order."""
        if not audio_paths:
            return []
        with ThreadPoolExecutor(max_workers=len(audio_paths)) as executor:
            # executor.map deliberately returns values in input order.
            return list(executor.map(self._transcribe_one, audio_paths))

    def _transcribe_one(self, audio_path: str) -> TranscriptionResult:
        try:
            audio = _prepare_wav(audio_path)
        except (OSError, EOFError, wave.Error) as exc:
            return TranscriptionResult(
                audio_filepath=audio_path,
                duration=0.0,
                pred_text="",
                error=f"failed_to_read_wav: {type(exc).__name__}",
            )

        try:
            payload = self._build_payload(audio.base64_data)
            response = self._send_with_retries(payload)
            transcript = self._extract_transcript(response)
        except OpenRouterFatalError:
            raise
        except OpenRouterRequestError as exc:
            return TranscriptionResult(
                audio_filepath=audio_path,
                duration=audio.duration_seconds,
                pred_text="",
                error=str(exc),
            )

        return TranscriptionResult(
            audio_filepath=audio_path,
            duration=audio.duration_seconds,
            pred_text=transcript,
        )

    def build_payload(self, audio_path: str) -> dict[str, Any]:
        """Build a payload from a WAV path for inspection and contract tests."""
        return self._build_payload(_prepare_wav(audio_path).base64_data)

    def _build_payload(self, base64_audio: str) -> dict[str, Any]:
        model = self.profile.model
        request = self.profile.request
        assert model is not None and request is not None

        if self.profile.adapter == ENDPOINT_CHAT_AUDIO:
            payload: dict[str, Any] = {
                "model": model.slug,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.profile.prompt},
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": base64_audio,
                                    "format": "wav",
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": request.max_tokens,
                "stream": False,
                "provider": dict(REQUIRED_PROVIDER_POLICY),
            }
        elif self.profile.adapter == ENDPOINT_STT:
            payload = {
                "model": model.slug,
                "input_audio": {
                    "data": base64_audio,
                    "format": "wav",
                },
                "language": self.profile.language,
                "provider": dict(REQUIRED_PROVIDER_POLICY),
            }
        else:  # pragma: no cover - profile and backend validation reject this
            raise AssertionError(f"unsupported OpenRouter adapter: {self.profile.adapter}")
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.reasoning is not None:
            payload["reasoning"] = asdict(request.reasoning)
        return payload

    def _send_with_retries(self, payload: dict[str, Any]) -> Mapping[str, Any]:
        """Send one request, retrying only temporary failures on the same route."""
        for attempt in range(1, self._max_attempts + 1):
            self._statistics.record_attempt()
            try:
                headers = {
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                }
                if self.profile.adapter == ENDPOINT_CHAT_AUDIO:
                    headers["X-OpenRouter-Metadata"] = "enabled"
                response = self._post(
                    self.endpoint_url,
                    headers=headers,
                    json=payload,
                    timeout=self._timeout_seconds,
                )
            except requests.RequestException as exc:
                error = f"temporary network failure ({type(exc).__name__})"
                if attempt == self._max_attempts:
                    self._statistics.record_failure()
                    raise OpenRouterRequestError(error) from exc
                self._wait_before_retry(response=None, attempt=attempt)
                continue

            status = int(response.status_code)
            if 200 <= status < 300:
                result = self._read_json_response(response)
                self._statistics.record_success(result)
                return result

            detail = _safe_response_detail(response).replace(
                self._api_key, "[REDACTED]"
            )
            error = f"HTTP {status}: {detail}"
            if status in FATAL_STATUS_CODES:
                self._statistics.record_failure()
                raise OpenRouterFatalError(error)
            if status not in RETRYABLE_STATUS_CODES or attempt == self._max_attempts:
                self._statistics.record_failure()
                raise OpenRouterRequestError(error)
            self._wait_before_retry(response=response, attempt=attempt)

        raise AssertionError("unreachable")

    def _read_json_response(self, response: Any) -> Mapping[str, Any]:
        try:
            result = response.json()
        except (TypeError, ValueError) as exc:
            self._statistics.record_failure()
            raise OpenRouterRequestError("OpenRouter returned invalid JSON") from exc
        if not isinstance(result, Mapping):
            self._statistics.record_failure()
            raise OpenRouterRequestError(
                "OpenRouter returned a non-object JSON response"
            )
        return result

    def _wait_before_retry(self, *, response: Any | None, attempt: int) -> None:
        self._statistics.record_retry()
        self._sleep(self._retry_delay(response, attempt))

    @staticmethod
    def _retry_delay(response: Any | None, attempt: int) -> float:
        if response is not None:
            retry_after = getattr(response, "headers", {}).get("Retry-After")
            try:
                if retry_after is not None:
                    return min(max(float(retry_after), 0.0), 30.0)
            except (TypeError, ValueError):
                pass
        return float(min(2 ** (attempt - 1), 30))

    def _extract_transcript(self, response: Mapping[str, Any]) -> str:
        transcript = (
            response.get("text")
            if self.profile.adapter == ENDPOINT_STT
            else self._extract_chat_content(response)
        )
        if not isinstance(transcript, str):
            raise OpenRouterRequestError(
                "OpenRouter response did not contain transcription text"
            )
        return transcript

    @staticmethod
    def _extract_chat_content(response: Mapping[str, Any]) -> Any:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        first_choice = choices[0]
        if not isinstance(first_choice, Mapping):
            return None
        message = first_choice.get("message")
        return message.get("content") if isinstance(message, Mapping) else None

    def model_identity(self) -> dict[str, Any]:
        model = self.profile.model
        assert model is not None
        return {
            "kind": "remote_api_model",
            "artifact": model.slug,
            "requested_model": model.slug,
            "source": f"https://openrouter.ai/{model.slug}",
            "local_files": False,
            "routing_region": self.profile.api.routing_region,
            "zdr_preflight_status": self._zdr_preflight["status"],
        }

    def metadata(self) -> dict[str, Any]:
        model = self.profile.model
        assert model is not None
        return {
            "adapter": "openrouter_remote_asr",
            "device": "remote_openrouter_provider",
            "endpoint_kind": self.profile.adapter,
            "endpoint_url": self.endpoint_url,
            "requested_model": model.slug,
            "workspace_binding": {
                "required_workspace_name": REQUIRED_WORKSPACE_NAME,
                "mechanism": "workspace_scoped_api_key",
                "request_override_available": False,
                "verified_by_inference_endpoint": False,
            },
            "region_enforcement": {
                "gateway": self.profile.api.routing_region,
                "base_url": self.profile.api.base_url,
                "fail_closed": True,
                "in_region_provider_required": (
                    self.profile.api.routing_region == "eu"
                ),
                "cross_gateway_fallback": False,
            },
            "privacy_enforcement": {
                "request_level_mandatory": True,
                "provider": dict(REQUIRED_PROVIDER_POLICY),
                "live_preflight": dict(self._zdr_preflight),
                "failure_policy": "fail_closed_before_audio_access",
            },
            "authentication": {
                "scheme": "bearer",
                "source": "inherited_environment",
                "environment_variable": OPENROUTER_API_KEY_ENV,
                "secret_recorded": False,
            },
            "request": {
                "audio_format": "wav",
                "stream": False,
                "one_audio_per_request": True,
                "timeout_seconds": self._timeout_seconds,
                "maximum_attempts": self._max_attempts,
            },
            "statistics": self._statistics.snapshot(),
        }

    def close(self) -> None:
        self._api_key = ""
