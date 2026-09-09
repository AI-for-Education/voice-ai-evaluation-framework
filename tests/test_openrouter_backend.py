from __future__ import annotations

import base64
import copy
import json
import threading
import time
import wave
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import requests

from inference.openrouter.backend import (
    OPENROUTER_EU_API_BASE_URL,
    OPENROUTER_GLOBAL_API_BASE_URL,
    OPENROUTER_ZDR_ENDPOINTS_URL,
    OpenRouterASRBackend,
    OpenRouterFatalError,
)
from inference.openrouter.infer import parse_args
from inference.profile import ProfileError, load_profile, parse_profile


REPOSITORY = Path(__file__).resolve().parents[1]
PROFILES = REPOSITORY / "inference" / "openrouter" / "profiles"


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}

    def json(self) -> dict[str, Any]:
        return self._payload


def _write_wav(path: Path, sample: int = 0) -> bytes:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(int(sample).to_bytes(2, "little", signed=True) * 80)
    return path.read_bytes()


def _profile(name: str):
    return load_profile(PROFILES / name)


def _eligible_get(profile, calls: list[dict[str, Any]] | None = None):
    assert profile.api is not None and profile.model is not None

    def get(url: str, **kwargs: Any) -> FakeResponse:
        if calls is not None:
            calls.append({"url": url, **kwargs})
        if url == OPENROUTER_ZDR_ENDPOINTS_URL:
            return FakeResponse(
                200,
                {
                    "data": [
                        {
                            "model_id": profile.model.slug,
                            "provider_name": "eligible-provider",
                            "tag": "eligible-route",
                            "status": 0,
                        }
                    ]
                },
            )
        assert url == f"{profile.api.base_url}/models"
        expected_params = {"zdr": "true"}
        if profile.adapter == "stt":
            expected_params["output_modalities"] = "transcription"
        else:
            expected_params["input_modalities"] = "audio"
        assert kwargs["params"] == expected_params
        output_modalities = (
            ["transcription"] if profile.adapter == "stt" else ["text"]
        )
        return FakeResponse(
            200,
            {
                "data": [
                    {
                        "id": profile.model.slug,
                        "architecture": {
                            "input_modalities": ["audio"],
                            "output_modalities": output_modalities,
                        },
                    }
                ]
            },
        )

    return get


def _backend(profile, **kwargs: Any) -> OpenRouterASRBackend:
    kwargs.setdefault("get", _eligible_get(profile))
    return OpenRouterASRBackend(profile, **kwargs)


@pytest.mark.parametrize(
    ("profile_name", "endpoint"),
    [
        ("google-gemini-3.1-flash-lite-sw.yaml", "/chat/completions"),
        ("google-gemini-3.5-flash-lite-sw.yaml", "/chat/completions"),
        ("google-gemini-3.6-flash-sw.yaml", "/chat/completions"),
        ("google-gemini-3.7-flash-sw.yaml", "/chat/completions"),
        ("google-gemini-3.8-flash-sw.yaml", "/chat/completions"),
        ("thinkingmachines-inkling-small-sw.yaml", "/chat/completions"),
        ("xiaomi-mimo-v2.5-sw.yaml", "/chat/completions"),
    ],
)
def test_every_request_uses_profile_gateway_and_exact_zdr_policy(
    tmp_path: Path, profile_name: str, endpoint: str
) -> None:
    audio = tmp_path / "sample.wav"
    raw_audio = _write_wav(audio)
    calls: list[dict[str, Any]] = []

    def post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append({"url": url, **kwargs})
        return FakeResponse(
            200,
            {
                "choices": [{"message": {"content": "habari"}}],
                "provider": "eligible/eu",
            },
        )

    profile = _profile(profile_name)
    backend = _backend(profile, api_key="secret", post=post)
    result = backend.transcribe_batch([str(audio)])

    assert result[0].pred_text == "habari"
    assert profile.api is not None
    assert calls[0]["url"] == f"{profile.api.base_url}{endpoint}"
    assert calls[0]["json"]["provider"] == {
        "zdr": True,
        "data_collection": "deny",
    }
    assert set(calls[0]["json"]["provider"]) == {"zdr", "data_collection"}
    assert calls[0]["headers"]["Authorization"] == "Bearer secret"
    assert calls[0]["headers"]["X-OpenRouter-Metadata"] == "enabled"
    audio_payload = calls[0]["json"]["messages"][0]["content"][1]["input_audio"]
    assert base64.b64decode(audio_payload["data"]) == raw_audio


def test_chat_settings_and_exact_gemma_prompt(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    _write_wav(audio)
    gemma_prompt = load_profile(
        REPOSITORY / "inference" / "multimodal" / "profiles" / "gemma-4-E2B-sw.yaml"
    ).prompt

    profile31 = _profile("google-gemini-3.1-flash-lite-sw.yaml")
    payload31 = _backend(profile31, api_key="x").build_payload(str(audio))
    assert profile31.prompt == gemma_prompt
    assert payload31["temperature"] == 0
    assert payload31["max_tokens"] == 512
    assert payload31["stream"] is False

    profile35 = _profile("google-gemini-3.5-flash-lite-sw.yaml")
    payload35 = _backend(profile35, api_key="x").build_payload(str(audio))
    assert profile35.prompt == gemma_prompt
    assert "temperature" not in payload35
    assert payload35["max_tokens"] == 512
    assert payload35["reasoning"] == {"effort": "minimal", "exclude": True}


@pytest.mark.parametrize(
    "profile_name",
    [
        "microsoft-mai-transcribe-2-sw.yaml",
        "google-chirp-3-sw.yaml",
        "fish-audio-transcribe-1-sw.yaml",
    ],
)
def test_stt_profiles_use_dedicated_endpoint_and_language_hint(
    tmp_path: Path, profile_name: str
) -> None:
    audio = tmp_path / "sample.wav"
    raw_audio = _write_wav(audio)
    calls: list[dict[str, Any]] = []

    def post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append({"url": url, **kwargs})
        return FakeResponse(200, {"text": "habari", "provider": "eligible/eu"})

    profile = _profile(profile_name)
    backend = _backend(
        profile, api_key="secret", post=post
    )
    result = backend.transcribe_batch([str(audio)])

    assert result[0].pred_text == "habari"
    assert profile.api is not None
    assert calls[0]["url"] == f"{profile.api.base_url}/audio/transcriptions"
    payload = calls[0]["json"]
    assert payload["provider"] == {"zdr": True, "data_collection": "deny"}
    assert payload["language"] == "sw"
    assert payload["temperature"] == 0
    assert base64.b64decode(payload["input_audio"]["data"]) == raw_audio
    assert payload["input_audio"]["format"] == "wav"
    assert "messages" not in payload
    assert "max_tokens" not in payload
    assert "stream" not in payload
    assert "X-OpenRouter-Metadata" not in calls[0]["headers"]


def test_stt_profile_rejects_chat_only_request_settings() -> None:
    payload = copy.deepcopy(_profile("google-chirp-3-sw.yaml").to_dict())
    payload.pop("parameter_evidence", None)
    payload["request"]["max_tokens"] = 512

    with pytest.raises(ProfileError, match="stt requests cannot define"):
        parse_profile(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            ("api", "base_url", "https://openrouter.ai/api/v1"),
            "exactly match api.routing_region",
        ),
        (("request", "provider", "zdr", False), "zdr must be true"),
        (("request", "provider", "data_collection", "allow"), "must be deny"),
    ],
)
def test_profile_rejects_weaker_privacy_settings(
    mutation: tuple[str, ...], message: str
) -> None:
    payload = copy.deepcopy(
        _profile("google-gemini-3.1-flash-lite-sw.yaml").to_dict()
    )
    payload.pop("parameter_evidence", None)
    current = payload
    for key in mutation[:-2]:
        current = current[key]
    current[mutation[-2]] = mutation[-1]
    with pytest.raises(ProfileError, match=message):
        parse_profile(payload)


def test_profile_rejects_missing_zdr() -> None:
    payload = copy.deepcopy(
        _profile("google-gemini-3.1-flash-lite-sw.yaml").to_dict()
    )
    payload.pop("parameter_evidence", None)
    del payload["request"]["provider"]["zdr"]
    with pytest.raises(ProfileError, match="zdr must be true"):
        parse_profile(payload)


def test_profile_requires_explicit_matching_routing_region() -> None:
    payload = copy.deepcopy(
        _profile("google-gemini-3.6-flash-sw.yaml").to_dict()
    )
    payload.pop("parameter_evidence", None)
    del payload["api"]["routing_region"]
    with pytest.raises(ProfileError, match="missing required key.*routing_region"):
        parse_profile(payload)

    payload = copy.deepcopy(
        _profile("google-gemini-3.6-flash-sw.yaml").to_dict()
    )
    payload.pop("parameter_evidence", None)
    payload["api"]["routing_region"] = "eu"
    with pytest.raises(ProfileError, match="exactly match api.routing_region"):
        parse_profile(payload)


def test_eu_is_optional_but_global_route_is_exact() -> None:
    profile = _profile("google-gemini-3.6-flash-sw.yaml")
    assert profile.api is not None
    assert profile.api.routing_region == "global"
    assert profile.api.base_url == OPENROUTER_GLOBAL_API_BASE_URL
    backend = _backend(profile, api_key="x")
    assert backend.endpoint_url == (
        f"{OPENROUTER_GLOBAL_API_BASE_URL}/chat/completions"
    )


def test_live_preflight_uses_two_authenticated_exact_model_checks() -> None:
    profile = _profile("google-gemini-3.6-flash-sw.yaml")
    calls: list[dict[str, Any]] = []
    backend = _backend(
        profile,
        api_key="secret",
        get=_eligible_get(profile, calls),
    )

    assert [call["url"] for call in calls] == [
        OPENROUTER_ZDR_ENDPOINTS_URL,
        f"{OPENROUTER_GLOBAL_API_BASE_URL}/models",
    ]
    assert all(call["headers"]["Authorization"] == "Bearer secret" for call in calls)
    preflight = backend.metadata()["privacy_enforcement"]["live_preflight"]
    assert preflight["status"] == "passed"
    assert preflight["requested_model"] == "google/gemini-3.6-flash"
    assert preflight["routing_region"] == "global"
    assert preflight["zdr_endpoint_exact_match_count"] == 1
    assert preflight["performed_before_audio_access"] is True


def test_absent_zdr_model_fails_before_model_catalogue_or_audio_access() -> None:
    profile = _profile("google-gemini-3.6-flash-sw.yaml")
    get_calls = 0
    post_calls = 0

    def get(_url: str, **_kwargs: Any) -> FakeResponse:
        nonlocal get_calls
        get_calls += 1
        return FakeResponse(200, {"data": []})

    def post(_url: str, **_kwargs: Any) -> FakeResponse:
        nonlocal post_calls
        post_calls += 1
        return FakeResponse(200, {})

    with pytest.raises(OpenRouterFatalError, match="absent.*ZDR endpoint catalogue"):
        OpenRouterASRBackend(profile, api_key="x", get=get, post=post)
    assert get_calls == 1
    assert post_calls == 0


def test_zdr_model_without_audio_catalogue_match_fails_closed() -> None:
    profile = _profile("google-gemini-3.6-flash-sw.yaml")

    def get(url: str, **_kwargs: Any) -> FakeResponse:
        if url == OPENROUTER_ZDR_ENDPOINTS_URL:
            return FakeResponse(
                200,
                {"data": [{"model_id": "google/gemini-3.6-flash"}]},
            )
        return FakeResponse(200, {"data": []})

    with pytest.raises(OpenRouterFatalError, match="did not appear exactly once"):
        OpenRouterASRBackend(profile, api_key="x", get=get)


def test_backend_rechecks_privacy_after_profile_validation() -> None:
    profile = _profile("google-gemini-3.1-flash-lite-sw.yaml")
    assert profile.api is not None and profile.request is not None
    with pytest.raises(ValueError, match="selected routing region"):
        OpenRouterASRBackend(
            replace(profile, api=replace(profile.api, base_url="https://openrouter.ai/api/v1")),
            api_key="x",
        )
    with pytest.raises(ValueError, match="require ZDR"):
        OpenRouterASRBackend(
            replace(
                profile,
                request=replace(
                    profile.request,
                    provider=replace(profile.request.provider, zdr=False),
                ),
            ),
            api_key="x",
        )
    with pytest.raises(ValueError, match="disable streaming"):
        OpenRouterASRBackend(
            replace(profile, request=replace(profile.request, stream=True)),
            api_key="x",
        )
    with pytest.raises(ValueError, match="credentials must come from"):
        OpenRouterASRBackend(
            replace(
                profile,
                api=replace(
                    profile.api,
                    api_key_environment_variable="A_WEAKER_KEY_SOURCE",
                ),
            ),
            api_key="x",
        )


def test_cli_exposes_no_privacy_or_routing_override() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--inference_profile",
                "profile.yaml",
                "--root_audio_dir",
                "audio",
                "--base_url",
                "https://openrouter.ai/api/v1",
            ]
        )


def test_temporary_failure_retries_at_most_five_times(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    _write_wav(audio)
    calls: list[dict[str, Any]] = []
    delays: list[float] = []

    def post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append({"url": url, **kwargs})
        if len(calls) < 5:
            return FakeResponse(429, {"error": {"message": "busy"}}, {"Retry-After": "0"})
        return FakeResponse(200, {"choices": [{"message": {"content": "sawa"}}]})

    profile = _profile("google-gemini-3.1-flash-lite-sw.yaml")
    backend = _backend(
        profile,
        api_key="x",
        post=post,
        sleep=delays.append,
    )
    result = backend.transcribe_batch([str(audio)])

    assert result[0].pred_text == "sawa"
    assert len(calls) == 5
    assert delays == [0.0, 0.0, 0.0, 0.0]
    assert all(call["url"].startswith(OPENROUTER_EU_API_BASE_URL) for call in calls)
    assert all(call["json"]["provider"]["zdr"] is True for call in calls)
    assert backend.metadata()["statistics"]["retries"] == 4


def test_network_failures_are_redacted_and_limited_to_five(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    _write_wav(audio)
    calls = 0

    def post(_url: str, **_kwargs: Any) -> FakeResponse:
        nonlocal calls
        calls += 1
        raise requests.ConnectionError("do not expose caller state")

    profile = _profile("google-gemini-3.1-flash-lite-sw.yaml")
    backend = _backend(
        profile,
        api_key="highly-secret",
        post=post,
        sleep=lambda _delay: None,
    )
    result = backend.transcribe_batch([str(audio)])

    assert calls == 5
    assert result[0].error == "temporary network failure (ConnectionError)"
    assert "highly-secret" not in json.dumps(backend.metadata())
    assert backend.metadata()["workspace_binding"] == {
        "required_workspace_name": "QA facility ASR",
        "mechanism": "workspace_scoped_api_key",
        "request_override_available": False,
        "verified_by_inference_endpoint": False,
    }


def test_fatal_auth_error_redacts_key(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    _write_wav(audio)

    def post(_url: str, **_kwargs: Any) -> FakeResponse:
        return FakeResponse(401, {"error": {"message": "bad key secret-value"}})

    profile = _profile("google-gemini-3.1-flash-lite-sw.yaml")
    backend = _backend(
        profile,
        api_key="secret-value",
        post=post,
    )
    with pytest.raises(OpenRouterFatalError, match="REDACTED") as caught:
        backend.transcribe_batch([str(audio)])
    assert "secret-value" not in str(caught.value)


def test_no_eligible_zdr_endpoint_at_request_time_fails_without_retry(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    _write_wav(audio)
    calls = 0

    def post(_url: str, **_kwargs: Any) -> FakeResponse:
        nonlocal calls
        calls += 1
        return FakeResponse(
            404,
            {"error": {"message": "No endpoints found matching provider policy"}},
        )

    profile = _profile("google-gemini-3.1-flash-lite-sw.yaml")
    backend = _backend(
        profile,
        api_key="x",
        post=post,
        sleep=lambda _delay: pytest.fail("fatal routing failure must not retry"),
    )
    with pytest.raises(OpenRouterFatalError, match="No endpoints found"):
        backend.transcribe_batch([str(audio)])
    assert calls == 1


def test_batch_requests_are_concurrent_but_results_remain_ordered(tmp_path: Path) -> None:
    audio_paths = [tmp_path / f"{index}.wav" for index in range(3)]
    raw_to_text = {
        _write_wav(path, index + 1): f"text-{index}" for index, path in enumerate(audio_paths)
    }
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def post(_url: str, **kwargs: Any) -> FakeResponse:
        nonlocal active, maximum_active
        content = kwargs["json"]["messages"][0]["content"]
        raw = base64.b64decode(content[1]["input_audio"]["data"])
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return FakeResponse(200, {"choices": [{"message": {"content": raw_to_text[raw]}}]})

    profile = _profile("google-gemini-3.1-flash-lite-sw.yaml")
    backend = _backend(
        profile,
        api_key="x",
        post=post,
    )
    results = backend.transcribe_batch([str(path) for path in audio_paths])

    assert maximum_active > 1
    assert [result.audio_filepath for result in results] == [
        str(path) for path in audio_paths
    ]
    assert [result.pred_text for result in results] == [
        "text-0",
        "text-1",
        "text-2",
    ]


def test_response_routing_metadata_and_cost_are_sanitized_and_recorded(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "sample.wav"
    _write_wav(audio)
    profile = _profile("google-gemini-3.6-flash-sw.yaml")

    def post(_url: str, **_kwargs: Any) -> FakeResponse:
        return FakeResponse(
            200,
            {
                "choices": [{"message": {"content": "habari"}}],
                "model": "google/gemini-3.6-flash",
                "provider": "Google",
                "usage": {"total_tokens": 12, "cost": 0.00125},
                "openrouter_metadata": {
                    "strategy": "price",
                    "edge": {"region": "eu-west"},
                    "selected_endpoint": {"provider_name": "Google"},
                    "api_key": "must-not-be-recorded",
                },
            },
        )

    backend = _backend(profile, api_key="x", post=post)
    assert backend.transcribe_batch([str(audio)])[0].pred_text == "habari"
    metadata = backend.metadata()
    statistics = metadata["statistics"]
    assert statistics["providers_returned"] == {"Google": 1}
    assert statistics["usage"] == {"cost": 0.00125, "total_tokens": 12}
    observations = statistics["routing_observations"]
    assert observations["responses_with_openrouter_metadata"] == 1
    sample = observations["openrouter_metadata_samples"][0]
    assert sample["edge"]["region"] == "eu-west"
    assert sample["selected_endpoint"]["provider_name"] == "Google"
    assert sample["api_key"] == "[REDACTED]"
