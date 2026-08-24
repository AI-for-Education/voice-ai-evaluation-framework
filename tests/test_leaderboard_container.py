from __future__ import annotations

from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_leaderboard_service_is_isolated_from_model_images() -> None:
    compose = yaml.safe_load(
        (REPOSITORY_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    )
    service = compose["services"]["leaderboard"]

    assert service["image"] == "voice-ai-evaluation-framework-leaderboard:latest"
    assert service["build"]["dockerfile"] == "Dockerfile.streamlit"
    assert service["profiles"] == ["dashboard"]
    assert "gpus" not in service
    assert service["read_only"] is True
    assert service["volumes"] == [
        "./input_output_data/output:/work/input_output_data/output:ro"
    ]


def test_leaderboard_dockerfile_is_minimal_and_does_not_inherit_asr() -> None:
    dockerfile = (REPOSITORY_ROOT / "Dockerfile.streamlit").read_text(
        encoding="utf-8"
    )

    assert dockerfile.startswith("FROM python:3.11-slim\n")
    assert "voice-ai-evaluation-framework-asr" not in dockerfile
    assert "streamlit==1.51.0" in dockerfile
    assert "COPY egra_leaderboard.py /app/egra_leaderboard.py" in dockerfile
    assert (
        "COPY egra_eval2/leaderboard_context.py "
        "/app/egra_eval2/leaderboard_context.py"
    ) in dockerfile
    assert (
        "COPY egra_eval2/leaderboard_view.py /app/egra_eval2/leaderboard_view.py"
    ) in dockerfile
    assert (
        "COPY egra_eval2/model_presentation.json "
        "/app/egra_eval2/model_presentation.json"
    ) in dockerfile
    assert (
        "COPY inference/pipeline_provenance.py "
        "/app/inference/pipeline_provenance.py"
    ) in dockerfile
    assert "COPY input_output_data" not in dockerfile
    assert "egra_dashboard.py" not in dockerfile
