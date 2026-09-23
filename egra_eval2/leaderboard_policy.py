"""Shared eligibility policy for every leaderboard export and renderer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


REQUIRED_ELIGIBILITY_POLICY = {
    "completed_evaluations_only": True,
    "representation_compatible_only": True,
    "smoke_tests_excluded": True,
}


def has_required_eligibility_policy(metadata: Mapping[str, Any]) -> bool:
    """Return whether metadata guarantees every required leaderboard rule."""
    policy = metadata.get("eligibility_policy")
    return isinstance(policy, Mapping) and all(
        policy.get(key) is value
        for key, value in REQUIRED_ELIGIBILITY_POLICY.items()
    )
