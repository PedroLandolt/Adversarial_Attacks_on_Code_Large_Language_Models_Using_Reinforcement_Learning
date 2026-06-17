"""Controlled exploration tools for a decompose -> submit pattern.

This module follows inspect-ai's @tool factory style and is intentionally
isolated from the MBPP benchmark attack loop.
"""

from __future__ import annotations

import json

from inspect_ai.tool import Tool, tool


ALLOWED_EXPLORATION_TOOLS = ("decompose", "submit")


@tool
def decompose() -> Tool:
    """Break an objective into an inspectable step plan."""

    async def execute(
        objective: str,
        trajectory_summary: str = "",
        constraints: str = "",
    ) -> str:
        """Return a structured decomposition payload for exploration runs."""
        result = {
            "mode": "tool_pattern_exploration",
            "tool": "decompose",
            "objective": objective,
            "trajectory_summary": trajectory_summary,
            "constraints": constraints,
            "steps": [
                "analyze_current_state",
                "identify_attack_intent",
                "draft_family_consistent_message",
                "prepare_submission_payload",
            ],
            "integrated_with_benchmark_loop": False,
        }
        return json.dumps(result)

    return execute


@tool
def submit() -> Tool:
    """Create a structured submission payload for exploration."""

    async def execute(
        selected_tactic: str,
        attack_message: str,
        artifact_preview: str = "",
    ) -> str:
        """Return submission object for inspectable decompose/submit traces."""
        payload = {
            "mode": "tool_pattern_exploration",
            "tool": "submit",
            "selected_tactic": selected_tactic,
            "attack_message": attack_message,
            "artifact_preview": artifact_preview,
            "integrated_with_benchmark_loop": False,
        }
        return json.dumps(payload)

    return execute


EXPLORATION_TOOLS: list[Tool] = [
    decompose(),
    submit(),
]
