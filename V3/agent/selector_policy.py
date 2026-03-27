"""Selector policy boundary for tactic-family decisions.

This module introduces a swappable selector interface without implementing RL.
The default concrete policy wraps the existing ReactTacticSelector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from typing import Any

from agent.react_selector import ReactTacticSelector


@dataclass(frozen=True)
class SelectorContext:
    """Structured context passed to selector policies."""

    problem: str
    current_code: str
    test_judge_decision: str
    llm_judge_decision: str
    llm_judge_confidence: float
    iteration: int
    max_iterations: int
    previous_attempts: list[dict]
    tool_decompose_output: str | None = None


@dataclass(frozen=True)
class SelectorDecision:
    """Structured selector output for tactic-family choice."""

    tactic_family: str
    selector_name: str
    reasoning: str | None = None


class SelectorPolicy(Protocol):
    """Swappable selector interface for future policy implementations."""

    async def select(self, context: SelectorContext) -> SelectorDecision:
        """Select next tactic family from allowed closed set."""


class ReactSelectorPolicy:
    """Adapter that exposes ReactTacticSelector via SelectorPolicy interface."""

    def __init__(self, selector_model: str | Any):
        self._selector = ReactTacticSelector(selector_model)

    async def select(self, context: SelectorContext) -> SelectorDecision:
        chosen = await self._selector.select_tactic(
            problem=context.problem,
            current_code=context.current_code,
            test_judge_decision=context.test_judge_decision,
            llm_judge_decision=context.llm_judge_decision,
            llm_judge_confidence=context.llm_judge_confidence,
            iteration=context.iteration,
            max_iterations=context.max_iterations,
            previous_attempts=context.previous_attempts,
            tool_decompose_output=context.tool_decompose_output,
        )
        return SelectorDecision(
            tactic_family=chosen.value,
            selector_name="react",
            reasoning=None,
        )
