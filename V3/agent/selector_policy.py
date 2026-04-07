"""Selector policy boundary for tactic-family decisions.

This module introduces a swappable selector interface without implementing RL.
The default concrete policy wraps the existing ReactTacticSelector.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Protocol
from typing import Any

from agent.react_selector import ReactTacticSelector
from agent.tactic_registry import TacticRegistryEntry, get_tactic_registry


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

    tactic_id: str
    tactic_family: str
    environment_support: tuple[str, ...]
    renderer_binding: str
    taxonomy_category: str | None
    selector_name: str
    selector_reasoning: str | None = None


class SelectorPolicy(Protocol):
    """Swappable selector interface for future policy implementations."""

    async def select(self, context: SelectorContext) -> SelectorDecision:
        """Select next tactic family from allowed closed set."""


class ReactSelectorPolicy:
    """Adapter that exposes ReactTacticSelector via SelectorPolicy interface."""

    def __init__(self, selector_model: str | Any, environment: str = "benchmark"):
        self._selector = ReactTacticSelector(selector_model, environment=environment)

    async def select(self, context: SelectorContext) -> SelectorDecision:
        chosen: TacticRegistryEntry = await self._selector.select_tactic(
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
            tactic_id=chosen.tactic_id,
            tactic_family=chosen.tactic_family,
            environment_support=chosen.environment_support,
            renderer_binding=chosen.renderer_binding,
            taxonomy_category=chosen.taxonomy_category,
            selector_name="agent_based_decision",
            selector_reasoning=None,
        )


class RandomSelectorPolicy:
    """Random baseline over the registry-backed benchmark action space."""

    def __init__(self, environment: str = "benchmark"):
        self._environment = environment
        self._actions = get_tactic_registry(environment)

    async def select(self, context: SelectorContext) -> SelectorDecision:
        chosen = random.choice(self._actions)
        return SelectorDecision(
            tactic_id=chosen.tactic_id,
            tactic_family=chosen.tactic_family,
            environment_support=chosen.environment_support,
            renderer_binding=chosen.renderer_binding,
            taxonomy_category=chosen.taxonomy_category,
            selector_name="random_choice",
            selector_reasoning=None,
        )


class RLBanditSelectorPolicy:
    """Placeholder policy-mode shell for later bandit implementation."""

    def __init__(self, environment: str = "benchmark"):
        self._environment = environment

    async def select(self, context: SelectorContext) -> SelectorDecision:
        raise NotImplementedError("policy_mode='rl_bandit' is reserved for Task 9.")
