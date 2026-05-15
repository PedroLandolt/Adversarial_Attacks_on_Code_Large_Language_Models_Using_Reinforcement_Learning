"""Selector policy boundary for tactic-family decisions.

This module introduces a swappable selector interface without implementing RL.
The default concrete policy wraps the existing ReactTacticSelector.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from agent.react_selector import ReactTacticSelector
from agent.tactic_registry import TacticRegistryEntry, get_tactic_registry
from utils.reward_accounting import compute_attempt_reward, normalize_arm_id


def _save_bandit_weights(
    path: str, pull_counts: dict, cumulative_rewards: dict
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {"pull_counts": pull_counts, "cumulative_rewards": cumulative_rewards},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _load_bandit_weights(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, KeyError):
        return None


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
    bandit_algorithm: str | None = None
    bandit_state: dict | None = None


class SelectorPolicy(Protocol):
    """Swappable selector interface for future policy implementations."""

    async def select(self, context: SelectorContext) -> SelectorDecision:
        """Select next tactic family from allowed closed set."""


class ReactSelectorPolicy:
    """Adapter that exposes ReactTacticSelector via SelectorPolicy interface."""

    def __init__(
        self,
        selector_model: str | Any,
        environment: str = "benchmark",
        use_chain_of_thought: bool = True,
    ):
        self._selector = ReactTacticSelector(
            selector_model,
            environment=environment,
            use_chain_of_thought=use_chain_of_thought,
        )

    async def select(self, context: SelectorContext) -> SelectorDecision:
        selection = await self._selector.select_tactic_with_reasoning(
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
        chosen: TacticRegistryEntry = selection.tactic
        return SelectorDecision(
            tactic_id=chosen.tactic_id,
            tactic_family=chosen.tactic_family,
            environment_support=chosen.environment_support,
            renderer_binding=chosen.renderer_binding,
            taxonomy_category=chosen.taxonomy_category,
            selector_name="agent_based_decision",
            selector_reasoning=selection.selector_reasoning,
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
    """Lightweight benchmark-only bandit baseline over the tactic registry."""

    def __init__(
        self,
        environment: str = "benchmark",
        bandit_algorithm: str = "ucb1",
        weights_path: str | None = None,
        freeze_weights: bool = False,
    ):
        if environment != "benchmark":
            raise ValueError(
                "RLBanditSelectorPolicy currently supports benchmark mode only."
            )
        if bandit_algorithm != "ucb1":
            raise ValueError(
                "RLBanditSelectorPolicy currently supports only bandit_algorithm='ucb1'."
            )
        self._environment = environment
        self._bandit_algorithm = bandit_algorithm
        # Resolve to absolute path at init time so cwd changes inside Inspect
        # do not silently load from or save to a different location.
        self._weights_path = str(Path(weights_path).resolve()) if weights_path else None
        self._freeze_weights = freeze_weights
        self._actions = get_tactic_registry(environment)
        self._pull_counts = {entry.tactic_id: 0 for entry in self._actions}
        self._cumulative_rewards = {entry.tactic_id: 0.0 for entry in self._actions}

        if self._weights_path is not None:
            loaded = _load_bandit_weights(self._weights_path)
            if loaded is not None:
                for arm_id, count in loaded.get("pull_counts", {}).items():
                    if arm_id in self._pull_counts:
                        self._pull_counts[arm_id] = int(count)
                for arm_id, reward in loaded.get("cumulative_rewards", {}).items():
                    if arm_id in self._cumulative_rewards:
                        self._cumulative_rewards[arm_id] = float(reward)

    async def select(self, context: SelectorContext) -> SelectorDecision:
        chosen = self._select_ucb1_action()
        chosen_arm_id = normalize_arm_id(
            tactic_id=chosen.tactic_id,
            tactic_family=chosen.tactic_family,
        )
        return SelectorDecision(
            tactic_id=chosen.tactic_id,
            tactic_family=chosen.tactic_family,
            environment_support=chosen.environment_support,
            renderer_binding=chosen.renderer_binding,
            taxonomy_category=chosen.taxonomy_category,
            selector_name="rl_bandit",
            selector_reasoning=None,
            bandit_algorithm=self._bandit_algorithm,
            bandit_state={
                "algorithm": self._bandit_algorithm,
                "total_pulls": sum(self._pull_counts.values()),
                "selected_arm_id": chosen_arm_id,
                "pull_counts": dict(self._pull_counts),
                "cumulative_rewards": dict(self._cumulative_rewards),
                "selected_arm_average_reward": self._average_reward(chosen.tactic_id),
            },
        )

    def record_outcome(self, decision: SelectorDecision, attempt_record: dict) -> dict:
        arm_id = normalize_arm_id(
            tactic_id=decision.tactic_id,
            tactic_family=decision.tactic_family,
        )
        if arm_id is None or arm_id not in self._pull_counts:
            return {
                "arm_id": arm_id,
                "reward": None,
                "reward_rule": None,
            }

        reward_info = compute_attempt_reward(
            attempt_record,
            attempt_record.get("failure_stage"),
        )
        if not self._freeze_weights:
            self._pull_counts[arm_id] += 1
            self._cumulative_rewards[arm_id] += float(reward_info["reward"])
            if self._weights_path is not None:
                _save_bandit_weights(
                    self._weights_path, self._pull_counts, self._cumulative_rewards
                )
        return {
            "arm_id": arm_id,
            "reward": reward_info["reward"],
            "reward_rule": reward_info["reward_rule"],
        }

    def _select_ucb1_action(self) -> TacticRegistryEntry:
        unpulled_actions = [
            entry for entry in self._actions if self._pull_counts[entry.tactic_id] == 0
        ]
        if unpulled_actions:
            return unpulled_actions[0]

        total_pulls = sum(self._pull_counts.values())
        return max(
            self._actions,
            key=lambda entry: self._ucb1_score(entry.tactic_id, total_pulls),
        )

    def _average_reward(self, tactic_id: str) -> float:
        pulls = self._pull_counts[tactic_id]
        if pulls == 0:
            return 0.0
        return self._cumulative_rewards[tactic_id] / pulls

    def _ucb1_score(self, tactic_id: str, total_pulls: int) -> float:
        pulls = self._pull_counts[tactic_id]
        if pulls == 0:
            return float("inf")
        return self._average_reward(tactic_id) + math.sqrt(
            (2.0 * math.log(total_pulls)) / pulls
        )
