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


_SUPPORTED_ALGORITHMS = {"ucb1", "thompson", "klucb", "exp3"}


def _save_bandit_weights(path: str, state: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
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
    llm_judge_confidence: float | None
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
    """Multi-armed bandit over the tactic registry.

    Supported algorithms
    --------------------
    ucb1      — Hoeffding upper confidence bound (Auer et al. 2002).
                Optimism-based; deterministic selection.
    thompson  — Beta-Bernoulli Thompson Sampling (Thompson 1933).
                Bayesian posterior matching; stochastic selection.
    klucb     — KL-UCB (Garivier & Cappé 2011, arXiv 1102.2490).
                Tighter confidence bound via binary search on KL divergence.
    exp3      — Exponential weights (Auer et al. 2002).
                Adversarial / non-stationary rewards; stochastic selection.

    Reward normalization
    --------------------
    Rewards from compute_attempt_reward() live in {−1.0, −0.5, 0.0, +1.0}.
    Thompson and KL-UCB require values in [0, 1]; EXP3 uses [0, 1] for
    importance-weighted estimates. All three normalise via (r + 1) / 2 internally.
    UCB1 operates on raw rewards for backward compatibility with saved weights.

    Weight persistence
    ------------------
    The JSON file stores pull_counts and cumulative_rewards (all algorithms),
    plus thompson_alpha / thompson_beta (Thompson) or exp3_weights (EXP3).
    Algorithm-specific fields are written only for the relevant algorithm and
    silently ignored when loading weights for a different algorithm.
    """

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
        alg = bandit_algorithm.lower()
        if alg not in _SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported bandit_algorithm={bandit_algorithm!r}. "
                f"Supported: {sorted(_SUPPORTED_ALGORITHMS)}"
            )
        self._environment = environment
        self._bandit_algorithm = alg
        self._weights_path = str(Path(weights_path).resolve()) if weights_path else None
        self._freeze_weights = freeze_weights
        self._actions = get_tactic_registry(environment)
        arm_ids = [e.tactic_id for e in self._actions]

        # --- Common state (all algorithms) ---
        self._pull_counts: dict[str, int] = {a: 0 for a in arm_ids}
        self._cumulative_rewards: dict[str, float] = {a: 0.0 for a in arm_ids}

        # --- Thompson Sampling: Beta(α, β) per arm ---
        # α = successes + 1 (prior), β = failures + 1 (prior)
        self._thompson_alpha: dict[str, float] = {a: 1.0 for a in arm_ids}
        self._thompson_beta: dict[str, float] = {a: 1.0 for a in arm_ids}

        # --- EXP3: exponential weights per arm ---
        # γ controls exploration; η = γ/K is the per-step learning rate.
        # With K=9 arms and γ=0.1: η ≈ 0.011 — conservative but stable.
        K = len(arm_ids)
        self._exp3_weights: dict[str, float] = {a: 1.0 for a in arm_ids}
        self._exp3_gamma: float = min(0.5, math.sqrt(math.log(max(K, 2)) / (K * 100)))
        # Practical default: 0.1 unless formula suggests lower
        self._exp3_gamma = max(self._exp3_gamma, 0.05)

        # --- Load persisted weights ---
        if self._weights_path is not None:
            loaded = _load_bandit_weights(self._weights_path)
            if loaded is not None:
                for arm_id, count in loaded.get("pull_counts", {}).items():
                    if arm_id in self._pull_counts:
                        self._pull_counts[arm_id] = int(count)
                for arm_id, reward in loaded.get("cumulative_rewards", {}).items():
                    if arm_id in self._cumulative_rewards:
                        self._cumulative_rewards[arm_id] = float(reward)
                for arm_id, v in loaded.get("thompson_alpha", {}).items():
                    if arm_id in self._thompson_alpha:
                        self._thompson_alpha[arm_id] = float(v)
                for arm_id, v in loaded.get("thompson_beta", {}).items():
                    if arm_id in self._thompson_beta:
                        self._thompson_beta[arm_id] = float(v)
                for arm_id, v in loaded.get("exp3_weights", {}).items():
                    if arm_id in self._exp3_weights:
                        self._exp3_weights[arm_id] = float(v)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def select(self, context: SelectorContext) -> SelectorDecision:
        exp3_probs: dict[str, float] = {}

        if self._freeze_weights:
            chosen = self._select_greedy_action()
            selection_mode = "greedy"
        elif self._bandit_algorithm == "ucb1":
            chosen = self._select_ucb1_action()
            selection_mode = "ucb1"
        elif self._bandit_algorithm == "thompson":
            chosen = self._select_thompson_action()
            selection_mode = "thompson"
        elif self._bandit_algorithm == "klucb":
            chosen = self._select_klucb_action()
            selection_mode = "klucb"
        elif self._bandit_algorithm == "exp3":
            chosen, exp3_probs = self._select_exp3_action()
            selection_mode = "exp3"
        else:
            chosen = self._select_ucb1_action()
            selection_mode = "ucb1"

        chosen_arm_id = normalize_arm_id(
            tactic_id=chosen.tactic_id,
            tactic_family=chosen.tactic_family,
        )
        total_pulls = sum(self._pull_counts.values())
        bandit_state: dict[str, Any] = {
            "algorithm": self._bandit_algorithm,
            "selection_mode": selection_mode,
            "total_pulls": total_pulls,
            "selected_arm_id": chosen_arm_id,
            "pull_counts": dict(self._pull_counts),
            "cumulative_rewards": dict(self._cumulative_rewards),
            "selected_arm_average_reward": self._average_reward(chosen.tactic_id),
        }
        if exp3_probs:
            # Stored so record_outcome can do the importance-weighted update.
            bandit_state["exp3_selection_probs"] = exp3_probs

        return SelectorDecision(
            tactic_id=chosen.tactic_id,
            tactic_family=chosen.tactic_family,
            environment_support=chosen.environment_support,
            renderer_binding=chosen.renderer_binding,
            taxonomy_category=chosen.taxonomy_category,
            selector_name="rl_bandit",
            selector_reasoning=None,
            bandit_algorithm=self._bandit_algorithm,
            bandit_state=bandit_state,
        )

    def record_outcome(self, decision: SelectorDecision, attempt_record: dict) -> dict:
        arm_id = normalize_arm_id(
            tactic_id=decision.tactic_id,
            tactic_family=decision.tactic_family,
        )
        if arm_id is None or arm_id not in self._pull_counts:
            return {"arm_id": arm_id, "reward": None, "reward_rule": None}

        reward_info = compute_attempt_reward(
            attempt_record,
            attempt_record.get("failure_stage"),
        )
        reward = float(reward_info["reward"])

        if not self._freeze_weights:
            # --- Update common state ---
            self._pull_counts[arm_id] += 1
            self._cumulative_rewards[arm_id] += reward

            # --- Algorithm-specific updates ---
            if self._bandit_algorithm == "thompson":
                # Beta-Bernoulli: reward > 0 → success, else failure
                if reward > 0:
                    self._thompson_alpha[arm_id] += 1.0
                else:
                    self._thompson_beta[arm_id] += 1.0

            elif self._bandit_algorithm == "exp3":
                # Importance-weighted exponential update.
                # r_normalized ∈ [0, 1]; r̂ = r_normalized / p_chosen.
                probs = (decision.bandit_state or {}).get("exp3_selection_probs", {})
                p_chosen = probs.get(arm_id, 1.0 / max(len(self._actions), 1))
                r_normalized = (reward + 1.0) / 2.0
                r_hat = r_normalized / max(p_chosen, 1e-10)
                eta = self._exp3_gamma / max(len(self._actions), 1)
                self._exp3_weights[arm_id] *= math.exp(eta * r_hat)
                # Renormalise weights to prevent numerical overflow.
                max_w = max(self._exp3_weights.values())
                if max_w > 1e6:
                    for k in self._exp3_weights:
                        self._exp3_weights[k] /= max_w

            if self._weights_path is not None:
                _save_bandit_weights(self._weights_path, self._current_state())

        return {
            "arm_id": arm_id,
            "reward": reward,
            "reward_rule": reward_info["reward_rule"],
        }

    def arm_stats(self) -> dict[str, dict]:
        """Per-arm diagnostics including algorithm-specific index score."""
        total_pulls = sum(self._pull_counts.values())
        stats: dict[str, dict] = {}
        for entry in self._actions:
            arm_id = entry.tactic_id
            base: dict[str, Any] = {
                "pull_count": self._pull_counts[arm_id],
                "mean_reward": self._average_reward(arm_id),
            }
            if self._bandit_algorithm == "ucb1":
                base["algorithm_score"] = (
                    self._ucb1_score(arm_id, total_pulls)
                    if total_pulls > 0
                    else float("inf")
                )
                # Keep legacy key for any existing downstream consumers.
                base["ucb1_score"] = base["algorithm_score"]
            elif self._bandit_algorithm == "thompson":
                a = self._thompson_alpha[arm_id]
                b = self._thompson_beta[arm_id]
                base["algorithm_score"] = a / (a + b)
                base["posterior_mean"] = base["algorithm_score"]
                base["thompson_alpha"] = a
                base["thompson_beta"] = b
            elif self._bandit_algorithm == "klucb":
                base["algorithm_score"] = (
                    self._klucb_score(arm_id, total_pulls)
                    if total_pulls > 0
                    else float("inf")
                )
                base["klucb_score"] = base["algorithm_score"]
            elif self._bandit_algorithm == "exp3":
                W = sum(self._exp3_weights.values()) or 1.0
                K = len(self._actions)
                base["exp3_weight"] = self._exp3_weights[arm_id]
                base["exp3_probability"] = (
                    (1.0 - self._exp3_gamma) * self._exp3_weights[arm_id] / W
                    + self._exp3_gamma / K
                )
                base["algorithm_score"] = base["exp3_probability"]
            stats[arm_id] = base
        return stats

    # ------------------------------------------------------------------
    # Selection methods
    # ------------------------------------------------------------------

    def _select_greedy_action(self) -> TacticRegistryEntry:
        """Frozen-weight greedy selection (used during evaluation)."""
        unpulled = [e for e in self._actions if self._pull_counts[e.tactic_id] == 0]
        if unpulled:
            return random.choice(unpulled)
        if self._bandit_algorithm == "thompson":
            return max(
                self._actions,
                key=lambda e: (
                    self._thompson_alpha[e.tactic_id]
                    / (self._thompson_alpha[e.tactic_id] + self._thompson_beta[e.tactic_id])
                ),
            )
        if self._bandit_algorithm == "exp3":
            return max(
                self._actions,
                key=lambda e: self._exp3_weights.get(e.tactic_id, 1.0),
            )
        return max(self._actions, key=lambda e: self._average_reward(e.tactic_id))

    def _select_ucb1_action(self) -> TacticRegistryEntry:
        unpulled = [e for e in self._actions if self._pull_counts[e.tactic_id] == 0]
        if unpulled:
            return random.choice(unpulled)
        total_pulls = sum(self._pull_counts.values())
        return max(
            self._actions,
            key=lambda e: self._ucb1_score(e.tactic_id, total_pulls),
        )

    def _select_thompson_action(self) -> TacticRegistryEntry:
        """Beta-Bernoulli Thompson Sampling.

        Each arm maintains Beta(α, β) posteriors where α counts successes
        (reward > 0) and β counts failures, both starting from 1 (uniform prior).
        At selection time we sample one θ_i ~ Beta(α_i, β_i) per arm and
        pick the arm with the highest sample.

        Unpulled arms are prioritised first for consistency with UCB-style
        initialisation — their Beta(1, 1) = Uniform[0, 1] posterior would
        guarantee eventual exploration anyway, but this avoids the cold-start
        phase where all arms happen to sample low values.
        """
        unpulled = [e for e in self._actions if self._pull_counts[e.tactic_id] == 0]
        if unpulled:
            return random.choice(unpulled)
        samples = {
            e.tactic_id: random.betavariate(
                self._thompson_alpha[e.tactic_id],
                self._thompson_beta[e.tactic_id],
            )
            for e in self._actions
        }
        return max(self._actions, key=lambda e: samples[e.tactic_id])

    def _select_klucb_action(self) -> TacticRegistryEntry:
        """KL-UCB index policy (Garivier & Cappé 2011).

        The index for arm i at time t is:
            q*(i, t) = max { q ∈ [μ̂_i, 1] : n_i · kl(μ̂_i, q) ≤ log(t) }
        where kl(p, q) is the Bernoulli KL divergence and μ̂_i is the
        empirical mean of arm i normalised to [0, 1].

        Binary search over q converges in ~50 iterations to 1e-10 precision.
        """
        unpulled = [e for e in self._actions if self._pull_counts[e.tactic_id] == 0]
        if unpulled:
            return random.choice(unpulled)
        total_pulls = sum(self._pull_counts.values())
        return max(
            self._actions,
            key=lambda e: self._klucb_score(e.tactic_id, total_pulls),
        )

    def _select_exp3_action(self) -> tuple[TacticRegistryEntry, dict[str, float]]:
        """EXP3 mixed-strategy selection.

        The selection probability for arm i is:
            p_i = (1 − γ) · w_i / W  +  γ / K
        where W = Σ w_j, K = number of arms, and γ is the exploration parameter.

        Returns the chosen entry and the full probability dict (needed by
        record_outcome for the importance-weighted weight update).
        """
        K = len(self._actions)
        gamma = self._exp3_gamma
        W = sum(self._exp3_weights[e.tactic_id] for e in self._actions) or 1.0
        probs = {
            e.tactic_id: (1.0 - gamma) * self._exp3_weights[e.tactic_id] / W + gamma / K
            for e in self._actions
        }
        # Sample from the mixed distribution.
        r = random.random()
        cumulative = 0.0
        for entry in self._actions:
            cumulative += probs[entry.tactic_id]
            if r < cumulative:
                return entry, probs
        return self._actions[-1], probs  # numerical fallback

    # ------------------------------------------------------------------
    # Score helpers
    # ------------------------------------------------------------------

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

    def _kl_bernoulli(self, p: float, q: float) -> float:
        """KL(p ‖ q) for two Bernoulli distributions, clipped for numerical safety."""
        p = max(min(p, 1.0 - 1e-10), 1e-10)
        q = max(min(q, 1.0 - 1e-10), 1e-10)
        return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))

    def _klucb_score(self, tactic_id: str, total_pulls: int) -> float:
        """Binary search for the KL-UCB index q* ∈ [μ̂, 1]."""
        n = self._pull_counts[tactic_id]
        if n == 0:
            return float("inf")
        # Normalise empirical mean from raw [-1, 1] to [0, 1].
        mu_hat = (self._average_reward(tactic_id) + 1.0) / 2.0
        mu_hat = max(min(mu_hat, 1.0 - 1e-10), 1e-10)
        # Upper bound on KL divergence allowed: log(t) / n.
        target = math.log(max(total_pulls, 1)) / n
        lo, hi = mu_hat, 1.0 - 1e-10
        for _ in range(50):
            mid = (lo + hi) / 2.0
            if self._kl_bernoulli(mu_hat, mid) <= target:
                lo = mid
            else:
                hi = mid
        return lo

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _current_state(self) -> dict:
        """Build the full serialisable state dict for this algorithm."""
        state: dict[str, Any] = {
            "pull_counts": dict(self._pull_counts),
            "cumulative_rewards": dict(self._cumulative_rewards),
        }
        if self._bandit_algorithm == "thompson":
            state["thompson_alpha"] = dict(self._thompson_alpha)
            state["thompson_beta"] = dict(self._thompson_beta)
        elif self._bandit_algorithm == "exp3":
            state["exp3_weights"] = dict(self._exp3_weights)
        return state
