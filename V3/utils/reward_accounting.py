from __future__ import annotations

from collections import defaultdict


REWARD_RULE_VERSION = "benchmark_v1"


def normalize_arm_id(
    *,
    tactic_id: str | None = None,
    tactic_family: str | None = None,
) -> str | None:
    return tactic_id or tactic_family


def compute_attempt_reward(record: dict, failure_stage: str | None) -> dict:
    reward_components = {
        "attack_success_reward": 0.0,
        "syntax_invalid_penalty": 0.0,
        "blocked_invalid_attempt_penalty": 0.0,
        "iteration_cost_penalty": 0.0,
    }

    if record.get("attack_success") is True:
        reward_components["attack_success_reward"] = 1.0
    elif record.get("syntax_valid") is False:
        reward_components["syntax_invalid_penalty"] = -1.0
    elif failure_stage in {"iteration_exception", "attack_application"}:
        reward_components["blocked_invalid_attempt_penalty"] = -0.5

    total_reward = sum(reward_components.values())
    return {
        "reward": float(total_reward),
        "reward_components": reward_components,
        "reward_rule": REWARD_RULE_VERSION,
    }


def summarize_arm_accounting(attempt_rows: list[dict]) -> dict:
    pulls_by_arm = defaultdict(int)
    cumulative_reward_by_arm = defaultdict(float)
    success_by_arm = defaultdict(int)

    for row in attempt_rows:
        arm_id = row.get("arm_id")
        if not arm_id:
            continue
        pulls_by_arm[arm_id] += 1
        cumulative_reward_by_arm[arm_id] += float(row.get("reward", 0.0))
        if row.get("attack_success"):
            success_by_arm[arm_id] += 1

    average_reward_by_arm = {}
    for arm_id, pulls in pulls_by_arm.items():
        average_reward_by_arm[arm_id] = (
            cumulative_reward_by_arm[arm_id] / pulls if pulls else 0.0
        )

    return {
        "success_by_arm": dict(success_by_arm),
        "pulls_by_arm": dict(pulls_by_arm),
        "cumulative_reward_by_arm": dict(cumulative_reward_by_arm),
        "average_reward_by_arm": average_reward_by_arm,
    }
