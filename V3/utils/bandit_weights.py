from __future__ import annotations

import json
from pathlib import Path


def save_bandit_weights(path: str, pull_counts: dict, cumulative_rewards: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pull_counts": pull_counts,
        "cumulative_rewards": cumulative_rewards,
    }
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_bandit_weights(path: str) -> dict | None:
    """Return dict with 'pull_counts' and 'cumulative_rewards', or None if file absent."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, KeyError):
        return None
