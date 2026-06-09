"""
log_entropy.py — compute and append per-epoch arm entropy to a curve file.

Usage:
    python V3/scripts/log_entropy.py <checkpoint.json> <base_weights.json>

Appends one entry to <base_weights_entropy_curve.json>:
    {"epoch": N, "arm_entropy": H, "normalized_entropy": H/log(K),
     "n_pulls": total, "checkpoint": "filename.json"}

normalized_entropy = 1.0 means fully uniform (pure exploration).
normalized_entropy = 0.0 means all pulls on one arm (pure exploitation).
"""
import json
import math
import os
import sys


def compute_entropy(checkpoint_path: str, base_weights_path: str) -> None:
    with open(checkpoint_path) as f:
        data = json.load(f)

    pull_counts = data.get("pull_counts", {})
    total = sum(pull_counts.values())
    n_arms = len(pull_counts)

    if total > 0 and n_arms > 1:
        probs = [c / total for c in pull_counts.values() if c > 0]
        entropy = -sum(p * math.log(p) for p in probs)
        max_entropy = math.log(n_arms)
        normalized = entropy / max_entropy
    else:
        entropy = 0.0
        normalized = 0.0

    curve_path = base_weights_path.replace(".json", "_entropy_curve.json")
    curve = []
    if os.path.exists(curve_path):
        with open(curve_path) as f:
            curve = json.load(f)

    epoch = len(curve) + 1
    entry = {
        "epoch": epoch,
        "arm_entropy": round(entropy, 4),
        "normalized_entropy": round(normalized, 4),
        "n_pulls": total,
        "checkpoint": os.path.basename(checkpoint_path),
    }
    curve.append(entry)

    with open(curve_path, "w") as f:
        json.dump(curve, f, indent=2)

    print(
        f"  epoch={epoch}  entropy={entropy:.4f}"
        f"  normalized={normalized:.3f}"
        f"  pulls={total}  arms={n_arms}"
    )
    print(f"  curve -> {curve_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python log_entropy.py <checkpoint.json> <base_weights.json>")
        sys.exit(1)
    compute_entropy(sys.argv[1], sys.argv[2])
