import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
V3_ROOT = REPO_ROOT / "V3"
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from plot import generate_plots
from utils.results_aggregation import aggregate_persisted_runs
from utils.results_persistence import persist_run_results


def _attempt(*, tactic_id: str, tactic_family: str, attack_success: bool, syntax_valid: bool = True) -> dict:
    return {
        "selected_tactic_action": {
            "tactic_id": tactic_id,
            "tactic_family": tactic_family,
        },
        "attack_success": attack_success,
        "syntax_valid": syntax_valid,
        "llm_judge": {"decision": "PASS", "confidence": 0.8},
        "test_judge": {"decision": "PASS" if attack_success else "FAIL"},
        "trace": {
            "summary": {
                "failure_stage": None,
            },
            "selector_output": {
                "tactic_id": tactic_id,
                "tactic_family": tactic_family,
            },
        },
    }


class PlottingPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp_root = REPO_ROOT / ".tmp_test_plotting" / uuid4().hex
        self.results_dir = self.temp_root / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def _write_run(self, *, benchmark: str, policy_mode: str, run_suffix: str, attack_succeeded: bool) -> None:
        persist_run_results(
            results_dir=str(self.results_dir),
            metadata={
                "benchmark": benchmark,
                "policy_mode": policy_mode,
                "experiment_mode": "iterative",
                "experiment_split": "test",
                "bandit_algorithm": "ucb1" if policy_mode == "rl_bandit" else None,
                "selector_cot_enabled": True,
                "bandit_freeze_weights_effective": False,
                "attack_succeeded": attack_succeeded,
                "successful_iteration": 2 if attack_succeeded else None,
                "baseline": _attempt(
                    tactic_id="semantic",
                    tactic_family="semantic",
                    attack_success=False,
                ),
                "all_attempts": [
                    _attempt(
                        tactic_id="semantic",
                        tactic_family="semantic",
                        attack_success=attack_succeeded,
                    ),
                    _attempt(
                        tactic_id="prompting",
                        tactic_family="prompting",
                        attack_success=False,
                    ),
                ],
            },
            task_name=f"task-{run_suffix}",
            benchmark=benchmark,
            policy_mode=policy_mode,
            experiment_split="test",
            split_definition="unit-test",
            bandit_algorithm="ucb1" if policy_mode == "rl_bandit" else None,
            selector_cot_enabled=True,
            bandit_freeze_weights_effective=False,
            experiment_mode="iterative",
            code_generation_model="test-model",
            target_model="judge-model",
            selector_model="selector-model",
            max_iterations=3,
        )

    def test_generate_plots_from_persisted_runs(self):
        self._write_run(
            benchmark="mbpp",
            policy_mode="rl_bandit",
            run_suffix="a",
            attack_succeeded=True,
        )
        self._write_run(
            benchmark="mbpp",
            policy_mode="rl_bandit",
            run_suffix="b",
            attack_succeeded=False,
        )
        self._write_run(
            benchmark="humaneval",
            policy_mode="agent_based_decision",
            run_suffix="c",
            attack_succeeded=True,
        )

        aggregation = aggregate_persisted_runs(results_dir=str(self.results_dir))
        output_dir = self.temp_root / "plots"
        manifest = generate_plots(aggregation, str(output_dir))

        self.assertEqual(manifest["run_count"], 3)
        self.assertGreaterEqual(manifest["plot_count"], 6)

        self.assertTrue((output_dir / "plot_manifest.json").exists())
        self.assertTrue((output_dir / "attack_success_rate_by_policy_mode.png").exists())
        self.assertTrue((output_dir / "success_by_benchmark.png").exists())
        self.assertTrue((output_dir / "syntax_invalid_rate_by_policy_mode.png").exists())
        self.assertTrue((output_dir / "one_shot_vs_iterative_comparison.png").exists())
        self.assertTrue((output_dir / "train_validation_test_comparison.png").exists())
        self.assertTrue((output_dir / "iterations_to_success_distribution.png").exists())
        self.assertTrue((output_dir / "arm_pull_counts.png").exists())
        self.assertTrue((output_dir / "average_reward_by_arm.png").exists())
        self.assertGreaterEqual(len(list(output_dir.glob("arm_preference_over_time_*.png"))), 1)
        self.assertGreaterEqual(len(list(output_dir.glob("rl_bandit_evolution_*.png"))), 1)

        manifest_payload = json.loads((output_dir / "plot_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest_payload["plot_count"], manifest["plot_count"])


if __name__ == "__main__":
    unittest.main()