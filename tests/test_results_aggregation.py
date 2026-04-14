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

from utils.results_aggregation import aggregate_persisted_runs, write_aggregation_artifacts


class ResultsAggregationTests(unittest.TestCase):
    def setUp(self):
        self.temp_root = REPO_ROOT / ".tmp_test_aggregation" / uuid4().hex
        self.results_dir = self.temp_root / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def _write_run(self, run_id: str, config: dict, summary: dict, attempts: list[dict]):
        run_dir = self.results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
        (run_dir / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        with (run_dir / "attempts.jsonl").open("w", encoding="utf-8") as handle:
            for attempt in attempts:
                handle.write(json.dumps(attempt) + "\n")

    def test_aggregate_groups_mbpp_and_humaneval_across_policy_modes(self):
        self._write_run(
            "2026-04-12_20-37-46_mbpp_agent_based_decision_model_aaaa1111",
            {
                "run_id": "2026-04-12_20-37-46_mbpp_agent_based_decision_model_aaaa1111",
                "timestamp": "2026-04-12T20:37:46+01:00",
                "benchmark": "mbpp",
                "policy_mode": "agent_based_decision",
                "experiment_mode": "iterative",
                "experiment_split": "test",
            },
            {
                "run_id": "2026-04-12_20-37-46_mbpp_agent_based_decision_model_aaaa1111",
                "benchmark": "mbpp",
                "policy_mode": "agent_based_decision",
                "experiment_mode": "iterative",
                "experiment_split": "test",
                "attack_success_rate": 0.5,
                "syntax_invalid_rate": 0.1,
                "invalid_attempt_rate": 0.2,
                "average_iterations_to_success": 2.0,
                "average_llm_confidence": 0.65,
                "successful_samples": 1,
                "failed_samples": 1,
            },
            [{"run_id": "x"}],
        )

        self._write_run(
            "2026-04-12_20-45-01_humaneval_rl_bandit_model_bbbb2222",
            {
                "run_id": "2026-04-12_20-45-01_humaneval_rl_bandit_model_bbbb2222",
                "timestamp": "2026-04-12T20:45:01+01:00",
                "benchmark": "humaneval",
                "policy_mode": "rl_bandit",
                "bandit_algorithm": "ucb1",
                "experiment_mode": "iterative",
                "experiment_split": "test",
            },
            {
                "run_id": "2026-04-12_20-45-01_humaneval_rl_bandit_model_bbbb2222",
                "benchmark": "humaneval",
                "policy_mode": "rl_bandit",
                "bandit_algorithm": "ucb1",
                "experiment_mode": "iterative",
                "experiment_split": "test",
                "attack_success_rate": 0.25,
                "syntax_invalid_rate": 0.3,
                "invalid_attempt_rate": 0.4,
                "average_iterations_to_success": 4.0,
                "average_llm_confidence": 0.4,
                "successful_samples": 1,
                "failed_samples": 3,
            },
            [{"run_id": "y"}],
        )

        aggregation = aggregate_persisted_runs(results_dir=str(self.results_dir))

        self.assertEqual(aggregation["aggregation_metadata"]["run_count"], 2)
        self.assertEqual(
            aggregation["aggregation_metadata"]["benchmarks"],
            ["humaneval", "mbpp"],
        )
        self.assertEqual(
            sorted(aggregation["aggregation_metadata"]["policy_modes"]),
            ["agent_based_decision", "rl_bandit"],
        )

        groups = aggregation["grouped_summary"]
        self.assertEqual(len(groups), 2)
        by_key = {
            (item["benchmark"], item["policy_mode"]): item
            for item in groups
        }
        self.assertAlmostEqual(
            by_key[("mbpp", "agent_based_decision")]["mean_attack_success_rate"],
            0.5,
        )
        self.assertAlmostEqual(
            by_key[("humaneval", "rl_bandit")]["mean_attack_success_rate"],
            0.25,
        )

    def test_aggregation_artifacts_are_written_and_evolution_is_time_sorted(self):
        self._write_run(
            "2026-04-12_20-00-00_mbpp_random_choice_model_cccc3333",
            {
                "run_id": "2026-04-12_20-00-00_mbpp_random_choice_model_cccc3333",
                "timestamp": "2026-04-12T20:00:00+01:00",
                "benchmark": "mbpp",
                "policy_mode": "random_choice",
                "experiment_mode": "iterative",
                "experiment_split": "test",
            },
            {
                "run_id": "2026-04-12_20-00-00_mbpp_random_choice_model_cccc3333",
                "benchmark": "mbpp",
                "policy_mode": "random_choice",
                "experiment_mode": "iterative",
                "experiment_split": "test",
                "attack_success_rate": 0.1,
                "syntax_invalid_rate": 0.5,
                "invalid_attempt_rate": 0.6,
                "average_iterations_to_success": None,
                "average_llm_confidence": 0.2,
                "successful_samples": 1,
                "failed_samples": 9,
            },
            [{"run_id": "1"}],
        )

        self._write_run(
            "2026-04-12_21-00-00_mbpp_random_choice_model_dddd4444",
            {
                "run_id": "2026-04-12_21-00-00_mbpp_random_choice_model_dddd4444",
                "timestamp": "2026-04-12T21:00:00+01:00",
                "benchmark": "mbpp",
                "policy_mode": "random_choice",
                "experiment_mode": "iterative",
                "experiment_split": "test",
            },
            {
                "run_id": "2026-04-12_21-00-00_mbpp_random_choice_model_dddd4444",
                "benchmark": "mbpp",
                "policy_mode": "random_choice",
                "experiment_mode": "iterative",
                "experiment_split": "test",
                "attack_success_rate": 0.2,
                "syntax_invalid_rate": 0.4,
                "invalid_attempt_rate": 0.5,
                "average_iterations_to_success": 3.0,
                "average_llm_confidence": 0.3,
                "successful_samples": 2,
                "failed_samples": 8,
            },
            [{"run_id": "2"}],
        )

        aggregation = aggregate_persisted_runs(results_dir=str(self.results_dir))
        output_dir = self.temp_root / "aggregates"
        outputs = write_aggregation_artifacts(
            aggregation=aggregation,
            output_dir=str(output_dir),
        )

        for path in outputs.values():
            self.assertTrue(Path(path).exists())

        evolution = aggregation["evolution_by_group"]
        self.assertEqual(len(evolution), 1)
        run_ids = [row["run_id"] for row in evolution[0]["runs"]]
        self.assertEqual(
            run_ids,
            [
                "2026-04-12_20-00-00_mbpp_random_choice_model_cccc3333",
                "2026-04-12_21-00-00_mbpp_random_choice_model_dddd4444",
            ],
        )


if __name__ == "__main__":
    unittest.main()
