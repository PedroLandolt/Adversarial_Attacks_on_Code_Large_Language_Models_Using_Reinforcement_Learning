# ==========================================================
# Benchmark command cheat sheet
# ==========================================================
#
# This file is only a copy/paste helper for the current
# benchmark path in:
#   V3/adversarial_attack.py@adversarial_code_llm
#
# Supported benchmark dimensions in the current code:
# - benchmark: mbpp | humaneval
# - policy_mode: random_choice | agent_based_decision | rl_bandit
# - experiment_mode: one_shot | iterative
#
# Important:
# The evals conda env may have a site-packages `utils` package that
# shadows `V3/utils`. If that happens, run with:
#
#   PYTHONPATH=V3 conda run -n evals inspect eval ...
#
# Or, if you are already inside the correct environment:
#
#   PYTHONPATH=V3 inspect eval ...
#
# Model defaults used below:
# - target model:   ollama/qwen3.5:0.8b
# - judge model:    ollama/qwen3.5:0.8b
# - selector model: ollama/qwen3.5:0.8b
#
# Convention:
# - quick smoke runs use: --max-samples 1 --limit 1
# - paper-ish runs use:   --max-samples 5 --limit 5
#

# ==========================================================
# 0. Quick launcher alternatives
# ==========================================================
#
# These top-level launchers are good for fast benchmark smoke tests:
#
#   bash V3/run.sh mbpp --quick
#   bash V3/run.sh humaneval --quick
#
# They currently default to the benchmark path with:
# - mutation_strategy=react
# - use_llm_judge=True
#
# For explicit policy-mode comparisons, use the inspect commands below.


# ==========================================================
# 1. MBPP - quick smoke runs
# ==========================================================

# # MBPP + random_choice + one_shot
# inspect eval V3/adversarial_attack.py@adversarial_code_llm \
#   --model ollama/qwen3.5:0.8b \
#   -T benchmark=mbpp \
#   -T mutation_strategy=react \
#   -T policy_mode=random_choice \
#   -T experiment_mode=one_shot \
#   -T use_llm_judge=True \
#   -T judge_model=ollama/qwen3.5:0.8b \
#   -T selector_model=ollama/qwen3.5:0.8b \
#   -T max_iterations=1 \
#   --max-samples 1 \
#   --limit 1

# # MBPP + agent_based_decision + one_shot
# inspect eval V3/adversarial_attack.py@adversarial_code_llm \
#   --model ollama/qwen3.5:0.8b \
#   -T benchmark=mbpp \
#   -T mutation_strategy=react \
#   -T policy_mode=agent_based_decision \
#   -T experiment_mode=one_shot \
#   -T use_llm_judge=True \
#   -T judge_model=ollama/qwen3.5:0.8b \
#   -T selector_model=ollama/qwen3.5:0.8b \
#   -T max_iterations=1 \
#   --max-samples 1 \
#   --limit 1

# # MBPP + rl_bandit (ucb1) + one_shot
# inspect eval V3/adversarial_attack.py@adversarial_code_llm \
#   --model ollama/qwen3.5:0.8b \
#   -T benchmark=mbpp \
#   -T mutation_strategy=react \
#   -T policy_mode=rl_bandit \
#   -T bandit_algorithm=ucb1 \
#   -T experiment_mode=one_shot \
#   -T use_llm_judge=True \
#   -T judge_model=ollama/qwen3.5:0.8b \
#   -T selector_model=ollama/qwen3.5:0.8b \
#   -T max_iterations=1 \
#   --max-samples 1 \
#   --limit 1


# # ==========================================================
# # 2. MBPP - iterative comparisons
# # ==========================================================

# # MBPP + random_choice + iterative
# inspect eval V3/adversarial_attack.py@adversarial_code_llm \
#   --model ollama/qwen3.5:0.8b \
#   -T benchmark=mbpp \
#   -T mutation_strategy=react \
#   -T policy_mode=random_choice \
#   -T experiment_mode=iterative \
#   -T use_llm_judge=True \
#   -T judge_model=ollama/qwen3.5:0.8b \
#   -T selector_model=ollama/qwen3.5:0.8b \
#   -T max_iterations=12 \
#   --max-samples 5 \
#   --limit 5

# # MBPP + agent_based_decision + iterative
# inspect eval V3/adversarial_attack.py@adversarial_code_llm \
#   --model ollama/qwen3.5:0.8b \
#   -T benchmark=mbpp \
#   -T mutation_strategy=react \
#   -T policy_mode=agent_based_decision \
#   -T experiment_mode=iterative \
#   -T use_llm_judge=True \
#   -T judge_model=ollama/qwen3.5:0.8b \
#   -T selector_model=ollama/qwen3.5:0.8b \
#   -T max_iterations=12 \
#   --max-samples 5 \
#   --limit 5

# # MBPP + rl_bandit (ucb1) + iterative
# inspect eval V3/adversarial_attack.py@adversarial_code_llm \
#   --model ollama/qwen3.5:0.8b \
#   -T benchmark=mbpp \
#   -T mutation_strategy=react \
#   -T policy_mode=rl_bandit \
#   -T bandit_algorithm=ucb1 \
#   -T experiment_mode=iterative \
#   -T use_llm_judge=True \
#   -T judge_model=ollama/qwen3.5:0.8b \
#   -T selector_model=ollama/qwen3.5:0.8b \
#   -T max_iterations=12 \
#   --max-samples 5 \
#   --limit 5


# # ==========================================================
# # 3. MBPP - rl_bandit with persistent weights
# # ==========================================================
# #
# # Use this when you want the bandit to keep learning across runs.

# # Train / update weights
# inspect eval V3/adversarial_attack.py@adversarial_code_llm \
#   --model ollama/qwen3.5:0.8b \
#   -T benchmark=mbpp \
#   -T experiment_split=train \
#   -T split_definition=mbpp:70_15_15:1-649 \
#   -T mutation_strategy=react \
#   -T policy_mode=rl_bandit \
#   -T bandit_algorithm=ucb1 \
#   -T bandit_weights_path=weights/mbpp_ucb1.json \
#   -T experiment_mode=iterative \
#   -T use_llm_judge=True \
#   -T judge_model=ollama/qwen3.5:0.8b \
#   -T selector_model=ollama/qwen3.5:0.8b \
#   -T max_iterations=12 \
#   --max-samples 5 \
#   --limit 5

# # Evaluate with frozen weights (no further updates)
# inspect eval V3/adversarial_attack.py@adversarial_code_llm \
#   --model ollama/qwen3.5:0.8b \
#   -T benchmark=mbpp \
#   -T experiment_split=test \
#   -T split_definition=mbpp:70_15_15:789-927 \
#   -T mutation_strategy=react \
#   -T policy_mode=rl_bandit \
#   -T bandit_algorithm=ucb1 \
#   -T bandit_weights_path=weights/mbpp_ucb1.json \
#   -T bandit_freeze_weights=True \
#   -T experiment_mode=iterative \
#   -T use_llm_judge=True \
#   -T judge_model=ollama/qwen3.5:0.8b \
#   -T selector_model=ollama/qwen3.5:0.8b \
#   -T max_iterations=12 \
#   --max-samples 5 \
#   --limit 5


# # ==========================================================
# # 4. HumanEval - quick smoke runs
# # ==========================================================

# # HumanEval + random_choice + one_shot
# inspect eval V3/adversarial_attack.py@adversarial_code_llm \
#   --model ollama/qwen3.5:0.8b \
#   -T benchmark=humaneval \
#   -T mutation_strategy=react \
#   -T policy_mode=random_choice \
#   -T experiment_mode=one_shot \
#   -T use_llm_judge=True \
#   -T judge_model=ollama/qwen3.5:0.8b \
#   -T selector_model=ollama/qwen3.5:0.8b \
#   -T max_iterations=1 \
#   --max-samples 1 \
#   --limit 1

# # HumanEval + agent_based_decision + one_shot
# inspect eval V3/adversarial_attack.py@adversarial_code_llm \
#   --model ollama/qwen3.5:0.8b \
#   -T benchmark=humaneval \
#   -T mutation_strategy=react \
#   -T policy_mode=agent_based_decision \
#   -T experiment_mode=one_shot \
#   -T use_llm_judge=True \
#   -T judge_model=ollama/qwen3.5:0.8b \
#   -T selector_model=ollama/qwen3.5:0.8b \
#   -T max_iterations=1 \
#   --max-samples 1 \
#   --limit 1

# # HumanEval + rl_bandit (ucb1) + one_shot
# inspect eval V3/adversarial_attack.py@adversarial_code_llm \
#   --model ollama/qwen3.5:0.8b \
#   -T benchmark=humaneval \
#   -T mutation_strategy=react \
#   -T policy_mode=rl_bandit \
#   -T bandit_algorithm=ucb1 \
#   -T experiment_mode=one_shot \
#   -T use_llm_judge=True \
#   -T judge_model=ollama/qwen3.5:0.8b \
#   -T selector_model=ollama/qwen3.5:0.8b \
#   -T max_iterations=1 \
#   --max-samples 1 \
#   --limit 1


# # ==========================================================
# # 5. HumanEval - iterative comparisons
# # ==========================================================

# # HumanEval + random_choice + iterative
# inspect eval V3/adversarial_attack.py@adversarial_code_llm \
#   --model ollama/qwen3.5:0.8b \
#   -T benchmark=humaneval \
#   -T mutation_strategy=react \
#   -T policy_mode=random_choice \
#   -T experiment_mode=iterative \
#   -T use_llm_judge=True \
#   -T judge_model=ollama/qwen3.5:0.8b \
#   -T selector_model=ollama/qwen3.5:0.8b \
#   -T max_iterations=12 \
#   --max-samples 5 \
#   --limit 5

# # HumanEval + agent_based_decision + iterative
# inspect eval V3/adversarial_attack.py@adversarial_code_llm \
#   --model ollama/qwen3.5:0.8b \
#   -T benchmark=humaneval \
#   -T mutation_strategy=react \
#   -T policy_mode=agent_based_decision \
#   -T experiment_mode=iterative \
#   -T use_llm_judge=True \
#   -T judge_model=ollama/qwen3.5:0.8b \
#   -T selector_model=ollama/qwen3.5:0.8b \
#   -T max_iterations=12 \
#   --max-samples 5 \
#   --limit 5

# # HumanEval + rl_bandit (ucb1) + iterative
# inspect eval V3/adversarial_attack.py@adversarial_code_llm \
#   --model ollama/qwen3.5:0.8b \
#   -T benchmark=humaneval \
#   -T mutation_strategy=react \
#   -T policy_mode=rl_bandit \
#   -T bandit_algorithm=ucb1 \
#   -T experiment_mode=iterative \
#   -T use_llm_judge=True \
#   -T judge_model=ollama/qwen3.5:0.8b \
#   -T selector_model=ollama/qwen3.5:0.8b \
#   -T max_iterations=12 \
#   --max-samples 5 \
#   --limit 5


# # ==========================================================
# # 6. HumanEval - rl_bandit with persistent weights
# # ==========================================================

# # Train / update weights
# inspect eval V3/adversarial_attack.py@adversarial_code_llm \
#   --model ollama/qwen3.5:0.8b \
#   -T benchmark=humaneval \
#   -T experiment_split=train \
#   -T split_definition=humaneval:70_15_15:train \
#   -T mutation_strategy=react \
#   -T policy_mode=rl_bandit \
#   -T bandit_algorithm=ucb1 \
#   -T bandit_weights_path=weights/humaneval_ucb1.json \
#   -T experiment_mode=iterative \
#   -T use_llm_judge=True \
#   -T judge_model=ollama/qwen3.5:0.8b \
#   -T selector_model=ollama/qwen3.5:0.8b \
#   -T max_iterations=12 \
#   --max-samples 5 \
#   --limit 5

# # Evaluate with frozen weights
# inspect eval V3/adversarial_attack.py@adversarial_code_llm \
#   --model ollama/qwen3.5:0.8b \
#   -T benchmark=humaneval \
#   -T experiment_split=test \
#   -T split_definition=humaneval:70_15_15:test \
#   -T mutation_strategy=react \
#   -T policy_mode=rl_bandit \
#   -T bandit_algorithm=ucb1 \
#   -T bandit_weights_path=weights/humaneval_ucb1.json \
#   -T bandit_freeze_weights=True \
#   -T experiment_mode=iterative \
#   -T use_llm_judge=True \
#   -T judge_model=ollama/qwen3.5:0.8b \
#   -T selector_model=ollama/qwen3.5:0.8b \
#   -T max_iterations=12 \
#   --max-samples 5 \
#   --limit 5


# # ==========================================================
# # 7. Split workflow defaults
# # ==========================================================
# #
# # Recommended default split strategy for current experiments:
# # - MBPP:      70 / 15 / 15 over 927 samples
# #   - train:      1-649
# #   - validation: 650-788
# #   - test:       789-927
# #
# # - HumanEval: use the same 70 / 15 / 15 convention and record the
# #   concrete subset in split_definition. If you keep the standard 164-task
# #   version, a practical default is:
# #   - train:      1-115
# #   - validation: 116-139
# #   - test:       140-164
# #
# # The benchmark itself should not slice the dataset for this task.
# # We keep the split explicit through:
# # - inspect `--limit` ranges
# # - `-T experiment_split=<train|validation|test>`
# # - `-T split_definition=<benchmark:scheme:range>`

# # MBPP train run (updates weights)
# inspect eval V3/adversarial_attack.py@adversarial_code_llm \
#   --model ollama/qwen3.5:0.8b \
#   -T benchmark=mbpp \
#   -T experiment_split=train \
#   -T split_definition=mbpp:70_15_15:1-649 \
#   -T mutation_strategy=react \
#   -T policy_mode=rl_bandit \
#   -T bandit_algorithm=ucb1 \
#   -T bandit_weights_path=weights/mbpp_ucb1.json \
#   -T experiment_mode=iterative \
#   -T use_llm_judge=True \
#   -T judge_model=ollama/qwen3.5:0.8b \
#   -T selector_model=ollama/qwen3.5:0.8b \
#   -T max_iterations=12 \
#   --max-samples 5 \
#   --limit 1-649

# # MBPP validation run (frozen weights)
# inspect eval V3/adversarial_attack.py@adversarial_code_llm \
#   --model ollama/qwen3.5:0.8b \
#   -T benchmark=mbpp \
#   -T experiment_split=validation \
#   -T split_definition=mbpp:70_15_15:650-788 \
#   -T mutation_strategy=react \
#   -T policy_mode=rl_bandit \
#   -T bandit_algorithm=ucb1 \
#   -T bandit_weights_path=weights/mbpp_ucb1.json \
#   -T bandit_freeze_weights=True \
#   -T experiment_mode=iterative \
#   -T use_llm_judge=True \
#   -T judge_model=ollama/qwen3.5:0.8b \
#   -T selector_model=ollama/qwen3.5:0.8b \
#   -T max_iterations=12 \
#   --max-samples 5 \
#   --limit 650-788

# # MBPP test run (frozen weights)
# inspect eval V3/adversarial_attack.py@adversarial_code_llm \
#   --model ollama/qwen3.5:0.8b \
#   -T benchmark=mbpp \
#   -T experiment_split=test \
#   -T split_definition=mbpp:70_15_15:789-927 \
#   -T mutation_strategy=react \
#   -T policy_mode=rl_bandit \
#   -T bandit_algorithm=ucb1 \
#   -T bandit_weights_path=weights/mbpp_ucb1.json \
#   -T bandit_freeze_weights=True \
#   -T experiment_mode=iterative \
#   -T use_llm_judge=True \
#   -T judge_model=ollama/qwen3.5:0.8b \
#   -T selector_model=ollama/qwen3.5:0.8b \
#   -T max_iterations=12 \
#   --max-samples 5 \
#   --limit 789-927

# ==========================================================
# 8. Useful variations
# ==========================================================
#
# Change only these knobs when needed:
#
# Smaller smoke test:
#   -T max_iterations=1 --max-samples 1 --limit 1
#
# More attack budget:
#   -T max_iterations=20
#
# Custom output folder:
#   -T results_dir=results/paper_runs
#
# Custom model:
#   --model <target_model>
#   -T judge_model=<judge_model>
#   -T selector_model=<selector_model>
#
# If you want to compare policies fairly, keep the following fixed:
# - benchmark
# - model / judge_model / selector_model
# - experiment_mode
# - max_iterations
# - max-samples
# - limit


# ==========================================================
# 9. Task 12 - final done done done validation pack
# ==========================================================
#
# Goal:
# - run comparable experiments for MBPP + HumanEval
# - include random_choice + agent_based_decision + rl_bandit
# - aggregate persisted runs into stable artifacts
# - verify coverage and evolution over multiple runs

# 9.1 Run aggregation unit tests (must pass)
c:/Users/cesar/Desktop/Landolt/Tese/Adversarial_Attacks_on_Code_Large_Language_Models_Using_Reinforcement_Learning/.venv/Scripts/python.exe -m unittest tests.test_results_aggregation -v


# 9.2 MBPP and HumanEval - compact comparable run set
#
# Run each command at least twice on different timestamps
# if you want stronger evolution-over-time evidence.

# MBPP + random_choice
inspect eval V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen3.5:0.8b \
  -T benchmark=mbpp \
  -T mutation_strategy=react \
  -T policy_mode=random_choice \
  -T experiment_mode=iterative \
  -T use_llm_judge=True \
  -T judge_model=ollama/qwen3.5:0.8b \
  -T selector_model=ollama/qwen3.5:0.8b \
  -T max_iterations=2 \
  --max-samples 2 \
  --limit 2

# MBPP + agent_based_decision
inspect eval V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen3.5:0.8b \
  -T benchmark=mbpp \
  -T mutation_strategy=react \
  -T policy_mode=agent_based_decision \
  -T selector_use_cot=False \
  -T experiment_mode=iterative \
  -T use_llm_judge=True \
  -T judge_model=ollama/qwen3.5:0.8b \
  -T selector_model=ollama/qwen3.5:0.8b \
  -T max_iterations=2 \
  --max-samples 2 \
  --limit 2

# MBPP + rl_bandit
inspect eval V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen3.5:0.8b \
  -T benchmark=mbpp \
  -T mutation_strategy=react \
  -T policy_mode=rl_bandit \
  -T bandit_algorithm=ucb1 \
  -T bandit_weights_path=weights/mbpp_ucb1.json \
  -T experiment_mode=iterative \
  -T use_llm_judge=True \
  -T judge_model=ollama/qwen3.5:0.8b \
  -T selector_model=ollama/qwen3.5:0.8b \
  -T max_iterations=2 \
  --max-samples 2 \
  --limit 2

# HumanEval + random_choice
inspect eval V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen3.5:0.8b \
  -T benchmark=humaneval \
  -T mutation_strategy=react \
  -T policy_mode=random_choice \
  -T experiment_mode=iterative \
  -T use_llm_judge=True \
  -T judge_model=ollama/qwen3.5:0.8b \
  -T selector_model=ollama/qwen3.5:0.8b \
  -T max_iterations=2 \
  --max-samples 2 \
  --limit 2

# HumanEval + agent_based_decision
inspect eval V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen3.5:0.8b \
  -T benchmark=humaneval \
  -T mutation_strategy=react \
  -T policy_mode=agent_based_decision \
  -T selector_use_cot=False \
  -T experiment_mode=iterative \
  -T use_llm_judge=True \
  -T judge_model=ollama/qwen3.5:0.8b \
  -T selector_model=ollama/qwen3.5:0.8b \
  -T max_iterations=2 \
  --max-samples 2 \
  --limit 2

# HumanEval + rl_bandit
inspect eval V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen3.5:0.8b \
  -T benchmark=humaneval \
  -T mutation_strategy=react \
  -T policy_mode=rl_bandit \
  -T bandit_algorithm=ucb1 \
  -T bandit_weights_path=weights/humaneval_ucb1.json \
  -T experiment_mode=iterative \
  -T use_llm_judge=True \
  -T judge_model=ollama/qwen3.5:0.8b \
  -T selector_model=ollama/qwen3.5:0.8b \
  -T max_iterations=2 \
  --max-samples 2 \
  --limit 2


# 9.3 Rebuild aggregates
c:/Users/cesar/Desktop/Landolt/Tese/Adversarial_Attacks_on_Code_Large_Language_Models_Using_Reinforcement_Learning/.venv/Scripts/python.exe V3/scripts/aggregate_results.py --results-dir results --output-dir results/aggregates


# 9.4 Validate required policy coverage in grouped_summary.json
c:/Users/cesar/Desktop/Landolt/Tese/Adversarial_Attacks_on_Code_Large_Language_Models_Using_Reinforcement_Learning/.venv/Scripts/python.exe - <<'PY'
import json
from pathlib import Path

summary_path = Path("results/aggregates/grouped_summary.json")
data = json.loads(summary_path.read_text(encoding="utf-8"))

required = {
    ("mbpp", "random_choice"),
    ("mbpp", "agent_based_decision"),
    ("mbpp", "rl_bandit"),
    ("humaneval", "random_choice"),
    ("humaneval", "agent_based_decision"),
    ("humaneval", "rl_bandit"),
}

present = {(row.get("benchmark"), row.get("policy_mode")) for row in data}
missing = sorted(required - present)

print("Required pairs:", sorted(required))
print("Present pairs:", sorted(present))
if missing:
    print("MISSING:", missing)
    raise SystemExit(1)

print("Coverage check: OK")
PY


# ==========================================================
# 10. Task 13 - plotting pipeline validation pack
# ==========================================================
#
# Goal:
# - generate figures from persisted runs only
# - validate the offline plotting entrypoint
# - keep the plotting workflow independent from benchmark execution

# 10.1 Plotting unit tests (must pass)
c:/Users/cesar/Desktop/Landolt/Tese/Adversarial_Attacks_on_Code_Large_Language_Models_Using_Reinforcement_Learning/.venv/Scripts/python.exe -m unittest tests.test_plotting_pipeline -v


# 10.2 Generate plots from the current persisted results
c:/Users/cesar/Desktop/Landolt/Tese/Adversarial_Attacks_on_Code_Large_Language_Models_Using_Reinforcement_Learning/.venv/Scripts/python.exe plot.py --results-dir results --output-dir plots/latest


# 10.3 Validate plotting outputs exist
c:/Users/cesar/Desktop/Landolt/Tese/Adversarial_Attacks_on_Code_Large_Language_Models_Using_Reinforcement_Learning/.venv/Scripts/python.exe - <<'PY'
import json
from pathlib import Path

manifest_path = Path("plots/latest/plot_manifest.json")
data = json.loads(manifest_path.read_text(encoding="utf-8"))

print("Plot manifest run_count:", data.get("run_count"))
print("Plot manifest plot_count:", data.get("plot_count"))
print("Plot files:")
for item in data.get("plots", []):
    print(" -", item)

if data.get("plot_count", 0) < 6:
    raise SystemExit("Expected at least 6 plot files")

print("Plot validation: OK")
PY


# 9.5 Validate evolution evidence (at least one group with >=2 runs)
c:/Users/cesar/Desktop/Landolt/Tese/Adversarial_Attacks_on_Code_Large_Language_Models_Using_Reinforcement_Learning/.venv/Scripts/python.exe - <<'PY'
import json
from pathlib import Path

evolution_path = Path("results/aggregates/evolution_by_group.json")
data = json.loads(evolution_path.read_text(encoding="utf-8"))

max_runs = 0
best_group = None
for group in data:
    count = len(group.get("runs", []))
    if count > max_runs:
        max_runs = count
        best_group = group

print("Best evolution group run_count:", max_runs)
if best_group:
    print(
        "Best group:",
        best_group.get("benchmark"),
        best_group.get("policy_mode"),
        best_group.get("experiment_mode"),
        best_group.get("experiment_split"),
    )

if max_runs < 2:
    raise SystemExit("Need at least one group with >= 2 runs for time-evolution evidence")

print("Evolution check: OK")
PY
