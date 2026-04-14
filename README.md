# Adversarial Attacks on Code LLMs using Reinforcement Learning

This repository implements the adversarial attack benchmark for Code LLMs in the conference track:

- benchmarks: MBPP and HumanEval
- policy modes: random_choice, agent_based_decision, rl_bandit
- experiment modes: one_shot and iterative
- success condition:

```text
Attack Success = (Test Judge = FAIL) AND (LLM Judge = PASS)
```

## Quick Start

### 1) Set up environment

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

### 2) Generate results (minimal example)

```bash
inspect eval V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen3.5:0.8b \
  -T benchmark=mbpp \
  -T mutation_strategy=react \
  -T policy_mode=agent_based_decision \
  -T experiment_mode=iterative \
  -T use_llm_judge=True \
  -T judge_model=ollama/qwen3.5:0.8b \
  -T selector_model=ollama/qwen3.5:0.8b \
  -T max_iterations=2 \
  --max-samples 2 \
  --limit 2
```

Note: raw run outputs are saved to `results/` (which is in gitignore).

### 3) Aggregate results

```bash
python V3/scripts/aggregate_results.py --results-dir results --output-dir results/aggregates
```

### 4) Generate final plots outside results (versionable)

```bash
python plot.py --results-dir results --output-dir plots/latest
```

Final plots are written to `plots/latest`, outside `results/`.

## Supported Configuration

Main benchmark parameters:

- `-T benchmark`: `mbpp` | `humaneval`
- `-T policy_mode`: `random_choice` | `agent_based_decision` | `rl_bandit`
- `-T experiment_mode`: `one_shot` | `iterative`
- `-T max_iterations`: integer >= 1
- `-T selector_use_cot`: `True` | `False` (relevant for `agent_based_decision`)
- `-T bandit_algorithm`: currently `ucb1`
- `-T bandit_weights_path`: path to persist/reuse weights
- `-T bandit_freeze_weights`: `True` | `False`
- `-T experiment_split`: `train` | `validation` | `test`
- `-T split_definition`: textual label of the selected split
- `--max-samples`: maximum number of samples
- `--limit`: task range/subset

Plot parameters:

- `--results-dir`: directory containing persisted runs (default: `results`)
- `--output-dir`: plot output directory (default: `plots/<timestamp>`)
- `--benchmark`: optional filter (repeatable)
- `--policy-mode`: optional filter (repeatable)

## Generated Plots

The current pipeline generates, when enough data is available:

- attack_success_rate_by_policy_mode
- success_by_benchmark
- syntax_invalid_rate_by_policy_mode
- one_shot_vs_iterative_comparison
- train_validation_test_comparison
- iterations_to_success_distribution
- arm_pull_counts
- average_reward_by_arm
- arm_preference_over_time per group
- rl_bandit_evolution per group (when applicable)

It also creates `plot_manifest.json` with the complete list of generated files.

## Ready-to-Use Commands

For the full command matrix and validation steps:

- `V3/script_copy_past.sh`

## References

- Inspect AI: https://inspect.aisi.org.uk/
- ReAct docs: https://inspect.aisi.org.uk/react-agent.html
- Agent pattern reference: https://github.com/rufimelo99/inspect_evals/blob/main/src/inspect_evals/agentharm/agents/default_agent.py
- MBPP: https://github.com/google-research/google-research/tree/master/mbpp
- HumanEval: https://github.com/openai/human-eval
- gitea Example (Rui melo): https://huggingface.co/spaces/rufimelo/github-red-trajectory-viewer
- Model selection: https://github.com/cheahjs/free-llm-api-resources?tab=readme-ov-file#huggingface-inference-providers
- tree-sitter Python grammar: https://github.com/tree-sitter/py-tree-sitter
