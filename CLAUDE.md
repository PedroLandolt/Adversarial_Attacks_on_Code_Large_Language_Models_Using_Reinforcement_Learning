# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Claude Code Project Context

This repo is the thesis project **Adversarial Attacks on Code Large Language Models Using Reinforcement Learning**.

## What this project does

It runs judge-attack experiments on code LLMs. The benchmark goal is:

```text
Attack Success = (Test Judge = FAIL) AND (LLM Judge = PASS)
```

Deterministic tests are the ground truth. The LLM judge is the thing being attacked.

## Main entrypoints

- `V3/adversarial_attack.py` is the main benchmark task.
- `V3/agent/selector_policy.py` defines the selector boundary and policy modes.
- `V3/utils/results_persistence.py` writes run outputs.
- `V3/utils/reward_accounting.py` defines reward and arm accounting.
- `V3/utils/benchmark_loader.py` loads MBPP and HumanEval.

## Project structure

```text
V3/
  adversarial_attack.py        # Main Inspect task for MBPP/HumanEval attack runs.
  AGENTS.md                    # Agent instructions: active conference-track scope.
  agent/
    react_selector.py          # ReactTacticSelector: formats prompt → calls selector model → parses TacticRegistryEntry.
    selector_policy.py         # SelectorPolicy protocol + 3 concrete policies (Random, ReAct, RLBandit).
    tactic_registry.py         # Registry of 9 tactics: tactic_id, tactic_family, renderer_binding.
  attacks/                     # Legacy attack helpers (instruction_perturbation, misleading_comments, variable_renaming).
  judge/
    llm_judge.py               # LLM judge wrapper and scoring logic.
    red_teaming_tactics.py     # Entry points: apply_tactic(), build_tactic_generation_prompt(), generate_tactic_artifact().
    test_judge.py              # Deterministic test judge wrapper.
  prompts/
    tactic_generation.json     # Per-tactic system prompts keyed by renderer_binding. Edit here, not Python.
  scripts/
    aggregate_results.py       # Aggregates persisted run outputs into results/aggregates/.
    run_full_experiment.sh     # Full 8-phase pipeline: RL train→val→test, random, agent, aggregate, plot.
    train_rl_overnight.sh      # Unattended RL training (N epochs) then val/test with frozen weights.
    run_adversarial_code_llm.sh # Single-run shell preset.
  utils/
    bandit_weights.py          # JSON persistence helpers for UCB1 arm weights.
    benchmark_loader.py        # MBPP / HumanEval loading and normalization.
    code_extraction.py         # Normalizes raw LLM completions → executable_code.
    results_aggregation.py     # Aggregation helpers for offline analysis.
    results_persistence.py     # Writes run_config, run_summary, attempts.jsonl.
    reward_accounting.py       # Reward calculation and arm accounting.
    syntax_validator.py        # tree-sitter Python syntax gate; called before every deterministic exec.
weights/
  mbpp_ucb1.json               # Persisted UCB1 arm weights (pull_counts + cumulative_rewards).
```

## Tactic registry

Nine tactics are defined in `V3/agent/tactic_registry.py`. Each has a `tactic_id`, `tactic_family`, `renderer_binding`, and `taxonomy_category`:

| tactic_family          | renderer_binding                  | taxonomy_category    |
| ---------------------- | --------------------------------- | -------------------- |
| `injection`            | `prompt_injection`                | structural_logic     |
| `output`               | `output_manipulation`             | obfuscation_noise    |
| `semantic`             | `semantic_inconsistency`          | narrative_contextual |
| `cot`                  | `cot_poisoning`                   | strategy_pacing      |
| `roleplay`             | `narrative_roleplay`              | narrative_contextual |
| `appeal_to_authority`  | `pressure_authority`              | pressure_persuasion  |
| `formatting_smuggling` | `structural_formatting_smuggling` | structural_logic     |
| `recursion_crescendo`  | `strategy_recursion_crescendo`    | strategy_pacing      |
| `crowding`             | `obfuscation_crowding`            | obfuscation_noise    |

`renderer_binding` is the key used to look up tactic-specific system prompts in `V3/prompts/tactic_generation.json` and to dispatch the `apply_tactic()` renderer in `V3/judge/red_teaming_tactics.py`.

## Experiment flow

1. build a baseline generation,
2. apply an adversarial tactic or mutation,
3. run the LLM judge,
4. run deterministic tests,
5. stop when the attack success condition is met,
6. otherwise continue until the iteration budget is exhausted.

The attack step must stay syntax-safe. If a tactic is selected by RL, a separate LLM step should still generate the actual attack text or prompt with enough context to stay consistent and executable.

Keep these artifact views separate:

- `raw_completion` - raw target-model output before normalization.
- `executable_code` - code extracted from the completion and sent to deterministic tests.
- `review_artifact` - the judge-facing rendering shown to the LLM judge.

Do not collapse them into one artifact.

## Benchmarks

- MBPP
- HumanEval

They are interchangeable through configuration. Do not hardcode benchmark-specific behavior into the task loop unless the benchmark loader requires it.

## Policy modes

- `random_choice`
- `agent_based_decision`
- `rl_bandit`

All policy modes must go through the same selector interface and return a consistent decision shape.

Rules:

- keep `tactic_id`, `tactic_family`, `renderer_binding`, and `selector_reasoning` consistent,
- do not bypass the selector abstraction,
- keep comparisons fair across benchmarks and runs,
- `rl_bandit` currently supports `ucb1` only.

## Experiment modes

- `one_shot`
- `iterative`

`one_shot` stops after one adversarial attempt. `iterative` keeps the adaptive loop going up to `max_iterations`.

Prefer one-shot validation when testing a specific attack in isolation. Use it to check whether a tactic is actually useful before keeping or expanding the attack set.

## Setup and environment

- Python: `>= 3.10`.
- Install dependencies from the repository root with `python -m pip install -r requirements.txt`.
- Create and activate a virtual environment first if needed.
- The run presets load environment variables from `.env` or `V3/.env` when present.
- Common variables are `MODEL`, `JUDGE_MODEL`, and `SELECTOR_MODEL`.
- Use Ollama locally for the default path; make sure the Ollama server is running before launching `ollama/...` models.
- API keys are only needed if you switch to non-Ollama providers through LiteLLM.
- **Docker is required**: the benchmark task runs in `sandbox="docker"`. Deterministic test execution uses `sandbox().exec(...)`.
- **`PYTHONPATH=V3`** is required when invoking Inspect from the repo root. All imports in `adversarial_attack.py` are relative to `V3/` (e.g. `from agent.selector_policy import ...`), not to the repo root.

## Architecture notes

**Conference scope**: `V3/AGENTS.md` defines the active conference-track boundaries: MBPP + HumanEval only, three policy modes, Gitea and sequential RL (SARSA/Q-learning) out of scope. Consult it before adding new policy types or benchmarks.

**Module layering:**
- `adversarial_attack.py` is the orchestration layer — it owns the benchmark loop, artifact lifecycle, and metadata.
- `agent/selector_policy.py` defines the `SelectorPolicy` protocol and three concrete implementations (`ReactSelectorPolicy`, `RandomSelectorPolicy`, `RLBanditSelectorPolicy`).
- `agent/react_selector.py` contains `ReactTacticSelector`, which is the LLM call inside `ReactSelectorPolicy`. It formats a prompt, calls the selector model, and parses the response into a `TacticRegistryEntry`.
- `judge/red_teaming_tactics.py` has three entry points: `apply_tactic()` (renders `review_artifact` from `executable_code`), `build_tactic_generation_prompt()` (system prompt steering target LLM during generation), and `generate_tactic_artifact()` (generates the attack text itself).
- `utils/code_extraction.py` normalizes raw LLM completions into `executable_code` before syntax validation and test execution.
- `utils/syntax_validator.py` runs tree-sitter on `executable_code` before every deterministic test run. Invalid syntax aborts execution and is recorded as a penalized attempt.

**Shared judge/selector model**: when `judge_model` and `selector_model` resolve to the same backend in `agent_based_decision` + `react` mode, `adversarial_attack.py` creates a single `get_model()` instance shared by both `LLMJudge` and `ReactSelectorPolicy`. Adding a new policy or judge must preserve this sharing opportunity.

**Bandit weight persistence**: `weights/mbpp_ucb1.json` stores `pull_counts` and `cumulative_rewards` keyed by `tactic_id`. Pass `--bandit_weights_path weights/mbpp_ucb1.json` to resume from saved weights. Pass `--bandit_freeze_weights True` to score without updating weights.

## Data and weights

Keep the pipeline open-source friendly. If datasets or learned weights are published, plan for Hugging Face storage so they can be updated over time without changing the core benchmark semantics.

## How to run

Use Inspect with the task entrypoint:

```bash
inspect eval V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/llama3.1:8b \
  -T benchmark=mbpp \
  -T mutation_strategy=react \
  -T policy_mode=agent_based_decision \
  -T experiment_mode=iterative \
  -T use_llm_judge=True \
  -T judge_model=ollama/llama3.1:8b \
  -T selector_model=ollama/llama3.1:8b \
  -T max_iterations=2 \
  --max-samples 2 \
  --limit 2
```

Key task parameters (`-T`):

| Parameter               | Values                                               | Notes                                                                          |
| ----------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------ |
| `benchmark`             | `mbpp`, `humaneval`                                  |                                                                                |
| `policy_mode`           | `random_choice`, `agent_based_decision`, `rl_bandit` |                                                                                |
| `experiment_mode`       | `one_shot`, `iterative`                              |                                                                                |
| `use_llm_judge`         | `True`, `False`                                      |                                                                                |
| `forced_tactic`         | any `tactic_id`                                      | Pins one tactic for isolation testing                                          |
| `bandit_weights_path`   | **absolute** file path                               | Must be absolute — Inspect changes cwd before task setup; use `os.path.abspath` |
| `bandit_freeze_weights` | `True`, `False`                                      | Score without updating weights                                                 |
| `bandit_algorithm`      | `ucb1`                                               | Only UCB1 is supported                                                         |
| `experiment_split`      | `train`, `validation`, `test`                        | Recorded in run metadata; use with `split_definition`                          |
| `split_definition`      | e.g. `mbpp:70_15_15:219-257`                         | Encodes benchmark, strategy, and sample range                                  |
| `selector_use_cot`      | `True`, `False`                                      | Enables chain-of-thought prompting in `agent_based_decision`                   |

After running experiments, aggregate and plot:

```bash
# Aggregate results from all runs in results/
python V3/scripts/aggregate_results.py

# Generate plots from aggregated data (output goes to plots/)
python plot.py
```

`results/` is gitignored; `plots/` is committed.

### Full experiment pipeline (paper results)

Run all three policies end-to-end (RL train→val→test, random, agent, aggregate, plot) with a single script. Must be run from the project root (`Adversarial_Attacks_on_Code_Large_Language_Models_Using_Reinforcement_Learning/`):

```bash
# Full run (5 RL epochs, all samples)
bash V3/scripts/run_full_experiment.sh

# With options
bash V3/scripts/run_full_experiment.sh --benchmark humaneval --epochs 5

# Smoke test (caps every phase to 5 samples, 1 epoch)
bash V3/scripts/run_full_experiment.sh --samples 5 --epochs 1

# Background (logs go to logs/experiment/)
nohup bash V3/scripts/run_full_experiment.sh > /dev/null 2>&1 &
```

### RL-only overnight training

```bash
bash V3/scripts/train_rl_overnight.sh
bash V3/scripts/train_rl_overnight.sh --benchmark humaneval --epochs 5
bash V3/scripts/train_rl_overnight.sh --samples 50   # cap train samples (quick test)
```

Weight checkpoints are saved per epoch (`weights/mbpp_ucb1_epoch_001.json`, …) so interrupted runs preserve progress.

### Benchmark split sizes (70/15/15)

| Benchmark  | Total | Train (1–N) | Val range | Test range |
| ---------- | ----- | ----------- | --------- | ---------- |
| MBPP       | 257   | 179         | 180–218   | 219–257    |
| HumanEval  | 164   | 115         | 116–139   | 140–164    |

## Running tests

```bash
# Full test suite
PYTHONPATH=V3 python -m pytest tests/ -v

# Single test file
PYTHONPATH=V3 python -m pytest tests/test_code_extraction.py -v
```

## Results

Each run is written to `results/<run_id>/` with:

- `run_config.json` for the exact configuration,
- `run_summary.json` for aggregated metrics,
- `attempts.jsonl` for attempt-level records.

The current implementation stores:

- run metadata like benchmark, policy mode, experiment mode, split settings, model names, limits, seed, and git commit when available,
- summary metrics like attack success rate, invalid attempt rate, average iterations to success, average LLM confidence, stop reasons, and arm-level reward accounting,
- per-attempt fields like iteration, tactic ids, test and LLM judge decisions, syntax validity, reward info, bandit state, selector reasoning, and compact artifact summaries.

## Reward and bandit

Current reward components:

- attack success: `+1.0`
- syntax invalid: `-1.0`
- blocked invalid attempt: `-0.5`
- iteration cost: present in the schema, currently `0.0`

For `rl_bandit`:

- the algorithm is `ucb1`,
- arms track pulls and cumulative reward,
- unpulled arms are selected first,
- `record_outcome()` updates arm stats unless weights are frozen,
- optional weight persistence uses a JSON file.

## Non-negotiable constraints

- deterministic tests are ground truth,
- do not change benchmark success semantics,
- do not bypass the selector abstraction,
- do not hardcode model-specific logic in the benchmark loop,
- do not merge `raw_completion`, `executable_code`, and `review_artifact`,
- do not assume bandit support beyond `ucb1` unless the code is changed,
- do not silently change benchmark names or split semantics.

Do not touch:

- `results/` outputs produced by benchmark runs,
- benchmark data files or generated evaluation artifacts,
- persisted run folders under `results/<run_id>/`,
- any files outside `V3/` unless the task explicitly requires it.
- anything that has to do with gitea for now, since that is an experimental path.

## How to work in this repo

1. read the relevant files first,
2. make the smallest change that solves the request,
3. keep changes local to the owning abstraction,
4. avoid scope creep,
5. summarize what changed after implementation.

## Build and Validate

Run this smoke test **before and after every code change** and do not declare the change complete until both runs pass.

**Smoke test command** (Windows; on Unix/macOS replace `.venv/Scripts/inspect` with `.venv/bin/inspect`):
```bash
PYTHONPATH=V3 .venv/Scripts/inspect eval V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/llama3.1:8b \
  -T benchmark=mbpp \
  -T mutation_strategy=react \
  -T policy_mode=agent_based_decision \
  -T experiment_mode=one_shot \
  -T use_llm_judge=True \
  -T judge_model=ollama/llama3.1:8b \
  -T selector_model=ollama/llama3.1:8b \
  -T max_iterations=1 \
  --max-samples 1 \
  --limit 1
```

**Validation checklist (all five must pass):**
1. A new `.eval` file appears in `logs/`.
2. A new folder appears under `results/<run_id>/`.
3. `results/<run_id>/run_summary.json` — `invalid_attempt_rate` is `0.0`.
4. `results/<run_id>/attempts.jsonl` — at least one attempt has `llm_judge_confidence > 0.0`.
5. Console output — no Python exceptions or tracebacks.

If any check fails after a change, revert or fix before continuing.

## Current thesis focus

The priority is no longer feature expansion. The focus is:

- stabilizing the RL attack pipeline,
- producing reproducible experiments,
- generating measurable benchmark results,
- preparing artifacts for the paper,
- and documenting the methodology clearly.

Priority order: reproducible experiments → measurable RL behavior → clean scientific methodology → useful benchmark outputs → paper-ready artifacts.

Avoid architecture changes or abstractions unless they directly improve experimentation or reproducibility.

## Paper

The thesis paper is in `../pedro-msc-paper/samplepaper.tex` (one level above this directory, in `Tese/pedro-msc-paper/`).

Target writing style: NeurIPS, ICLR, ICML, AAAI, ACL.

Writing guidelines:
- prefer direct scientific writing,
- avoid marketing language,
- avoid overly long sentences,
- avoid semicolons when possible,
- avoid em dashes,
- be explicit and empirical,
- prefer concrete observations over vague claims.

Every generated result should be easy to export into tables for the paper. Every experiment must be reproducible.

## Living notes

Use this file to keep a short record of decisions that matter later.

- Add only confirmed project decisions.
- Add only recurring problems that affect implementation.
- Keep entries short and dated when useful.
- Do not turn this into a full changelog.
- Prefer notes about prompts, tactics, evaluation, persistence, and benchmark rules.
