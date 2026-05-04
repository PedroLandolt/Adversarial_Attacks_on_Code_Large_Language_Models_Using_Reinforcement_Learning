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
  prompt.md                    # Paste-ready Claude Code task prompts.
  AGENTS.md                    # Agent instructions specific to the V3 subtree.
  run.sh                       # General shell entrypoint for V3 runs.
  script_copy_past.sh          # Command helper with ready-to-copy run presets.
  __init__.py                  # Package marker for V3.
  agent/
    __init__.py                # Agent package marker.
    react_selector.py          # ReAct-style selector implementation used by policies.
    selector_policy.py         # Shared selector boundary and policy modes.
    tactic_registry.py         # Registry of tactic families and tactic metadata.
    tool_pattern_exploration.py # Experimental tooling for tactic/pattern exploration.
  attacks/
    __init__.py                # Attack package marker.
    instruction_perturbation.py # Attack family that perturbs instructions.
    misleading_comments.py      # Attack family that injects misleading comments.
    variable_renaming.py        # Attack family that renames variables for obfuscation.
    gitea_redteam_taxonomy.py   # Taxonomy helpers for the Gitea-oriented path.
  docs/
    BANDIT_AND_SPLIT_JUSTIFICATION_NOTE.md # Notes on bandit choice and split strategy.
    CONFERENCE_BENCHMARK_NOTE.md # Conference-track benchmark scope and guidance.
    DATA_CONTRACT_ARQUITECTURE.md # Data and architecture contract for the project.
    MBPP_BENCHMARK_FLOW.md      # MBPP-specific flow and evaluation notes.
    RESULTS_AND_PLOTTING_CONTRACT.md # Output and plotting contract for persisted runs.
  gitea/
    __init__.py                 # Gitea package marker.
    schemas.py                  # Data contracts for the Gitea path.
    tools.py                    # Gitea tool bindings and helpers.
  judge/
    __init__.py                 # Judge package marker.
    llm_judge.py                # LLM judge wrapper and scoring logic.
    red_teaming_tactics.py      # Renders tactics into judge-facing artifacts.
    test_judge.py               # Deterministic test judge wrapper.
  scripts/
    aggregate_results.py        # Aggregates persisted run outputs.
    run_adversarial_code_llm.sh # Shell preset for benchmark runs.
    run_adversarial_gitea_react_attack.sh # Shell preset for the Gitea path.
    setup_gitea.sh              # Helper for setting up Gitea-related tooling.
  utils/
    __init__.py                 # Utility package marker.
    bandit_weights.py           # Persistence helpers for bandit weights.
    benchmark_loader.py         # Benchmark loading and normalization helpers.
    results_aggregation.py      # Aggregation helpers for result analysis.
    results_persistence.py      # Run config, summary, and attempt persistence.
    reward_accounting.py        # Reward calculation and arm accounting.
    syntax_validator.py         # Python syntax validation for generated code.
```

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

## Data and weights

Keep the pipeline open-source friendly. If datasets or learned weights are published, plan for Hugging Face storage so they can be updated over time without changing the core benchmark semantics.

## How to run

Use Inspect with the task entrypoint:

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

**Smoke test command:**
```bash
PYTHONPATH=V3 .venv/Scripts/inspect eval V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen3.5:0.8b \
  -T benchmark=mbpp \
  -T mutation_strategy=react \
  -T policy_mode=agent_based_decision \
  -T experiment_mode=one_shot \
  -T use_llm_judge=True \
  -T judge_model=ollama/qwen3.5:0.8b \
  -T selector_model=ollama/qwen3.5:0.8b \
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

## Living notes

Use this file to keep a short record of decisions that matter later.

- Add only confirmed project decisions.
- Add only recurring problems that affect implementation.
- Keep entries short and dated when useful.
- Do not turn this into a full changelog.
- Prefer notes about prompts, tactics, evaluation, persistence, and benchmark rules.
