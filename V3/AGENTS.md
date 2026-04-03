# Adversarial Attacks on Code LLMs using Reinforcement Learning

> **Conference alignment update — May 31 paper track**
>
> This document narrows the operational focus of the repository for the current paper deadline.
> The thesis can remain broader, but the active implementation plan for now must prioritize the conference path below.

---

## 1. Current active scope

The active scope for the conference is:

1. **MBPP** as a controlled judge-attack benchmark.
2. **HumanEval** as a second benchmark under the same adapter contract.
3. **Selector policy comparison** across:
   - `random_choice`
   - `agent_based_decision`
   - `rl_bandit`
4. **Offline experiment persistence and plotting**.

### Explicitly out of scope for the conference track

- Gitea workflow experimentation
- rewriting protected `V3/gitea/` files
- sequential RL methods such as SARSA / Q-learning
- full cross-environment generalization claims

Gitea remains part of the broader thesis direction, but it is not part of the current conference execution path.

---

## 2. Core benchmark phenomenon

The target benchmark phenomenon remains:

```text
Attack Success = (Test Judge = FAIL) AND (LLM Judge = PASS)
```

Interpretation:
- deterministic tests are the ground truth,
- the LLM judge is the attack target,
- the current benchmark studies evaluator manipulation.

This condition must remain unchanged in the conference path.

---

## 3. Architecture already assumed by the conference path

The benchmark path should continue to preserve three explicit artifacts:

- `raw_completion`: full model answer,
- `executable_code`: normalized Python code used for deterministic execution,
- `review_artifact`: judge-facing artifact after attack rendering.

Do not collapse these artifacts back into a single ambiguous field.

---

## 4. Policy modes

The conference path must treat selector policy as a first-class experiment variable.

Required policy modes:
- `random_choice`
- `agent_based_decision`
- `rl_bandit`

### Policy intent

- `random_choice`: non-adaptive baseline
- `agent_based_decision`: current LLM-led selector baseline
- `rl_bandit`: first RL baseline for tactic/arm selection

All three must operate through the same selector-facing contract.

---

## 5. Experiment modes

The conference benchmark must support two experiment modes:

- `one_shot`: only first adversarial attempt
- `iterative`: adaptive loop up to configured iteration budget

Experiment mode must be recorded in run metadata and persisted outputs.

---

## 6. Syntax gate

Generated `executable_code` must be checked with **tree-sitter** before deterministic test execution.

If syntax is invalid:
- the attempt must be recorded explicitly,
- deterministic execution must not proceed for that artifact,
- the outcome must not be buried in generic exception handling.

This is required for clean attempt accounting and paper-ready metrics.

---

## 7. Persistence and plotting

Benchmark runs may be slow and expensive.
Therefore the project must persist structured outputs per run so plots can be regenerated offline.

Required direction:
- write stable outputs to `results/`,
- save run config,
- save run summary,
- save attempt-level records,
- support a later `plot.py` path that reads saved results only.

Do not rely only on Inspect-visible metadata for long-term analysis.

---

## 8. Result categories that matter now

The conference path should make it possible to analyze at least:

- attack success rate,
- success rate by benchmark,
- success rate by policy mode,
- success rate by tactic/arm,
- syntax-invalid rate,
- iterations to success,
- arm pull frequency,
- average reward by arm,
- one-shot vs iterative differences.

---

## 9. Working principles

1. Prefer small, validated changes.
2. Keep benchmark and policy boundaries clear.
3. Avoid redesigning the repo just before the conference deadline.
4. Optimize only after correctness, accounting, and persistence are stable.
5. Keep the thesis title unchanged.
6. Keep Gitea out of the conference hot path.
7. Make figures reproducible from saved results.

---

## 10. What contributors should assume by default

When changing this repo for the conference path, assume the following priority order:

1. benchmark correctness,
2. attempt accounting integrity,
3. syntax validation,
4. selector policy comparability,
5. result persistence,
6. plotting and presentation,
7. only then performance polish.

---

## 11. Short operational summary

A contributor working on the current paper should understand the project as follows:

- We are attacking the **LLM judge**, not redefining correctness.
- We compare **three selector policies** on **two benchmarks**.
- We stop early on syntactically invalid artifacts before execution.
- We save every run in a stable format for later plotting.
- We generate paper figures from saved results, not from rerunning experiments.
- We are **not** using Gitea in the current conference track.

---

## 12. References

- Inspect AI: https://inspect.aisi.org.uk/
- ReAct docs: https://inspect.aisi.org.uk/react-agent.html
- Agent pattern reference: https://github.com/rufimelo99/inspect_evals/blob/main/src/inspect_evals/agentharm/agents/default_agent.py
- MBPP: https://github.com/google-research/google-research/tree/master/mbpp
- HumanEval: https://github.com/openai/human-eval
- gitea Example (Rui melo): https://huggingface.co/spaces/rufimelo/github-red-trajectory-viewer
- Model selection: https://github.com/cheahjs/free-llm-api-resources?tab=readme-ov-file#huggingface-inference-providers
- tree-sitter Python grammar: https://github.com/tree-sitter/py-tree-sitter