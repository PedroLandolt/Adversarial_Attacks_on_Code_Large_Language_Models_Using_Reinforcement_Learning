# Reproducing the results

This document records the experimental protocol used for the dissertation, in enough detail
to rerun it. Chapter 6 of [the dissertation](../MSc_PedroLandolt_MESW_202103337.pdf) gives the
full rationale; this is the operational version.

## Environment

| Requirement | Version used |
| --- | --- |
| Python | 3.10+ |
| Docker | Any recent release, running, for sandboxed test execution |
| Ollama | Any recent release, running locally |
| GPU | 12 GB VRAM |

The 12 GB budget is the reason every model is open-weight and in the 7 to 8B range: the
attacker and the judge have to be resident at the same time, which comes to roughly 10 GB.
Results are therefore relative comparisons at that scale, not claims about frontier models.

## Models

| Role | Model | Notes |
| --- | --- | --- |
| Attacker and generator | `llama3.1:8b` | Fixed across every experiment |
| Judge (primary) | `qwen2.5-coder:7b` | Baseline acceptance around 6%, which makes it informative |
| Judge | `deepseek-coder:6.7b` | Informative |
| Judge | `codellama:7b` | Accepts 65% of buggy code unattacked, so there is little headroom to measure |
| Judge | `starcoder2:7b` | Hardest judge; see the caveat at the end |

Keeping the attacker fixed is deliberate. Only the judge changes across the cross-judge
experiments, so any difference is attributable to the judge.

## Datasets

| Corpus | Records | Construction |
| --- | --- | --- |
| `adversarial_code_buggy` | 982 | MBPP and HumanEval problems, bug planted by the generator, then filtered |
| `cubert_wbo` | 1000 | Wrong-binary-operator bugs sampled from the ETH Py150 Open corpus |

Both corpora pass a two-judge filter. A record is kept only when the deterministic tests
reject the function **and** an unframed judge also rejects it. Records the judge already
accepts without any attack are removed, because they would measure baseline permissiveness
rather than the effect of the framing. This is what keeps the baseline attack success rate
low, around 6% on the primary judge, and it is the reason the reported gains are attributable
to the tactics.

The corpora are committed as JSONL under [`datasets/`](../datasets). The primary corpus is
also on [HuggingFace](https://huggingface.co/datasets/PedroLandolt/adversarial-code-buggy)
with baseline judge verdicts included.

## Splits

Each corpus is split 70/15/15 over the full record list, so a problem appears in exactly one
partition.

| Corpus | Total | Train | Validation | Test |
| --- | --- | --- | --- | --- |
| `adversarial_code_buggy` | 982 | 687 | 147 | 148 |
| `cubert_wbo` | 1000 | 700 | 150 | 150 |

The split is selected with `-T experiment_split` and recorded in `run_config.json`, so any
archived run can be traced back to the partition it used.

## Training protocol

Only the bandit trains. Random and ReAct carry no learnable state and go straight to
evaluation, on the same held-out test partition, so the comparison is not confounded by
differing sample counts.

1. **Initialize.** Every arm starts at zero pulls and zero cumulative reward, so the initial
   exploration is unbiased.
2. **One pass over the train split.** With nine arms, 687 training problems, and up to six
   attempts per sample, a pass delivers roughly 160 to 170 pulls per arm. That is enough for
   the ranking to stabilize past the initial exploration phase.
3. **Checkpoint.** Arm state is written to a file keyed by algorithm, corpus, and the
   attacker-judge pair.

Each attacker-judge pair trains its own weight file. Learned preferences are never shared
across pairs, so a difference between judges reflects the judge rather than transfer.

One training pass takes roughly two to three hours on the primary pair, varying by algorithm
(Thompson 2h03, KL-UCB 2h21, UCB1 2h30, EXP3 3h09).

```bash
bash JESTER/scripts/run_rl_train.sh adversarial_code_buggy
```

There is no loss curve. A bandit does not backpropagate; it updates arm means from a scalar
reward. Training progress is tracked through average reward per pull and arm entropy, both
recorded in the run summary.

## Evaluation protocol

Evaluation loads the checkpoint and runs with `bandit_freeze_weights=True`. The policy selects
according to what it learned but applies no updates, so no evaluation feedback reaches the
weights.

The validation split is used to compare algorithm variants. The test split is used only after
training is complete, and it is what the reported numbers come from.

```bash
bash JESTER/scripts/run_rl_eval.sh adversarial_code_buggy
bash JESTER/scripts/run_random.sh  adversarial_code_buggy
bash JESTER/scripts/run_react.sh   adversarial_code_buggy
```

Because the learned state is nine pulls and nine reward sums, and is not a function of the
problem text, the policy cannot memorize samples. Empirically the frozen policy matches or
beats its own training-time success on unseen problems in all 24 configurations, with the
largest excess of training over test at 0.1 pp.

## Per-tactic isolation

Per-tactic numbers come from forced-tactic runs, where the selector is bypassed and one
tactic is pinned for the whole run:

```bash
bash JESTER/scripts/run_forced_tactic.sh
```

These runs report **tactic-only** attack success rate: an attempt counts only when the judge
rejected the raw buggy code but accepted it once the tactic was applied. This isolates the
contribution of the framing. Tactic-only numbers are not comparable with the policy numbers
above, which also credit baseline passes. Comparing the two directly overstates the policy by
several percentage points.

## Experiment order

The reported results come from these conditions, run in this order:

1. Baselines on both corpora, primary judge: no tactic, random, ReAct.
2. Forced-tactic runs, all nine tactics, both corpora, primary judge.
3. Bandit training and frozen-weight evaluation, four algorithms, both corpora, primary judge.
4. Cross-judge repetition of the policy comparison on DeepSeek-Coder and CodeLlama.
5. Attempt budgets of one, three, and six for every policy.

## Aggregation and figures

Aggregation and plotting read persisted results only and never invoke a model, so figures can
be rebuilt from archived runs.

```bash
python JESTER/scripts/aggregate_results.py --results-dir results --output-dir results/aggregates
python results_csv/make_thesis_figures.py
```

[`results_csv/`](../results_csv) holds the curated tables behind the dissertation figures.
`make_thesis_figures.py` is the single source for regenerating them; it reads those CSVs and
writes the figures used in Chapter 7 and Appendix 3. `plot.py` at the repository root is a
separate exploratory tool that works from raw run directories and was not used for the
dissertation figures.

## Caveats

**Single seed, single training pass.** Every reported number comes from one run. The 95%
binomial confidence interval, about ±8 pp at n = 148, bounds sampling error but not
run-to-run inference variance. Repeated trials remain future work.

**StarCoder2 has no bandit result.** Inference on that judge was too slow to complete a
training pass within the available time, so the hardest judge is reported for the random and
ReAct policies only. This is a resource limit, not a negative finding.

**ReAct collapses on StarCoder2.** The 8B selector failed to emit parseable structured output
489 times on that judge, so selection fell back repeatedly to a single weak tactic and ReAct
scored below random. This is a limit of small-model structured output, not evidence against
reasoning-based selection.
