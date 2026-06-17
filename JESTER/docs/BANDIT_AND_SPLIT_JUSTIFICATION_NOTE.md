# Bandit and Split Justification Note (Conference Scope)

Thesis title (unchanged):
Adversarial Attacks on Code LLMs using Reinforcement Learning

## Decision summary

For the conference timeline, keep:
- bandit baseline: UCB1 under `rl_bandit`
- primary split strategy: `70 / 15 / 15` (train/validation/test)
- fallback split when budget is tight: `80 / 10 / 10`
- minimal split (`80 / 20`) only for constrained ablations where no model-selection loop is performed

This is intentionally a focused methodological justification, not a full literature chapter.

## Literature-backed rationale for bandit choice

### Why UCB1 is appropriate now

UCB1 remains a strong first baseline for a discrete action set because it is:
- simple and deterministic to implement,
- low in tuning burden compared with alternatives,
- backed by finite-time regret guarantees,
- easy to analyze across runs in paper-ready plots.

In this project, each tactic/arm receives scalar reward and pull counts, which matches the classic stochastic bandit framing used by UCB-style methods.

### Alternatives considered

1. Epsilon-greedy:
- Pros: very simple and cheap.
- Cons: sensitive exploration schedule; weaker behavior under short budgets if epsilon is not tuned.

2. Thompson Sampling:
- Pros: often strong empirical performance and robust exploration.
- Cons: requires prior/likelihood choices and calibration details we do not want to overfit under the current deadline.

3. Contextual bandits (for richer state-dependent selection):
- Pros: can outperform context-free bandits when reliable context features exist.
- Cons: requires stable feature engineering for context and careful offline policy evaluation; this is intentionally postponed.

4. Heavier RL (Q-learning, actor-critic, policy gradients):
- Pros: potentially stronger long-horizon adaptation.
- Cons: significantly higher tuning and stability cost, not aligned with the current conference delivery window.

### Why not claim SOTA from this baseline

The goal is a defensible, reproducible baseline for policy comparison (`random_choice`, `agent_based_decision`, `rl_bandit`), not to claim that UCB1 is universally optimal for all non-stationary adversarial settings.

## Literature-backed rationale for split strategy

### Why separate train/validation/test now

Because `rl_bandit` adapts from observed rewards, a held-out `test` split is needed to avoid optimistic reporting from adaptive reuse of evaluation data. A separate `validation` split is useful when choosing configuration details (for example, whether to reuse weights, or split-specific orchestration choices).

### Split options considered

1. `70 / 15 / 15`:
- Best balance for this stage.
- Keeps enough train volume for bandit learning while preserving non-trivial validation and test sets.
- Recommended default for conference runs.

2. `80 / 10 / 10`:
- Useful when total sample budget is small and training signal is weak.
- Slightly weaker confidence on validation/test estimates due to smaller held-out partitions.

3. `80 / 20` (no validation):
- Acceptable only for simple ablations where no separate model-selection decision is made.
- Not preferred for main paper claims when adaptive choices are present.

## What is intentionally postponed

To keep delivery realistic before the conference deadline, we postpone:
- contextual-bandit feature modeling,
- non-stationary bandit variants (discounted/sliding-window methods),
- broad algorithm sweeps across many RL families,
- exhaustive sensitivity studies over split ratios.

## Practical recommendation for this repository

- Keep UCB1 as the default first RL selector baseline.
- Use `70 / 15 / 15` as conference default split.
- Allow `80 / 10 / 10` when run budgets are constrained.
- Use `80 / 20` only for limited, clearly labeled ablations.

This supports defensible methodology without blocking implementation velocity.

## References (focused)

1. Auer, P., Cesa-Bianchi, N., Fischer, P. (2002). Finite-time Analysis of the Multiarmed Bandit Problem. Machine Learning.
2. Lattimore, T., Szepesvari, C. (2020). Bandit Algorithms. Cambridge University Press.
3. Thompson, W. R. (1933). On the Likelihood that One Unknown Probability Exceeds Another in View of the Evidence of Two Samples. Biometrika.
4. Li, L., Chu, W., Langford, J., Schapire, R. E. (2010). A Contextual-Bandit Approach to Personalized News Article Recommendation. WWW.
5. Agarwal, A., Hsu, D., Kale, S., Langford, J., Li, L., Schapire, R. (2014). Taming the Monster: A Fast and Simple Algorithm for Contextual Bandits. ICML.
6. Cawley, G. C., Talbot, N. L. C. (2010). On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation. JMLR.
7. Kohavi, R. (1995). A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection. IJCAI.
