# Claude Code Tasks

Use these as paste-ready prompts for Claude Code.

---

# Global Rules

- Read the relevant files before making changes.
- Keep the architecture simple and reproducible.
- Prefer local changes over abstractions.
- Do not silently change benchmark semantics.
- Do not change attack success criteria unless explicitly requested.
- Keep attack prompts syntax-safe and deterministic.
- Avoid unnecessary prompt verbosity.
- Preserve separation between:
  - `raw_completion`
  - `executable_code`
  - `review_artifact`
- Every experiment must be reproducible.
- Every generated result should be easy to export into tables for the paper.
- Prefer implementations that help produce measurable scientific results.

---

# Current Thesis Focus

The priority is no longer feature expansion.

The focus now is:
- stabilizing the RL attack pipeline,
- producing reproducible experiments,
- generating measurable benchmark results,
- preparing artifacts for the paper,
- and documenting the methodology clearly.

The paper is being written in:
- `paper/pedro-msc-paper`

The target writing style should resemble papers from:
- NeurIPS
- ICLR
- ICML
- AAAI
- ACL

Writing guidelines:
- Prefer direct scientific writing.
- Avoid marketing language.
- Avoid overly long sentences.
- Avoid semicolons when possible.
- Avoid em dashes.
- Be explicit and empirical.
- Prefer concrete observations over vague claims.

---

# [ ] Task 1 — Prepare Experiment Pipeline for Paper Results

```text
Read the current benchmark and RL execution pipeline first.

The goal is to prepare the system for reproducible experiments that can directly support the paper results section.

Before changing code:
- identify where experiments start,
- where seeds are defined,
- where attack success is computed,
- where outputs/results are stored,
- and what metrics are already available.

Then implement only the minimum changes needed to:
- save reproducible run metadata,
- export results in a paper-friendly format,
- compare runs consistently,
- and avoid losing experiment artifacts.

The system should remain simple.

Do not redesign the architecture.
```

---

# [ ] Task 2 — Improve Result Logging for Scientific Analysis

```text
Inspect how benchmark results are currently stored.

The current output is not sufficiently structured for scientific analysis and paper tables.

I want the smallest practical improvement that allows:
- comparing attacks,
- comparing tactics,
- comparing RL vs non-RL behavior,
- tracking success/failure causes,
- and generating tables later.

Focus on:
- reproducibility,
- clean logging,
- and minimal architecture changes.

Avoid overengineering.
```

---

# [ ] Task 3 — Hugging Face Dataset and Weight Persistence

```text
Investigate how to publish:
- benchmark datasets,
- successful attack prompts,
- failed prompts,
- RL weights,
- and experiment metadata

using Hugging Face in a clean and updateable way.

Before changing code:
- identify where datasets enter the pipeline,
- where RL/bandit weights are persisted,
- and what serialization formats are currently used.

Then:
- either implement the minimum viable integration,
- or provide a short technical proposal if implementation is premature.

The solution must remain compatible with the current project structure.
```

---

# [ ] Task 4 — RL Evaluation Methodology Research

```text
Do brief research about evaluation methodologies in reinforcement learning papers.

Focus specifically on:
- whether k-fold cross validation is commonly used in RL,
- why it is or is not used,
- and what evaluation strategies are considered standard.

Return:
- a concise Portuguese summary,
- maximum 10 lines,
- focused only on methodological relevance for this thesis.
```

---

# [ ] Task 5 — Prepare Scientific Analysis of RL Behavior

```text
Inspect the current RL training loop and selector policy behavior.

The goal is to understand and later explain scientifically:
- how the RL agent evolves,
- how tactics are selected,
- how rewards influence future choices,
- and whether learning is actually occurring.

I want:
- minimal instrumentation,
- interpretable metrics,
- and outputs that can later support plots/tables in the paper.

Do not add unnecessary complexity.
```

---

# [ ] Task 6 — Generate Initial Paper Draft Structure

```text
Read the LaTeX paper structure in:
- `paper/pedro-msc-paper`

Then prepare an improved scientific draft structure for:
- Abstract
- Introduction
- Related Work
- Methodology
- Experiments
- Results
- Discussion
- Limitations
- Conclusion

Important:
- the RL methodology section is extremely important,
- the paper must clearly justify why RL is being used,
- and the experimental methodology must be reproducible.

Do not generate filler text.

Prefer:
- placeholders,
- section scaffolding,
- TODO markers,
- and concrete structure suggestions.
```

---

# [ ] Task 7 — Create Claude Writing Skills for Scientific Style

```text
Read:
- https://code.claude.com/docs/en/skills

Then create reusable Claude skills/guidelines for scientific writing in this repository.

The goal is to help future paper writing follow a consistent style inspired by:
- NeurIPS
- ICLR
- ICML
- AAAI
- ACL

The writing guidelines should encourage:
- concise scientific writing,
- direct claims,
- empirical reasoning,
- methodological clarity,
- and readable paragraph structure.

Avoid:
- exaggerated claims,
- marketing language,
- semicolons,
- and overly verbose prose.

Keep the skills practical and reusable.
```

---

# Priority Reminder

The thesis priority is:

1. reproducible experiments,
2. measurable RL behavior,
3. clean scientific methodology,
4. useful benchmark outputs,
5. and paper-ready artifacts.

Avoid unnecessary architecture changes or abstractions unless they directly improve experimentation or reproducibility.