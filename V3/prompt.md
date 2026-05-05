# Claude Code Tasks

Use these as paste-ready prompts for Claude Code.

## Global rules

- Read the relevant files first.
- Make the smallest change that solves the request.
- Keep changes local to the owning abstraction.
- Validate the touched slice after the first substantive edit.
- Do not change benchmark success semantics.
- Do not bypass the selector abstraction.
- Do not hardcode model-specific logic inside the benchmark loop.
- Do not merge `raw_completion`, `executable_code`, and `review_artifact`.
- Do not silently change benchmark names or split semantics.

## [ ] Task 1 - Strengthen attack generation

```text
Read `V3/adversarial_attack.py`, `V3/agent/selector_policy.py`, `V3/attacks/*`, and the judge logic first.

The current problem is that the LLM generates attack prompts with weak syntax and low consistency, which makes the system fail too often. I want you to strengthen attack generation so that, after RL chooses a tactic, that tactic is always turned by an LLM into a clear, specific, syntax-safe attack instruction.

Make the smallest local change possible without changing benchmark semantics. If needed, improve the prompt/template that generates the attack for each tactic type, but keep these separate:
- `raw_completion`
- `executable_code`
- `review_artifact`

Goal: drastically reduce syntax failures and make each tactic more consistent and reproducible.
```

## [ ] Task 2 - Make tactic prompts robust

```text
Inspect how attack tactics are defined today and identify each tactic family/type that the system supports.

Prepare much more robust and direct prompts for generating the selected tactic. Each attack type must give the LLM enough context to understand:
- the intent of the attack,
- the expected output style,
- what it must not do,
- how to avoid generating extra text that breaks syntax.

The focus is to reduce prompt noise and prevent the LLM from mixing explanation with executable code. I do not want a large or abstract solution; I want something practical, short, and easy for Claude Code to apply with minimal changes.
```

## [ ] Task 3 - Store datasets and weights on Hugging Face

```text
Investigate and prepare a concrete proposal for storing datasets and weights on Hugging Face in an open-source and updateable way.

Before changing code, identify:
- where datasets enter the pipeline,
- where bandit weights or other weights are persisted today,
- what format would be the cleanest for publishing on Hugging Face.

Then implement the minimum integration needed or, if implementation is not yet prudent, return a short technical proposal with the files and functions that would need to change.

Do not complicate the architecture. I want compatibility with the current project state.
```

## [ ] Task 4 - Short note on k-fold in RL

```text
Do only brief research and return a Portuguese text of at most 10 lines.

I want to know whether papers use k-fold in RL, whether they do or not, and why. The goal is to understand whether k-fold makes sense for our evaluation or whether that is not the standard in RL.

Do not write a long review. I want a short, factual answer that is useful for the methodological decision.
```

## [ ] Task 5 - Validate attacks one by one

```text
I want a way to test attacks individually in one-shot mode with a specific attack chosen by us.

Read the current pipeline and implement the smallest possible change to allow:
- manually selecting a specific attack/tactic,
- running one-shot with only that attack,
- analyzing whether the attack makes sense in isolation,
- comparing good and bad attacks easily.

The goal is to validate whether we should keep all current attacks or reduce/change the set. Do not break the normal benchmark flow.
```

## [ ] Task 6 - Diagnose before changing

```text
Before changing anything, read `V3/prompt.md`, `V3/adversarial_attack.py`, `V3/agent/selector_policy.py`, and the persistence files.

The system is failing a lot because of syntax and weak prompts. I want an objective conclusion about:
- where attack generation is going wrong,
- where syntax is being lost,
- which part of the pipeline should be strengthened first.

After that, make only the first minimal change that has real impact.
```

## Priority reminder

The system must stay simple and reliable. If a change increases syntax failures, prompt noise, or unnecessary complexity, simplify it.

When generating or editing attack-related prompts, keep them short, explicit, and syntax-safe. Do not let explanatory text leak into executable code.
