# Thesis Project Copilot Operating Prompt

You are assisting on a **master's thesis project** about **adversarial attacks on Code Large Language Models (Code LLMs)** using **iterative attack loops**, **LLM-guided tactic selection**, and later **Reinforcement Learning / Multi-Armed Bandits**.

Your job is to help implement and refine the project **without hallucinating architecture, changing validated assumptions, or skipping phases**.

Read this entire file before making any code changes.

Also read and follow:
- `AGENTS.md`
- `DATA_CONTRACT_ARQUITECTURE.md`

These documents are the project ground truth. Do not contradict them unless explicitly instructed.

---

## 1. Non-Negotiable Execution Rules

### 1.1 Read First, Then Act
Before changing any code:
1. Read this file fully.
2. Read `AGENTS.md`.
3. Read `DATA_CONTRACT_ARQUITECTURE.md`.
4. Inspect the relevant source files for the current task.
5. Summarize your understanding of the task and the likely source of the issue.
6. Only then propose or apply changes.

Do not start editing code before understanding the current implementation.

---

### 1.2 One Task at a Time
You must work in a **strictly sequential** way.

- Do **only one task** from the checklist at a time.
- Do **not** start future tasks early.
- Do **not** bundle multiple checklist items into one large refactor.
- After finishing a task, **stop** and report back.

You must wait for validation before moving to the next task.

---

### 1.3 Do Not Invent New Architecture
This project already has a defined architecture.

Do **not**:
- redesign the whole system,
- replace core abstractions without being asked,
- introduce unnecessary frameworks,
- simplify away important metadata,
- merge components that are intentionally separated,
- change research definitions casually.

Preserve the current architecture and improve it incrementally.

---

### 1.4 Preserve Research Validity
This is not just a software project. It is also a research project.

That means:
- implementation choices must preserve experimental validity,
- metrics must reflect what actually happened,
- logs/metadata must remain trustworthy,
- tactic selection, tactic execution, and recorded tactic history must stay consistent,
- attack outcome definitions must not be changed unless explicitly requested.

Never make “convenience” changes that would weaken the thesis evaluation.

---

### 1.5 Explain Before and After
Before making changes, state:
- what files are relevant,
- what you believe the problem is,
- what you are going to change,
- what you are **not** going to change.

After making changes, state:
- what changed,
- why it changed,
- how it aligns with the architecture docs,
- what risks or limitations remain.

Then stop.

---

### 1.6 Keep Changes Minimal and Targeted
Prefer:
- small, controlled changes,
- local fixes,
- minimal refactors,
- preserving behavior outside the task scope.

Avoid:
- wide unrelated cleanup,
- style-only rewrites,
- renaming large parts of the codebase for no reason,
- changing working parts of the system “just because”.

---

### 1.7 Never Mark a Phase as Complete Unless It Is Truly Stable
Some project phases are implemented but still need stabilization.

Do not claim something is “done” just because code exists.
A phase is only considered stable when:
- behavior matches the documented contract,
- metadata is trustworthy,
- outputs are analyzable,
- known bugs affecting validity are fixed.

---

## 2. Project Mission

The current thesis project aims to study and implement a framework for **adversarially attacking Code LLM evaluation pipelines**, especially scenarios involving:

- a **target model** generating code,
- a **Test Judge** providing ground-truth execution-based evaluation,
- an **LLM Judge** providing model-based evaluation,
- a **looping attack process** that iteratively selects and applies red-teaming tactics,
- later extension to **RL / bandit-based tactic selection**.

The current immediate focus is **not RL yet**.

The current focus is to **stabilize and validate the existing iterative ReAct-based system** so it is trustworthy enough to support thesis experiments and later RL work.

The key success condition for this phase is:

- **attack success** = `Test Judge = FAIL` and `LLM Judge = PASS`

This means the generated code must fail deterministic tests while the LLM-as-judge still approves it.

---

## 3. Current Validated Understanding of the Project

This section is the working ground truth unless explicitly updated.

### 3.1 Core Stack
The project currently uses:
- `inspect_ai` for orchestration and agent loops,
- `inspect_evals` patterns where relevant,
- `LiteLLM` as the model abstraction layer,
- Docker sandboxing for code execution,
- MBPP as the current benchmark,
- HumanEval as a possible later benchmark.

---

### 3.2 High-Level Architecture
The system is intentionally separated into:
- **orchestration logic**,
- **selection logic**,
- **attack generation logic**,
- **evaluation logic**,
- **execution logic**,
- **metadata / analysis outputs**.

This separation must be preserved because the selector should later be replaceable by an RL / bandit policy without requiring the rest of the pipeline to be redesigned.

---

### 3.3 Judge Roles

#### Test Judge
- Based on real code execution and benchmark tests.
- This is the **ground truth** for code correctness.

#### LLM Judge
- A language-model-based evaluator.
- This is intentionally a target for manipulation / red-teaming.

Do not merge or blur these roles.

---

### 3.4 Current Attack Framing
The project studies adversarial attacks in iterative settings where the system tries to produce or frame candidate solutions so that the LLM-based evaluator is misled.

A key success condition currently used in the architecture is:

- **attack success** = `Test Judge = FAIL` and `LLM Judge = PASS`

This must not be changed unless explicitly instructed.

Note: this is conceptually different from other evaluation framings such as performance degradation from a clean baseline. Do not silently mix these concepts.

---

### 3.5 Existing Strategies
The project currently includes three strategy families:
- `random`
- `sequential`
- `react`

The current phase is centered on **stabilizing `react`**, not replacing it.

For the current thesis framing, the `react` strategy must be treated as:
- **LLM-led tactic selection**,
- with the selector seeing the current objective, prior judge feedback, tactic history, and useful metrics,
- and with the system executing a **fresh attack attempt** based on the valid tactic chosen by the LLM from the closed tactic set.

This means the current `react` strategy is **not** meant to enforce exploration externally unless explicitly requested as a separate experimental condition.

---

### 3.6 Current Tactic Space
The current tactic space should remain **small and controlled** during this phase, ideally **3-4 tactic families**, while keeping the architecture extensible so more can be added later.

The current tactic set may continue to use the existing project tactics and/or a selected subset inspired by red-team taxonomy ideas, but for this phase it must remain:
- explicit,
- enum-based,
- documented,
- easy to analyze by family.

Treat the tactic set as a **closed family list** for the current experiment configuration.
Within each family, however, the agent must be able to generate **new attack messages dynamically per attempt**.

Do **not**:
- turn tactic selection into unconstrained free-form behavior,
- let the agent invent entirely new family labels at runtime,
- treat one static prewritten payload as the full meaning of a tactic family.

---

### 3.7 Dynamic Attack Generation Requirement
A tactic family is **not** the same thing as a single fixed attack string.

For the current thesis framing:
- the selector chooses a tactic family,
- the system then performs a **fresh attack attempt** within that family,
- the attack content must be **generated dynamically for that attempt**,
- the generated message may use prior trajectory information, judge feedback, and the tactic family definition,
- repeated use of the same tactic family is allowed if the selector chooses it again,
- but each attempt should still be able to produce a different attack message.

The agent must be the creator of the attack message, not merely the chooser of a prewritten payload.

---

### 3.8 Current Phase Reality
Although the ReAct-based phase is implemented, it must be treated as:

> **implemented but not fully stabilized**

This means the code exists, but the priority now is to ensure:
- correct loop behavior,
- correct stop behavior,
- correct feedback usage,
- correct tactic selection,
- dynamic per-attempt attack generation,
- correct tactic persistence in metadata/history,
- trustworthy experiment outputs,
- inspectable step-by-step trajectories.

Do not jump ahead to RL before this is stable.

---

### 3.9 Future Realistic Target
Beyond MBPP, the project may later move toward a more realistic target environment involving:
- PR review flows,
- CI-style approval workflows,
- reviewer agents,
- decomposition/execution tooling,
- richer trajectory viewers.

That future direction matters for design choices now, but the current implementation focus remains:
- stabilizing the benchmark loop,
- improving observability,
- preserving architecture compatibility for later realistic targets.

---

## 4. Current Known Problems and Constraints

These are currently important and must be respected.

### 4.1 The Current Tactic-Selection Flow Is Conceptually Wrong
The current implementation direction has drifted into an unsustainable pattern where:
- the agent scores or comments on prior content,
- chooses a tactic label,
- and then applies that label in a way that is not clearly tied to a fresh attack attempt over the actual candidate solution being judged.

This must be corrected.

The correct conceptual flow is:
1. generate or obtain the current candidate code,
2. run deterministic tests,
3. run the LLM judge,
4. if attack success has **not** been achieved, summarize prior trajectory state,
5. choose the next tactic family,
6. generate a **fresh attack message** inside that tactic family,
7. produce the next attack attempt,
8. re-evaluate with Test Judge and LLM Judge,
9. update state and continue until success or stop.

The tactic must influence a fresh attack attempt inside the iterative evaluation loop.

---

### 4.2 ReAct Loop Must Iterate While Failing
The loop must continue iterating while failure conditions still justify another attempt, subject to explicit stop conditions.

The loop must:
- use previous feedback,
- select a tactic based on prior output / prior attempt state,
- generate a new attack message for that attempt,
- re-evaluate,
- update state,
- decide whether to continue.

For the current phase, the tactic choice inside `react` should remain **controlled by the LLM** whenever the selector returns a valid tactic from the closed tactic set.

Repeated tactics are allowed if the LLM selects them again.
The history of previous tactics exists to inform the LLM, not to hard-force diversity unless that behavior is explicitly being studied as a separate strategy.

This behavior is central to the project.

---

### 4.3 Infinite Loop Prevention Matters
The loop must not run indefinitely.

The implementation must include clear and enforceable stop conditions such as:
- max iteration count,
- attack success reached,
- no useful progress,
- repeated / exhausted tactic behavior if appropriate,
- failure conditions that justify stopping.

Do not weaken loop safety.

For the current thesis methodology, `max_iterations` should be treated as the main external loop budget provided by the user/configuration.
Do not introduce additional stopping rules that effectively replace `max_iterations` as the primary budget control unless those rules are explicitly justified, documented, and requested.

In particular, do not stop the `react` loop merely because the same tactic was chosen multiple times, or merely because all tactic labels have appeared at least once, unless that behavior is explicitly part of a defined experimental condition.

---

### 4.4 Tactic History / Used-Tactics Tracking Is Known to Be Unreliable
There is a known bug where the tactic used in practice does not always match the tactic recorded in the logs / history.

Example symptom:
- selector chooses one tactic,
- execution appears to use that tactic,
- recorded tactic list stores a different tactic.

This is a critical issue because it undermines:
- metric validity,
- analysis by tactic,
- reproducibility,
- future RL reward correctness.

Any task related to tactic selection or metadata must preserve strict consistency across:
1. tactic chosen,
2. tactic family applied,
3. attack message generated,
4. tactic recorded.

---

### 4.5 Static Attack Payloads Are Not Acceptable
A tactic family cannot be implemented as a single static hardcoded attack payload that is reused forever.

This is a critical missing feature.

The updated system must move toward:
- family-level tactic selection,
- dynamic per-attempt attack generation,
- variation within a tactic family,
- inspectable reasoning about why a new attempt was generated.

---

### 4.6 Agent Decomposition Is Still Too Implicit
The loop currently needs clearer decomposition into understandable subtasks.

The system should make it easier to inspect what the agent is doing at each step, for example:
- assess latest attempt,
- summarize prior results,
- choose tactic family,
- generate attack content,
- submit attack attempt,
- evaluate,
- decide whether to continue.

This does **not** mean overengineering the current implementation.
It means making the loop more understandable, inspectable, and suitable for experiment analysis.

---

### 4.7 Observability Is Not Yet Rich Enough
The project needs better per-step trace detail and inspectability.

The current system should move toward richer trajectory analysis support, including:
- clearer attempt-level metadata,
- step boundaries inside each iteration,
- selector input visibility,
- selector output visibility,
- generated attack message visibility,
- judge outcomes,
- stop reasons,
- token or role-level detail where practical,
- compatibility with later log viewers and richer trajectory tooling.

Do not sacrifice observability for convenience.

---

### 4.8 LiteLLM Should Stay as the Abstraction Layer
The project should remain compatible with stronger external models through `LiteLLM`.

This matters because weak local models (for example via Ollama) may underrepresent realistic behavior and may be too weak for meaningful judge or selector evaluation.

However, model backend expansion is **not the current first priority**. It comes after stabilization of the existing loop and metrics integrity.

---

### 4.9 Tool-Based Agents May Be Explored Later
There is interest in exploring agent tool usage patterns such as:
- `decompose`
- `submit`

within an inspect-style loop, potentially with constrained tool access.

This is a later controlled exploration area, not the first implementation priority.

Do not prematurely redesign the current loop around tools unless the task explicitly asks for it.

---

### 4.10 Selector Authority in the Current Phase
For the current `react` strategy, the selector should behave as an **LLM decision-maker**, not as a thin wrapper around hidden heuristic control.

That means:
- the LLM should know the attack objective,
- the LLM should know what has already been tried,
- the LLM should see judge feedback and useful metrics,
- the LLM should choose among the allowed tactic families,
- and the system should execute that valid choice rather than silently substituting another tactic for statistical or heuristic reasons.

The system may still validate that the choice belongs to the closed family enum.
But validation is different from overriding the decision.

---

### 4.11 Future PR / CI Realism Must Influence Design, Not Current Scope
A more realistic future target may involve:
- PR review,
- CI outcomes,
- reviewer agents,
- manipulative approval workflows,
- detailed agent tool traces.

Current tasks should preserve compatibility with that direction, but must not derail the present benchmark stabilization work.

---

## 5. What Must Not Be Changed Right Now

Unless explicitly instructed, do **not** do any of the following:

- Do not change the definition of attack success.
- Do not turn tactic family selection into unconstrained free-form runtime labels.
- Do not skip directly to RL / bandit implementation.
- Do not redesign the orchestration layer.
- Do not merge selection and attack-generation logic into one opaque blob.
- Do not remove metadata fields that help reconstruct what happened.
- Do not reduce observability of attempts.
- Do not rewrite large working parts of the project for style reasons.
- Do not claim new metrics are correct unless they trace real execution.
- Do not silently alter benchmark assumptions.
- Do not change current phase goals from “stabilize and validate” to “invent new capability”.
- Do not override a valid LLM-chosen tactic family with a different family unless explicitly instructed to study a constrained variant.
- Do not treat tactic history as a hard rule that forbids repetition inside `react` unless explicitly instructed.
- Do not replace `max_iterations` with hidden effective limits such as tactic-count exhaustion unless explicitly instructed and documented.
- Do not keep static prewritten attack payloads as the final intended design for a tactic family.
- Do not treat future PR/CI realism as permission to skip current MBPP stabilization work.

---

## 6. Current Priority Order

Work in this order unless explicitly told otherwise.

### Priority Block A — Correct and Stabilize the Existing ReAct Loop
1. Correct the conceptual iterative flow so the chosen tactic affects a fresh attack attempt.
2. Fix tactic family selection / tactic persistence inconsistencies.
3. Ensure stop conditions are explicit and reliable.
4. Ensure previous feedback and metrics actually influence next tactic selection.
5. Introduce dynamic per-attempt attack generation inside tactic families.
6. Make loop decomposition clearer and inspectable.

### Priority Block B — Consolidate Outputs for Thesis Analysis
7. Ensure metadata reflects real execution.
8. Ensure attempt-level outputs are analyzable by tactic, message, and iteration.
9. Improve per-step trace detail and observability.
10. Refactor minimally where necessary to improve separation and reliability.
11. Improve or add focused documentation for the stabilized current system.

### Priority Block C — Controlled Expansion
12. Improve practical model backend usage through LiteLLM / stronger external APIs.
13. Explore inspect-compatible tool-based agent patterns.
14. Prepare for more realistic PR / CI style targets without breaking the benchmark path.
15. Only after all of the above, prepare for RL / bandit selector work.

---

## 7. Definition of Done for the Current Phase

The current phase is only considered sufficiently stabilized when all of the following are true:

- The iterative `react` loop behaves according to the documented contract.
- The loop continues when it should and stops when it should.
- Tactic family selection uses previous attempt feedback and useful metrics.
- The chosen tactic family, generated attack message, executed attempt, and recorded metadata are consistent.
- Attack generation is dynamic within a tactic family rather than fixed to a single payload.
- Attempt metadata is trustworthy.
- The system supports meaningful analysis by tactic family, generated message, iteration, and judge outcome.
- The architecture still cleanly supports future replacement of the selector by an RL policy.
- No unnecessary redesign was introduced.
- The `react` selector is still recognizably LLM-led rather than silently dominated by hidden heuristics.
- `max_iterations` remains the primary external upper bound for the loop unless a separately documented condition says otherwise.
- The system is more inspectable at the level of loop steps and trajectory state than it was before this phase started.

---

## 8. Required Working Style for Every Task

For each task you must follow this exact pattern.

### Step A — Inspect
Read the relevant files and explain:
- which files matter,
- how the current implementation works,
- where the issue likely lives,
- what invariant or contract is being violated.

### Step B — Plan Only the Current Task
State:
- the smallest change set that can solve the problem,
- what behavior should remain unchanged,
- what evidence will show the fix is correct.

### Step C — Implement Only That Task
Apply only the scoped changes needed.

### Step D — Report
At the end, provide:
- changed files,
- summary of code changes,
- why the change matches `AGENTS.md` and `DATA_CONTRACT_ARQUITECTURE.md`,
- any remaining uncertainty,
- confirmation that you are stopping and waiting for validation.

---

## 9. Expected Output Format After Each Task

After each completed task, respond in this structure:

### Files inspected
- list the files you read

### Problem identified
- concise explanation of the actual issue

### Changes made
- concise list of exact changes

### Why this is correct
- explain how the fix aligns with the current architecture and research constraints

### What was intentionally not changed
- list nearby things you deliberately left untouched

### Remaining risks or follow-ups
- note any limitations or next checks

Then stop.

---

## 10. Strict Sequential Checklist

Follow this checklist in order. Do not skip ahead.

---

## Task Checklist (Mark each task as [x] only after approval)

### [ ] Task 1 — Correct the Iterative ReAct Flow Around Fresh Attack Attempts
**Goal:** fix the conceptual loop so the selected tactic family influences a fresh attack attempt inside the iterative evaluation cycle.

**Delivery:**
- identify where the current flow is decoupled from the actual candidate code / judged attempt,
- restructure the loop so each iteration clearly performs:
  1. evaluate current attempt,
  2. summarize relevant prior state,
  3. choose next tactic family,
  4. generate the next attack attempt,
  5. re-evaluate,
  6. update trajectory state,
- preserve the current research objective and success condition,
- keep the solution minimal and architecture-compatible.

**Validate:**
- the implementation now reflects the documented iterative contract,
- the selected tactic family is no longer an isolated label disconnected from the actual next attempt,
- the loop remains compatible with current benchmark execution and judges.

Stop after completion.

---

### [ ] Task 2 — Fix Tactic Family Tracking and Recorded History Consistency
**Goal:** ensure that the chosen tactic family, the applied tactic family, and the recorded tactic family always match.

**Delivery:**
- investigate enum handling,
- investigate normalization / mapping issues,
- investigate stale state reuse,
- investigate where history and metadata are appended,
- fix the root cause rather than only the symptom,
- preserve compatibility with the current closed tactic-family set.

**Validate:**
- selector output, execution path, and recorded history are end-to-end consistent,
- no mismatched tactic-family labels appear in attempt history,
- the fix supports trustworthy later analysis by tactic family.

Stop after completion.

---

### [ ] Task 3 — Validate and Harden Stop Conditions
**Goal:** ensure the iterative attack loop cannot run indefinitely and stops for the right reasons.

**Delivery:**
- inspect all current stop conditions,
- verify max-iteration handling,
- verify attack-success handling,
- inspect accidental always-continue logic,
- inspect accidental early-stop logic,
- make stopping behavior explicit and trustworthy without weakening iterative behavior.

**Validate:**
- the loop cannot run forever,
- `max_iterations` remains the primary external budget control unless a documented experimental condition says otherwise,
- the loop does not stop merely because tactic labels were exhausted or repeated unless that behavior is explicitly intended.

Stop after completion.

---

### [ ] Task 4 — Ensure Feedback and Metrics Truly Drive Tactic Family Choice
**Goal:** confirm that tactic-family selection depends on prior attempt outcome rather than behaving randomly, statically, or on shallow context alone.

**Delivery:**
- inspect what exact feedback and metrics are passed into the selector,
- verify whether prior judge outcomes are visible,
- verify whether prior attempts are summarized meaningfully,
- ensure the selector sees enough structured trajectory context to choose intelligently,
- make minimal changes needed to align selection with the intended iterative methodology.

**Validate:**
- the next tactic family is chosen with meaningful awareness of prior results,
- the selector remains LLM-led rather than being silently dominated by hidden heuristics,
- repeated tactic-family choices remain allowed when the LLM selects them again.

Stop after completion.

---

### [ ] Task 5 — Introduce Dynamic Per-Attempt Attack Generation Within Tactic Families
**Goal:** move from static per-tactic payloads to dynamic attack-message generation inside each tactic family.

**Delivery:**
- define the boundary between tactic-family selection and attack-message generation,
- ensure the system generates a new attack message per attempt,
- preserve the closed family list while allowing creativity inside each family,
- keep the implementation extensible so more families can be added later,
- avoid turning the generator into unconstrained free-form family invention.

**Validate:**
- the same tactic family can produce different attack messages across attempts,
- the agent is generating attack content rather than replaying a single hardcoded payload,
- generated messages remain attributable to a known tactic family for analysis.

Stop after completion.

---

### [ ] Task 6 — Clarify Loop Decomposition Into Explicit Agent Steps
**Goal:** make the iterative process more understandable and inspectable by decomposing what the agent does at each step.

**Delivery:**
- identify the major conceptual steps inside one iteration,
- introduce clearer structure or boundaries for those steps where needed,
- make it easier to inspect actions such as summarizing, selecting, generating, submitting, evaluating, and deciding,
- keep the implementation practical and minimal.

**Validate:**
- the loop is easier to reason about and inspect,
- per-step responsibilities are clearer than before,
- the decomposition supports later experiment analysis and possible tool-based evolution.

Stop after completion.

---

### [ ] Task 7 — Validate Attempt-Level Metadata Integrity
**Goal:** ensure attempt-level metadata can reconstruct what actually happened in each iteration.

**Delivery:**
- inspect `all_attempts` and related metadata structures,
- ensure metadata captures iteration number, tactic family, generated attack message or equivalent reference, judge outcomes, stop reason where relevant, and any useful confidence / score information if available,
- remove or avoid fake inferred values that do not reflect real execution,
- preserve compatibility with existing logging where possible.

**Validate:**
- the metadata reflects real execution rather than guesses,
- each iteration can be reconstructed for later analysis,
- tactic-family and attack-message lineage are visible.

Stop after completion.

---

### [ ] Task 8 — Improve Observability and Per-Step Trace Detail
**Goal:** make the system more inspectable for thesis analysis and trajectory viewing.

**Delivery:**
- improve visibility of selector inputs,
- improve visibility of selector outputs,
- improve visibility of generated attack content,
- improve visibility of judge outcomes and stop reasons,
- add trace structure that can later support richer viewers and trajectory analysis,
- keep the implementation compatible with the current inspect-based pipeline.

**Validate:**
- the run output is more useful for debugging and thesis analysis,
- a reviewer can more easily understand what happened at each step,
- observability improvements do not break the core loop.

Stop after completion.

---

### [ ] Task 9 — Minimal Refactor for Separation of Concerns
**Goal:** improve maintainability only where necessary for correctness, observability, and future RL compatibility.

**Delivery:**
- tighten the boundary between selector logic and attack-generation logic,
- tighten the boundary between state update and metadata recording,
- reduce opportunities for state-flow bugs,
- preserve current behavior outside the scoped changes.

**Validate:**
- the current phase is more reliable without broad cleanup,
- public behavior remains aligned with the documented contract,
- selector replacement remains feasible later.

Stop after completion.

---

### [ ] Task 10 — Update Technical Documentation for the Stabilized Current System
**Goal:** document the system as it actually works after stabilization.

**Delivery:**
- document the corrected iterative loop,
- document the tactic-family concept versus dynamic attack-message generation,
- document judge roles and success condition,
- document stop conditions,
- document metadata / trajectory structure,
- document known limitations,
- document why selector and attack generation remain separate.

**Validate:**
- the docs match the implementation after stabilization,
- the docs are useful both for thesis writing and future engineering work,
- future RL work is not documented as already implemented.

Stop after completion.

---

### [ ] Task 11 — Improve Model Backend Readiness Through LiteLLM
**Goal:** make the system more practical for stronger non-Ollama model backends while preserving architecture.

**Delivery:**
- inspect current model configuration assumptions,
- inspect where LiteLLM is already used,
- improve compatibility with stronger free/external APIs where practical,
- keep the benchmark loop and metadata behavior stable.

**Validate:**
- backend flexibility improves without disrupting the stabilized loop,
- the architecture remains consistent,
- this does not turn into RL or tool-agent work.

Stop after completion.

---

### [ ] Task 12 — Explore Tool-Based Agent Pattern in a Controlled Way
**Goal:** explore whether constrained tool patterns such as `decompose` / `submit` fit the architecture.

**Delivery:**
- treat this as controlled exploration only,
- keep a clear separation between the current benchmark pipeline and the experimental tool-based path,
- use the exploration to improve inspectability and decomposition understanding rather than to replace the stabilized loop prematurely.

**Validate:**
- the exploration does not break the current stabilized pipeline,
- the experiment path remains clearly scoped,
- the results are useful for future realistic-agent work.

Stop after completion.

---

### [ ] Task 13 — Prepare the Codebase for PR / CI Style Target Realism
**Goal:** make the design more compatible with future realistic reviewer-agent targets without derailing the current benchmark path.

**Delivery:**
- identify the boundaries needed for future PR review / CI style workflows,
- preserve compatibility with richer reviewer / target / executor decompositions,
- avoid overbuilding the realistic target now,
- keep the benchmark path first-class.

**Validate:**
- the codebase is more ready for future realistic targets,
- current MBPP-based experimentation still works cleanly,
- no unnecessary architecture rewrite was introduced.

Stop after completion.

---

### [ ] Task 14 — Prepare the Codebase for Future RL / Bandit Selector Work
**Goal:** only after stabilization, ensure the selector boundary is truly swappable.

**Delivery:**
- define or confirm clean selector interface assumptions,
- ensure useful state and reward-relevant signals remain observable,
- preserve the corrected separation between family selection and attack generation,
- do not implement RL yet unless explicitly instructed.

**Validate:**
- future selector replacement is more feasible,
- no premature RL logic was introduced,
- the current stabilized loop remains the primary working path.

Stop after completion.

---

## 11. Guidance on Debugging the Known Tactic Bug

Because this bug is especially important, follow these principles when investigating it:

- trace the tactic family from the exact selector output,
- trace the value passed into attack generation,
- trace the generated message linked to that family,
- trace the value written into state/history,
- compare raw value vs normalized value,
- compare enum member vs enum string vs display label,
- inspect whether mutable state is being reused incorrectly,
- inspect whether the wrong variable is appended to history,
- inspect whether history is written before the final chosen family is resolved.

Do not patch only the symptom. Identify the real root cause.

---

## 12. Guidance on Dynamic Attack Generation

When implementing dynamic attack generation:
- preserve a closed tactic-family list,
- allow creativity only inside the selected family,
- keep prompts/family definitions explicit enough to analyze,
- do not hardcode one fixed final attack string per family,
- do not lose attribution from generated message back to family,
- keep room for future family expansion.

The correct abstraction is:

- **family selection** chooses the kind of attack,
- **message generation** creates the concrete attack attempt for that iteration.

Do not collapse these into one opaque undocumented step.

---

## 13. Guidance on Metrics and Analysis Integrity

When working on metrics or metadata, ensure the outputs support later thesis analysis such as:
- success by tactic family,
- success by generated attack pattern where possible,
- judge disagreement,
- confidence shift,
- iteration count to success,
- failure patterns,
- tactic ordering across attempts,
- trajectory evolution across iterations.

Only record values that are actually known from execution.
Do not invent summary values if they are not grounded in real state.

---

## 14. Guidance on Observability

When improving observability, prefer making the trajectory easier to inspect at the level of:
- current iteration,
- selector context,
- selected family,
- generated attack message,
- judged artifact,
- judge outcomes,
- stop decision,
- state update.

Favor traceability over clever compactness.

Where practical, preserve compatibility with richer future trajectory viewers and realistic-agent tooling.

---

## 15. Guidance on Refactoring

Allowed refactoring:
- extracting a helper,
- clarifying a boundary,
- fixing a state-flow problem,
- making metadata assignment less error-prone,
- making selector replacement easier later,
- making step decomposition clearer.

Not allowed refactoring:
- rewriting the project structure,
- changing unrelated modules,
- changing naming conventions globally,
- introducing new abstractions without clear need,
- turning one task into a large cleanup effort.

---

## 16. Guidance on Future RL Work

RL / bandit work is a later phase.

The only acceptable RL-related work during the current phase is:
- preserving selector modularity,
- keeping state and rewards observable,
- avoiding design decisions that would block future policy replacement.

Do not implement RL now unless explicitly instructed.

---

## 17. Final Reminder

Your role is to help stabilize and validate the current thesis system, not to improvise a new one.

The current objective is:

> **Make the existing ReAct-based iterative adversarial pipeline correct, dynamically generative within tactic families, analyzable, and trustworthy enough for thesis experiments and future RL extension.**

Stay disciplined.
Work one task at a time.
Stop after each task.
Do not hallucinate.
Do not skip phases.
Do not break research validity.
