# Prompt for Copilot — Stabilize Inspect-AI Adversarial MBPP Loop

You are working on a Python codebase that uses `inspect_ai` and `inspect_evals.mbpp`.
Your job is to fix the current benchmark loop step by step, with minimal and controlled changes.

The current system is slow, misleading, and produces empty or untrustworthy logs.
Do not invent behavior.
Do not redesign the architecture.
Do not add speculative abstractions.
Do not change unrelated files.
Do not optimize anything before correctness and observability are restored.

The immediate goal is to make the MBPP adversarial benchmark loop correct, inspectable, and fast enough for debugging.
Future-facing Gitea / PR / tool-agent work must not interfere with the immediate benchmark path.

---

## Non-negotiable rules

1. Make changes in small steps.
2. After each task, explain exactly what changed and why.
3. After each task, provide validation steps and expected outcome.
4. Do not silently refactor unrelated code.
5. Preserve current public task names unless a rename is strictly required.
6. Prefer minimal diffs over clever designs.
7. If something is uncertain, inspect the current code and follow the existing data contract instead of inventing a new one.
8. The benchmark loop must prioritize:
   - correctness,
   - deterministic debugging,
   - inspectability,
   - then performance.
9. Do not touch the Gitea / PR workflow unless a task explicitly says so.
10. The `decompose` / `submit` exploration path is not the production MBPP benchmark loop and must remain isolated.

---

## Known problems that must be treated as ground truth

### Problem A — Fake test results
The code currently builds internal `test_result` objects from `output.pass_pred` with a fallback to `False`.
That is not a reliable source of ground-truth MBPP execution.
The benchmark must use real executed test results, not inferred booleans.

### Problem B — Massive over-execution during debugging
The custom task inherits `epochs=base_task.epochs`.
That is multiplying runtime unnecessarily during debugging and making one sample far more expensive than expected.

### Problem C — Artifact / scorer mismatch
The code applies an attack/transformation to a generated artifact, but the object ultimately evaluated or logged by Inspect may not match the attacked artifact.
This causes misleading logs and inconsistent evaluation.

### Problem D — Weak observability
Important intermediate states are stored in a way that is hard to inspect, and logs do not clearly show:
- baseline artifact,
- attacked artifact,
- real test outcome,
- LLM judge outcome,
- stop reason.

### Problem E — Benchmark path contaminated by future-facing scaffolding
Tool exploration and future agent architecture must not interfere with the controlled MBPP benchmark path.

---

## Global execution policy

Work through the tasks below in order.
Do not skip ahead.
Do not merge multiple tasks into a giant rewrite.
At the end of each task, stop and show:
- files changed,
- exact logic changed,
- why it was necessary,
- how to validate it,
- what still remains.

If a task requires deleting dead code, only delete code proven to be unused or harmful for the current benchmark path.

---

# [x] TASK 1 — Create a true smoke-test mode before any correctness work

## Goal
Make the adversarial MBPP task runnable in under a few minutes for 1 sample on local hardware.

## Required changes
Implement a dedicated debug/smoke-test mode for the custom adversarial task.

This mode must:
- force `epochs=1`
- force `limit=1` in validation examples
- force `max_iterations=1`
- disable any non-essential hot-path work
- keep behavior changes local to the custom task

For the React path specifically:
- remove `decompose()` from the hot path
- remove `submit()` from the hot path
- keep selector behavior minimal and direct
- do not call any exploration scaffold during controlled MBPP debugging

## Additional requirement
Count and document how many LLM calls happen per sample in:
- baseline-only mode
- react mode with `max_iterations=1`
- react mode with general `max_iterations=N`

## Delivery
Produce the smallest possible code change that creates a reproducible smoke-test path.

## Validate
Provide an exact command for a minimal local run.

Expected validation command style:
- 1 sample
- 1 epoch
- 1 iteration
- 1 model connection
- no exploration tools in the hot path

## Expected result
A single sample should complete fast enough to debug manually, and logs should clearly show that the run is not executing extra exploration scaffolding.

## Do not
- do not only change `epochs`
- do not leave `decompose()` or `submit()` inside the MBPP React benchmark loop
- do not refactor unrelated architecture

# [x] TASK 1.1 — Add temporary instrumentation to prove where time is spent

## Goal
Measure actual latency per step before further fixes.

## Required changes
Add per-step timing around:
- baseline generation
- baseline LLM judge
- selector call
- iteration generation
- iteration LLM judge
- metadata/log finalization

Store timings in structured metadata for the run.

## Expected result
One run should make it obvious which exact step is consuming most wall-clock time.

---

# [x] TASK 2 — Replace fake test_result construction with real ground-truth test execution

## Goal
Stop using `output.pass_pred` or any equivalent guessed pass/fail signal as the internal test judge source.

## Required changes
- Identify how the MBPP benchmark obtains the actual test program / asserts for a sample.
- Execute the real generated code against the real tests in the benchmark path used by this task.
- Build the internal `test_result` from real execution output.
- The deterministic `test_judge` must consume that real execution result.

## Delivery
Implement a real execution path for the artifact under evaluation, returning a structured result like:
- pass: bool
- stdout: str
- stderr: str
- optional counts if naturally available

Keep the structure compatible with the current `test_judge` unless a minimal schema extension is necessary.

## Validate
Run one MBPP sample where:
- baseline code is generated,
- real tests are executed,
- internal `test_judge` reflects the executed result,
- result is no longer a fake default `False`.

Show expected evidence in logs or metadata.

## Do not
- do not rely on `pass_pred`,
- do not hardcode fake success/failure,
- do not replace real execution with LLM judgment.

---

# [x] TASK 3 — Ensure the attacked artifact is the exact artifact being evaluated

## Goal
Make the pipeline evaluate the same artifact that the LLM judge sees and that the attempt record stores.

## Required changes
- Trace the full flow:
  - baseline generated artifact,
  - transformed / attacked artifact,
  - internal test execution target,
  - LLM judge input,
  - Inspect-visible final artifact.
- Remove any mismatch between:
  - generated output,
  - attacked output,
  - evaluated output,
  - logged output.

## Delivery
At the end of an iteration there must be one unambiguous artifact under review.
That exact artifact must be:
- stored in the attempt record,
- passed into the real test execution,
- passed into the LLM judge,
- exposed in the final state/log path used by Inspect.

## Validate
For one iteration, show that:
- `artifact_under_review`,
- `attacked_code` / final mutated artifact,
- LLM judge input,
- test execution input,
all refer to the same concrete string.

Expected result:
- no more scorer/log disagreement about what was evaluated.

## Do not
- do not keep parallel ambiguous artifact fields unless absolutely necessary,
- do not hide the final evaluated artifact only inside local variables.

---

# [ ] TASK 4 — Align final state, metadata, and Inspect-visible output

## Goal
Make Inspect logs and stored metadata reflect what actually happened.

## Required changes
- Ensure final task state exposes the correct final artifact.
- Ensure metadata contains a clean, stable structure for:
  - baseline,
  - all_attempts,
  - attack_succeeded,
  - successful_iteration,
  - stop_reason,
  - total_iterations,
  - selector model,
  - judge model.
- Reconcile current metadata with the existing data contract instead of inventing a new schema.

## Delivery
Produce a clean final state that is actually useful in Inspect logs and viewers.

## Validate
After one debug run, the log should clearly answer:
- what the baseline output was,
- what each attacked artifact was,
- what the real test result was,
- what the LLM judge said,
- why the loop stopped.

## Do not
- do not dump huge noisy blobs without structure,
- do not remove existing useful fields unless replacing them with cleaner equivalents.

---

# [ ] TASK 5 — Improve observability without changing benchmark semantics

## Goal
Make failures diagnosable in one run.

## Required changes
- Add structured per-iteration trace information.
- Preserve attempt history in a compact but informative way.
- Record:
  - selected tactic,
  - previous tactics,
  - attacked artifact summary,
  - real test decision,
  - LLM judge decision,
  - confidence,
  - stop reason if terminal,
  - any exception text.

If Inspect has a better native mechanism than raw metadata for step-level observability, use it only if the change is small and justified.

## Delivery
Improve debugging signal, not architecture complexity.

## Validate
A single failed run should make it obvious whether the problem came from:
- generation,
- attack application,
- test execution,
- judge parsing,
- selector choice,
- iteration exception.

## Do not
- do not add a large tracing framework,
- do not add excessive logging spam that hides the signal.

---

# [ ] TASK 6 — Keep future-facing tool exploration isolated from MBPP benchmark execution

## Goal
Guarantee that MBPP benchmark stabilization is not polluted by tool-pattern experimentation.

## Required changes
- Audit imports and execution flow for `decompose`, `submit`, and any other future-facing scaffolding.
- Confirm that these are not part of the hot path for the controlled MBPP benchmark loop.
- If they are accidentally in the hot path, remove or isolate them from the benchmark execution path.
- Preserve them only as separate exploration utilities if they are still needed.

## Delivery
A clean separation between:
- controlled MBPP benchmark path,
- future tool-agent exploration path.

## Validate
Explain whether the current MBPP loop executes any exploration scaffold.
If yes, show the minimal fix that removes it from the hot path.

## Do not
- do not delete future-facing code unless it is clearly harmful and unused,
- do not merge Gitea / PR workflow logic into MBPP debugging.

---

# [ ] TASK 7 — Performance audit after correctness is restored

## Goal
Only after correctness is fixed, remove avoidable latency from the benchmark loop.

## Required changes
Audit the loop for unnecessary cost multipliers such as:
- repeated model instantiation,
- repeated expensive judge calls with unchanged inputs,
- duplicate generation calls,
- repeated deep copies that are not needed,
- unnecessary multiple model roles using the same backend in the same iteration.

Do not change behavior yet unless the optimization is obvious and semantics-preserving.

## Delivery
First provide a short audit list:
- issue,
- cost impact,
- safe fix.

Then apply only the clearly safe fixes.

## Validate
Compare before vs after for a small debug run:
- same behavior,
- fewer model calls or lower runtime.

## Do not
- do not “optimize” by removing validation,
- do not change scientific meaning of the experiment.

---

# [ ] TASK 8 — Regression-proof the benchmark with focused tests

## Goal
Prevent the same failure mode from coming back.

## Required changes
Add a small, focused test layer for the benchmark logic, especially for:
- real test_result construction,
- attack artifact alignment,
- iteration record schema,
- early stop condition,
- metadata completeness.

Prefer targeted unit/integration tests over broad test suites.

## Delivery
Add only the tests needed to catch the current class of bugs.

## Validate
Show which previous failure each new test would catch.

## Do not
- do not add fragile snapshot tests for huge logs,
- do not create tests that depend on random model behavior unless mocked or isolated.

---

# [ ] TASK 9 — Final cleanup and documentation of the fixed benchmark path

## Goal
Leave the benchmark loop understandable and maintainable.

## Required changes
- Add concise comments where the logic was previously misleading.
- Document the benchmark flow in plain English:
  1. generate baseline,
  2. obtain real test result,
  3. judge baseline,
  4. select tactic,
  5. build attacked artifact,
  6. evaluate attacked artifact,
  7. record trajectory,
  8. stop or continue.
- Document clearly which source is ground truth and which source is subjective:
  - test judge = deterministic ground truth,
  - LLM judge = manipulable evaluator.

## Delivery
A short markdown note or code comments, not a giant document.

## Validate
A new contributor should be able to understand:
- what is being measured,
- what counts as attack success,
- what Inspect is actually logging.

## Do not
- do not write vague architecture prose,
- do not claim the system works unless validation evidence supports it.

---

## Definition of done

This work is only complete when all of the following are true:

- A 1-sample debug run finishes in reasonable time.
- The benchmark does not depend on fake `pass_pred` fallbacks for internal ground-truth test results.
- The attacked artifact is exactly the artifact being tested and judged.
- Inspect-visible outputs and metadata match the actual evaluated artifact.
- Attempt logs are informative and non-empty.
- Future tool-pattern exploration remains isolated from the controlled MBPP benchmark loop.
- The final code is simpler or clearer than before, not more magical.

---

## Output format you must follow after each task

For each completed task, respond with exactly these sections:

### Task completed
State which task number was completed.

### Files changed
List only changed files.

### What changed
Concrete summary of code changes.

### Why this was necessary
Tie the change directly to the bug or performance problem.

### How to validate
Exact command(s) or test(s) to run.

### Expected result
What should now happen.

### Remaining risks
Only real remaining concerns, not generic filler.

Do not skip this format.