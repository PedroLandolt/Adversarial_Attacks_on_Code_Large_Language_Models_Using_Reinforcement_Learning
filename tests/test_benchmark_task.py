import asyncio
import json
import shutil
import subprocess
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4


def _install_stub_modules() -> None:
    if "inspect_ai" in sys.modules:
        return

    inspect_ai = types.ModuleType("inspect_ai")

    class Task:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    inspect_ai.Task = Task
    inspect_ai.task = lambda fn: fn
    sys.modules["inspect_ai"] = inspect_ai

    solver_mod = types.ModuleType("inspect_ai.solver")
    solver_mod.solver = lambda fn: fn
    solver_mod.TaskState = object
    sys.modules["inspect_ai.solver"] = solver_mod

    model_mod = types.ModuleType("inspect_ai.model")

    class ChatMessageUser:
        def __init__(self, content):
            self.content = content

    class GenerateConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DummyModel:
        async def generate(self, *args, **kwargs):
            return SimpleNamespace(completion="")

    model_mod.ChatMessageUser = ChatMessageUser
    model_mod.GenerateConfig = GenerateConfig
    model_mod.get_model = lambda name: DummyModel()
    sys.modules["inspect_ai.model"] = model_mod

    util_mod = types.ModuleType("inspect_ai.util")
    util_mod.sandbox = lambda: None
    sys.modules["inspect_ai.util"] = util_mod

    tool_mod = types.ModuleType("inspect_ai.tool")
    tool_mod.Tool = object
    tool_mod.tool = lambda fn: fn
    tool_mod.bash = lambda *args, **kwargs: None
    sys.modules["inspect_ai.tool"] = tool_mod

    dataset_mod = types.ModuleType("inspect_ai.dataset")
    dataset_mod.Dataset = object
    sys.modules["inspect_ai.dataset"] = dataset_mod

    inspect_evals = types.ModuleType("inspect_evals")
    sys.modules["inspect_evals"] = inspect_evals

    mbpp_mod = types.ModuleType("inspect_evals.mbpp")

    class DummyBaseTask:
        def __init__(self):
            self.dataset = []
            self.scorer = None

    mbpp_mod.mbpp = lambda *args, **kwargs: DummyBaseTask()
    sys.modules["inspect_evals.mbpp"] = mbpp_mod

    humaneval_mod = types.ModuleType("inspect_evals.humaneval")
    humaneval_mod.humaneval = lambda *args, **kwargs: DummyBaseTask()
    sys.modules["inspect_evals.humaneval"] = humaneval_mod


def _install_repo_import_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    v3_root = repo_root / "V3"
    if str(v3_root) not in sys.path:
        sys.path.insert(0, str(v3_root))


_install_stub_modules()
_install_repo_import_path()

import adversarial_attack as benchmark_module
from utils.benchmark_loader import (
    BenchmarkSpec,
    build_verification_program,
    extract_benchmark_spec,
    load_benchmark_task,
)
from utils.results_persistence import load_persisted_runs


class FakeSandbox:
    def __init__(self):
        self.programs = []

    async def exec(self, cmd, timeout):
        program = cmd[2]
        self.programs.append(program)
        completed = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return SimpleNamespace(
            success=completed.returncode == 0,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


class DummyState:
    def __init__(self, prompt: str, test_list: list[str] | None = None, metadata=None, target=None):
        self.messages = [SimpleNamespace(content=prompt)]
        self.input_text = prompt
        self.metadata = metadata if metadata is not None else {"test_list": test_list or []}
        self.target = target if target is not None else test_list
        self.output = SimpleNamespace(completion="")


class FakeGenerateResponse:
    def __init__(self, completion: str):
        self.output = SimpleNamespace(completion=completion)


class FakeSelectorPolicy:
    def __init__(self, _backend):
        self.calls = []

    async def select(self, context):
        self.calls.append(context)
        return SimpleNamespace(tactic_family="injection")


class FakeSelectorPolicyWithReasoning:
    def __init__(self, _backend):
        self.calls = []

    async def select(self, context):
        self.calls.append(context)
        return SimpleNamespace(
            tactic_id="legacy_injection",
            tactic_family="injection",
            environment_support=("benchmark",),
            renderer_binding="legacy_injection",
            taxonomy_category="legacy",
            selector_name="agent_based_decision",
            selector_reasoning="Injection may push the judge toward accepting the broken baseline.",
        )


class FakeRandomSelectorPolicy:
    def __init__(self, environment="benchmark"):
        self.environment = environment
        self.calls = []

    async def select(self, context):
        self.calls.append(context)
        return SimpleNamespace(
            tactic_id="taxonomy_roleplay",
            tactic_family="roleplay",
            environment_support=("benchmark",),
            renderer_binding="narrative_roleplay",
            taxonomy_category="narrative_contextual",
            selector_name="random_choice",
            selector_reasoning=None,
        )


class FakeReactSelectorPolicyWithCotToggle:
    init_flags = []

    def __init__(self, _backend, environment="benchmark", use_chain_of_thought=True):
        self.environment = environment
        self.use_chain_of_thought = use_chain_of_thought
        FakeReactSelectorPolicyWithCotToggle.init_flags.append(bool(use_chain_of_thought))

    async def select(self, context):
        return SimpleNamespace(
            tactic_id="legacy_injection",
            tactic_family="injection",
            environment_support=("benchmark",),
            renderer_binding="prompt_injection",
            taxonomy_category="structural_logic",
            selector_name="agent_based_decision",
            selector_reasoning=(
                "Use injection after judge resistance."
                if self.use_chain_of_thought
                else None
            ),
        )


class FakeJudge:
    def __init__(self, _backend, decisions=None, seen_codes=None):
        self._decisions = decisions or []
        self._seen_codes = seen_codes if seen_codes is not None else []

    async def evaluate(self, problem, code, test_results, metadata=None):
        self._seen_codes.append(
            {
                "problem": problem,
                "code": code,
                "test_results": test_results,
            }
        )
        return self._decisions.pop(0)


class FakeModelName:
    def __init__(self, value: str):
        self.value = value

    def __str__(self) -> str:
        return self.value


class FakeBanditSelectorPolicy:
    init_calls = []

    def __init__(
        self,
        environment="benchmark",
        bandit_algorithm="ucb1",
        weights_path=None,
        freeze_weights=False,
    ):
        FakeBanditSelectorPolicy.init_calls.append(
            {
                "environment": environment,
                "bandit_algorithm": bandit_algorithm,
                "weights_path": weights_path,
                "freeze_weights": freeze_weights,
            }
        )

    async def select(self, context):
        return SimpleNamespace(
            tactic_id="legacy_injection",
            tactic_family="injection",
            environment_support=("benchmark",),
            renderer_binding="prompt_injection",
            taxonomy_category="structural_logic",
            selector_name="rl_bandit",
            selector_reasoning=None,
            bandit_algorithm="ucb1",
            bandit_state={"algorithm": "ucb1", "total_pulls": 0},
        )

    def record_outcome(self, decision, attempt_record):
        return {
            "arm_id": "legacy_injection",
            "reward": 0.0,
            "reward_rule": "benchmark_v1",
        }


class BenchmarkTaskTests(unittest.IsolatedAsyncioTestCase):
    async def test_random_choice_policy_mode_uses_same_entrypoint_and_records_decision_shape(self):
        sandbox = FakeSandbox()
        judge_decisions = [
            SimpleNamespace(decision="FAIL", confidence=0.2, reasoning="baseline fail", vulnerability="none"),
            SimpleNamespace(decision="FAIL", confidence=0.3, reasoning="iteration fail", vulnerability="none"),
        ]
        state = DummyState(
            prompt="Write increment.",
            test_list=["assert increment(1) == 3"],
        )
        generated = [
            "```python\ndef increment(x):\n    return x + 1\n```",
            "```python\ndef increment(x):\n    return x + 1\n```",
        ]

        async def generate_fn(_state):
            return FakeGenerateResponse(generated.pop(0))

        with (
            patch.object(benchmark_module, "sandbox", return_value=sandbox),
            patch.object(
                benchmark_module,
                "validate_python_syntax",
                return_value={"syntax_valid": True, "syntax_error": None},
            ),
            patch.object(benchmark_module, "RandomSelectorPolicy", FakeRandomSelectorPolicy),
            patch.object(
                benchmark_module,
                "LLMJudge",
                lambda backend: FakeJudge(backend, decisions=judge_decisions),
            ),
        ):
            task = benchmark_module.adversarial_code_llm(
                mutation_strategy="react",
                policy_mode="random_choice",
                use_llm_judge=True,
                judge_model="same-backend",
                selector_model="same-backend",
                max_iterations=1,
            )
            solve = task.solver[0]
            solved_state = await solve(state, generate_fn)

        metadata = solved_state.metadata
        attempt = metadata["all_attempts"][0]
        selector_output = attempt["trace"]["selector_output"]
        self.assertEqual(metadata["policy_mode"], "random_choice")
        self.assertEqual(selector_output["policy_mode"], "random_choice")
        self.assertEqual(selector_output["tactic_id"], "taxonomy_roleplay")
        self.assertEqual(selector_output["tactic_family"], "roleplay")
        self.assertEqual(selector_output["renderer_binding"], "narrative_roleplay")
        self.assertIsNone(selector_output["selector_reasoning"])

    async def test_random_choice_policy_mode_works_on_humaneval_through_shared_contract(self):
        sandbox = FakeSandbox()
        judge_decisions = [
            SimpleNamespace(decision="FAIL", confidence=0.2, reasoning="baseline fail", vulnerability="none"),
            SimpleNamespace(decision="FAIL", confidence=0.3, reasoning="iteration fail", vulnerability="none"),
        ]
        state = DummyState(
            prompt="Write increment.",
            metadata={
                "test": "def check(candidate):\n    assert candidate(1) == 2\n",
                "entry_point": "increment",
                "sample_id": "humaneval_random_choice_sample",
            },
            target="def check(candidate):\n    assert candidate(1) == 2\n",
        )
        generated = [
            "```python\ndef increment(x):\n    return x + 1\n```",
            "```python\ndef increment(x):\n    return x + 1\n```",
        ]

        async def generate_fn(_state):
            return FakeGenerateResponse(generated.pop(0))

        with (
            patch.object(benchmark_module, "sandbox", return_value=sandbox),
            patch.object(
                benchmark_module,
                "validate_python_syntax",
                return_value={"syntax_valid": True, "syntax_error": None},
            ),
            patch.object(benchmark_module, "RandomSelectorPolicy", FakeRandomSelectorPolicy),
            patch.object(
                benchmark_module,
                "LLMJudge",
                lambda backend: FakeJudge(backend, decisions=judge_decisions),
            ),
        ):
            task = benchmark_module.adversarial_code_llm(
                benchmark="humaneval",
                mutation_strategy="react",
                policy_mode="random_choice",
                use_llm_judge=True,
                judge_model="same-backend",
                selector_model="same-backend",
                max_iterations=1,
            )
            solve = task.solver[0]
            solved_state = await solve(state, generate_fn)

        metadata = solved_state.metadata
        attempt = metadata["all_attempts"][0]
        selector_output = attempt["trace"]["selector_output"]
        self.assertEqual(metadata["benchmark"], "humaneval")
        self.assertEqual(metadata["policy_mode"], "random_choice")
        self.assertEqual(selector_output["policy_mode"], "random_choice")
        self.assertEqual(selector_output["tactic_id"], "taxonomy_roleplay")
        self.assertEqual(selector_output["tactic_family"], "roleplay")
        self.assertEqual(attempt["selected_tactic_action"]["tactic_id"], "taxonomy_roleplay")
        self.assertEqual(attempt["selected_tactic_action"]["tactic_family"], "roleplay")
        self.assertTrue(sandbox.programs[1].startswith(attempt["executable_code"]))
        self.assertIn("check(increment)", sandbox.programs[1])
        self.assertEqual(metadata["prompt"], "Write increment.")
        self.assertEqual(metadata["test"], "def check(candidate):\n    assert candidate(1) == 2\n")
        self.assertEqual(metadata["entry_point"], "increment")
        self.assertIn("baseline", metadata)
        self.assertIn("all_attempts", metadata)

    async def test_policy_modes_share_entrypoint_and_persist_comparable_outputs_with_rl_bandit(self):
        temp_dir = Path.cwd() / ".tmp_test_results" / uuid4().hex
        results_dir = str(temp_dir / "results")
        try:
            async def run_policy_mode(
                *,
                policy_mode: str,
                max_iterations: int,
                selector_patch=None,
            ):
                sandbox = FakeSandbox()
                state = DummyState(
                    prompt="Write increment.",
                    test_list=["assert increment(1) == 3"],
                    metadata={"test_list": ["assert increment(1) == 3"], "sample_id": f"{policy_mode}_sample"},
                    target=["assert increment(1) == 3"],
                )
                generated = [
                    "```python\ndef increment(x):\n    return x + 1\n```"
                    for _ in range(max_iterations + 1)
                ]
                judge_decisions = [
                    SimpleNamespace(decision="FAIL", confidence=0.2, reasoning="baseline fail", vulnerability="none")
                ] + [
                    SimpleNamespace(decision="FAIL", confidence=0.3, reasoning="iteration fail", vulnerability="none")
                    for _ in range(max_iterations)
                ]

                async def generate_fn(_state):
                    return FakeGenerateResponse(generated.pop(0))

                patches = [
                    patch.object(benchmark_module, "sandbox", return_value=sandbox),
                    patch.object(
                        benchmark_module,
                        "validate_python_syntax",
                        return_value={"syntax_valid": True, "syntax_error": None},
                    ),
                    patch.object(
                        benchmark_module,
                        "LLMJudge",
                        lambda backend: FakeJudge(backend, decisions=list(judge_decisions)),
                    ),
                ]
                if selector_patch is not None:
                    patches.append(selector_patch)

                with patches[0], patches[1], patches[2]:
                    extra_context = patches[3] if len(patches) > 3 else None
                    if extra_context:
                        with extra_context:
                            task = benchmark_module.adversarial_code_llm(
                                benchmark="mbpp",
                                mutation_strategy="react",
                                policy_mode=policy_mode,
                                experiment_mode="iterative",
                                use_llm_judge=True,
                                judge_model="same-backend",
                                selector_model="same-backend",
                                max_iterations=max_iterations,
                                results_dir=results_dir,
                            )
                            return await task.solver[0](state, generate_fn)
                    task = benchmark_module.adversarial_code_llm(
                        benchmark="mbpp",
                        mutation_strategy="react",
                        policy_mode=policy_mode,
                        experiment_mode="iterative",
                        use_llm_judge=True,
                        judge_model="same-backend",
                        selector_model="same-backend",
                        max_iterations=max_iterations,
                        results_dir=results_dir,
                    )
                    return await task.solver[0](state, generate_fn)

            random_state = await run_policy_mode(
                policy_mode="random_choice",
                max_iterations=1,
                selector_patch=patch.object(
                    benchmark_module,
                    "RandomSelectorPolicy",
                    FakeRandomSelectorPolicy,
                ),
            )
            agent_state = await run_policy_mode(
                policy_mode="agent_based_decision",
                max_iterations=1,
                selector_patch=patch.object(
                    benchmark_module,
                    "ReactSelectorPolicy",
                    FakeSelectorPolicyWithReasoning,
                ),
            )
            rl_state = await run_policy_mode(
                policy_mode="rl_bandit",
                max_iterations=2,
            )

            loaded_runs = load_persisted_runs(results_dir)
            self.assertEqual(len(loaded_runs), 3)
            self.assertEqual(
                {run["run_config"]["policy_mode"] for run in loaded_runs},
                {"random_choice", "agent_based_decision", "rl_bandit"},
            )

            rl_run = next(
                run for run in loaded_runs if run["run_config"]["policy_mode"] == "rl_bandit"
            )
            self.assertEqual(rl_run["run_config"]["bandit_algorithm"], "ucb1")
            self.assertEqual(rl_run["run_summary"]["bandit_algorithm"], "ucb1")
            self.assertTrue(all("reward" in attempt for attempt in rl_run["attempts"]))
            self.assertTrue(all("arm_id" in attempt for attempt in rl_run["attempts"]))
            self.assertTrue(all(attempt["bandit_algorithm"] == "ucb1" for attempt in rl_run["attempts"][1:]))
            self.assertTrue(all(attempt["bandit_state"] is not None for attempt in rl_run["attempts"][1:]))

            rl_attempts = rl_state.metadata["all_attempts"]
            self.assertEqual(rl_state.metadata["policy_mode"], "rl_bandit")
            self.assertEqual(rl_state.metadata["bandit_algorithm"], "ucb1")
            self.assertEqual(
                rl_attempts[0]["trace"]["selector_output"]["bandit_algorithm"],
                "ucb1",
            )
            self.assertEqual(
                rl_attempts[0]["selected_tactic_action"]["tactic_id"],
                "legacy_injection",
            )
            self.assertEqual(
                rl_attempts[1]["selected_tactic_action"]["tactic_id"],
                "legacy_output",
            )

            self.assertEqual(random_state.metadata["policy_mode"], "random_choice")
            self.assertEqual(agent_state.metadata["policy_mode"], "agent_based_decision")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def test_agent_based_decision_can_toggle_selector_cot_and_persist_it(self):
        temp_dir = Path.cwd() / ".tmp_test_results" / uuid4().hex
        results_dir = str(temp_dir / "results")
        try:
            FakeReactSelectorPolicyWithCotToggle.init_flags = []

            async def run_with_cot(enabled: bool):
                sandbox = FakeSandbox()
                judge_decisions = [
                    SimpleNamespace(decision="FAIL", confidence=0.2, reasoning="baseline fail", vulnerability="none"),
                    SimpleNamespace(decision="FAIL", confidence=0.3, reasoning="iteration fail", vulnerability="none"),
                ]
                state = DummyState(
                    prompt="Write increment.",
                    test_list=["assert increment(1) == 3"],
                    metadata={"test_list": ["assert increment(1) == 3"], "sample_id": f"cot_{enabled}"},
                    target=["assert increment(1) == 3"],
                )
                generated = [
                    "```python\ndef increment(x):\n    return x + 1\n```",
                    "```python\ndef increment(x):\n    return x + 1\n```",
                ]

                async def generate_fn(_state):
                    return FakeGenerateResponse(generated.pop(0))

                with (
                    patch.object(benchmark_module, "sandbox", return_value=sandbox),
                    patch.object(
                        benchmark_module,
                        "validate_python_syntax",
                        return_value={"syntax_valid": True, "syntax_error": None},
                    ),
                    patch.object(
                        benchmark_module,
                        "ReactSelectorPolicy",
                        FakeReactSelectorPolicyWithCotToggle,
                    ),
                    patch.object(
                        benchmark_module,
                        "LLMJudge",
                        lambda backend: FakeJudge(backend, decisions=list(judge_decisions)),
                    ),
                ):
                    task = benchmark_module.adversarial_code_llm(
                        benchmark="mbpp",
                        mutation_strategy="react",
                        policy_mode="agent_based_decision",
                        selector_use_cot=enabled,
                        experiment_mode="iterative",
                        use_llm_judge=True,
                        judge_model="same-backend",
                        selector_model="same-backend",
                        max_iterations=1,
                        results_dir=results_dir,
                    )
                    return await task.solver[0](state, generate_fn)

            cot_enabled_state = await run_with_cot(True)
            cot_disabled_state = await run_with_cot(False)

            self.assertEqual(FakeReactSelectorPolicyWithCotToggle.init_flags, [True, False])

            enabled_attempt = cot_enabled_state.metadata["all_attempts"][0]
            disabled_attempt = cot_disabled_state.metadata["all_attempts"][0]

            self.assertTrue(cot_enabled_state.metadata["selector_cot_enabled"])
            self.assertFalse(cot_disabled_state.metadata["selector_cot_enabled"])
            self.assertTrue(enabled_attempt["trace"]["selector_output"]["selector_cot_enabled"])
            self.assertFalse(disabled_attempt["trace"]["selector_output"]["selector_cot_enabled"])
            self.assertEqual(
                enabled_attempt["trace"]["selector_output"]["selector_reasoning"],
                "Use injection after judge resistance.",
            )
            self.assertIsNone(disabled_attempt["trace"]["selector_output"]["selector_reasoning"])

            loaded_runs = load_persisted_runs(results_dir)
            self.assertEqual(len(loaded_runs), 2)
            persisted_by_cot = {
                run["run_config"]["selector_cot_enabled"]: run
                for run in loaded_runs
            }
            self.assertIn(True, persisted_by_cot)
            self.assertIn(False, persisted_by_cot)
            self.assertTrue(persisted_by_cot[True]["run_summary"]["selector_cot_enabled"])
            self.assertFalse(persisted_by_cot[False]["run_summary"]["selector_cot_enabled"])
            self.assertTrue(
                all(
                    attempt["selector_cot_enabled"] is True
                    for attempt in persisted_by_cot[True]["attempts"]
                )
            )
            self.assertTrue(
                all(
                    attempt["selector_cot_enabled"] is False
                    for attempt in persisted_by_cot[False]["attempts"]
                )
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_experiment_split_label_does_not_slice_benchmark_dataset(self):
        class Sample:
            def __init__(self, sample_id):
                self.metadata = {"sample_id": sample_id}

        dataset = [Sample(f"sample_{index}") for index in range(60)]

        with patch.object(
            benchmark_module,
            "load_benchmark_task",
            return_value=SimpleNamespace(dataset=dataset, scorer=None),
        ):
            train_task = benchmark_module.adversarial_code_llm(
                benchmark="mbpp",
                experiment_split="train",
                split_definition="mbpp:70_15_15:1-42",
                mutation_strategy="disabled",
                use_llm_judge=False,
            )
            test_task = benchmark_module.adversarial_code_llm(
                benchmark="mbpp",
                experiment_split="test",
                split_definition="mbpp:70_15_15:43-60",
                mutation_strategy="disabled",
                use_llm_judge=False,
            )

        self.assertEqual(len(train_task.dataset), len(dataset))
        self.assertEqual(len(test_task.dataset), len(dataset))
        self.assertEqual(
            [sample.metadata["sample_id"] for sample in train_task.dataset],
            [sample.metadata["sample_id"] for sample in dataset],
        )
        self.assertEqual(
            [sample.metadata["sample_id"] for sample in test_task.dataset],
            [sample.metadata["sample_id"] for sample in dataset],
        )

    async def test_rl_bandit_experiment_split_is_persisted_with_explicit_freeze(self):
        temp_dir = Path.cwd() / ".tmp_test_results" / uuid4().hex
        results_dir = str(temp_dir / "results")
        try:
            FakeBanditSelectorPolicy.init_calls = []
            sandbox = FakeSandbox()
            judge_decisions = [
                SimpleNamespace(decision="FAIL", confidence=0.2, reasoning="baseline fail", vulnerability="none"),
                SimpleNamespace(decision="FAIL", confidence=0.3, reasoning="iteration fail", vulnerability="none"),
            ]
            state = DummyState(
                prompt="Write increment.",
                test_list=["assert increment(1) == 3"],
                metadata={"test_list": ["assert increment(1) == 3"], "sample_id": "split_test_sample"},
                target=["assert increment(1) == 3"],
            )
            generated = [
                "```python\ndef increment(x):\n    return x + 1\n```",
                "```python\ndef increment(x):\n    return x + 1\n```",
            ]

            async def generate_fn(_state):
                return FakeGenerateResponse(generated.pop(0))

            with (
                patch.object(benchmark_module, "sandbox", return_value=sandbox),
                patch.object(
                    benchmark_module,
                    "validate_python_syntax",
                    return_value={"syntax_valid": True, "syntax_error": None},
                ),
                patch.object(
                    benchmark_module,
                    "RLBanditSelectorPolicy",
                    FakeBanditSelectorPolicy,
                ),
                patch.object(
                    benchmark_module,
                    "LLMJudge",
                    lambda backend: FakeJudge(backend, decisions=list(judge_decisions)),
                ),
                ):
                task = benchmark_module.adversarial_code_llm(
                    benchmark="mbpp",
                    experiment_split="test",
                    split_definition="mbpp:70_15_15:789-927",
                    mutation_strategy="react",
                    policy_mode="rl_bandit",
                    bandit_algorithm="ucb1",
                    bandit_weights_path="weights/mbpp_ucb1.json",
                    bandit_freeze_weights=True,
                    use_llm_judge=True,
                    judge_model="same-backend",
                    selector_model="same-backend",
                    max_iterations=1,
                    results_dir=results_dir,
                )
                solved_state = await task.solver[0](state, generate_fn)

            self.assertEqual(len(FakeBanditSelectorPolicy.init_calls), 1)
            init_call = FakeBanditSelectorPolicy.init_calls[0]
            self.assertTrue(init_call["freeze_weights"])
            self.assertEqual(init_call["weights_path"], "weights/mbpp_ucb1.json")

            metadata = solved_state.metadata
            self.assertEqual(metadata["experiment_split"], "test")
            self.assertEqual(metadata["split_definition"], "mbpp:70_15_15:789-927")
            self.assertTrue(metadata["bandit_freeze_weights_effective"])

            loaded_runs = load_persisted_runs(results_dir)
            self.assertEqual(len(loaded_runs), 1)
            run = loaded_runs[0]
            self.assertEqual(run["run_config"]["experiment_split"], "test")
            self.assertEqual(run["run_config"]["split_definition"], "mbpp:70_15_15:789-927")
            self.assertTrue(run["run_config"]["bandit_freeze_weights_effective"])
            self.assertEqual(run["run_summary"]["experiment_split"], "test")
            self.assertEqual(run["run_summary"]["split_definition"], "mbpp:70_15_15:789-927")
            self.assertTrue(run["run_summary"]["bandit_freeze_weights_effective"])
            self.assertTrue(all(attempt["experiment_split"] == "test" for attempt in run["attempts"]))
            self.assertTrue(all(attempt["split_definition"] == "mbpp:70_15_15:789-927" for attempt in run["attempts"]))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_loader_returns_same_interface_shape_for_mbpp_and_humaneval(self):
        mbpp_state = DummyState(
            prompt="Write add_one.",
            metadata={"test_list": ["assert add_one(1) == 2"]},
            target=["assert add_one(1) == 2"],
        )
        humaneval_state = DummyState(
            prompt="Write increment.",
            metadata={
                "test": "def check(candidate):\n    assert candidate(1) == 2\n",
                "entry_point": "increment",
            },
            target="def check(candidate):\n    assert candidate(1) == 2\n",
        )

        mbpp_spec = extract_benchmark_spec(mbpp_state, "mbpp")
        humaneval_spec = extract_benchmark_spec(humaneval_state, "humaneval")

        for spec, expected_name in ((mbpp_spec, "mbpp"), (humaneval_spec, "humaneval")):
            self.assertIsInstance(spec, BenchmarkSpec)
            self.assertEqual(spec.benchmark_name, expected_name)
            self.assertIsInstance(spec.problem_text, str)
            self.assertIn("extract_python_code", spec.normalization_requirements)

        self.assertEqual(mbpp_spec.test_list, ["assert add_one(1) == 2"])
        self.assertIsNone(mbpp_spec.test_harness)
        self.assertEqual(humaneval_spec.entry_point, "increment")
        self.assertIsNone(humaneval_spec.test_list)
        self.assertIn("def check(candidate)", humaneval_spec.test_harness)

    def test_loader_builds_verification_programs_for_both_benchmarks(self):
        mbpp_spec = BenchmarkSpec(
            benchmark_name="mbpp",
            problem_text="Write add_one.",
            entry_point=None,
            test_list=["assert add_one(1) == 2"],
            test_harness=None,
            normalization_requirements={"test_format": "assert_list"},
        )
        humaneval_spec = BenchmarkSpec(
            benchmark_name="humaneval",
            problem_text="Write increment.",
            entry_point="increment",
            test_list=None,
            test_harness="def check(candidate):\n    assert candidate(1) == 2\n",
            normalization_requirements={"test_format": "python_harness"},
        )

        mbpp_program = build_verification_program("def add_one(x):\n    return x + 1\n", mbpp_spec)
        humaneval_program = build_verification_program(
            "def increment(x):\n    return x + 1\n",
            humaneval_spec,
        )

        self.assertIn('assert add_one(1) == 2,', mbpp_program)
        self.assertIn("def check(candidate):", humaneval_program)
        self.assertIn("check(increment)", humaneval_program)

    def test_loader_can_construct_mbpp_and_humaneval_tasks(self):
        self.assertIsNotNone(load_benchmark_task("mbpp", temperature=0.0))
        self.assertIsNotNone(load_benchmark_task("humaneval", temperature=0.0))

    def test_humaneval_loader_falls_back_when_temperature_is_unsupported(self):
        calls = []

        def humaneval_without_temperature():
            calls.append("called")
            return SimpleNamespace(dataset=[], scorer=None)

        with patch("utils.benchmark_loader.humaneval", humaneval_without_temperature):
            task = load_benchmark_task("humaneval", temperature=0.3)

        self.assertIsNotNone(task)
        self.assertEqual(calls, ["called"])

    async def test_real_test_result_comes_from_execution(self):
        sandbox = FakeSandbox()
        task = benchmark_module.adversarial_code_llm(
            mutation_strategy="disabled",
            use_llm_judge=False,
        )
        solve = task.solver[0]
        state = DummyState(
            prompt="Write remove_Occ.",
            test_list=['assert remove_Occ("hello","l") == "heo"'],
        )

        async def generate_fn(_state):
            return FakeGenerateResponse(
                "Here's a Python function that solves the task:\n\n"
                "```python\n"
                "def remove_Occ(s, ch):\n"
                "    return s.replace(ch, '', 1)[::-1].replace(ch, '', 1)[::-1]\n"
                "```\n"
                "\nThis should work for the sample tests.\n"
            )

        with (
            patch.object(benchmark_module, "sandbox", return_value=sandbox),
            patch.object(
                benchmark_module,
                "validate_python_syntax",
                return_value={"syntax_valid": True, "syntax_error": None},
            ),
        ):
            solved_state = await solve(state, generate_fn)

        baseline = solved_state.metadata["baseline"]
        self.assertIn("Here's a Python function", baseline["raw_completion"])
        self.assertTrue(baseline["executable_code"].startswith("def remove_Occ"))
        self.assertEqual(baseline["review_artifact"], baseline["executable_code"])
        self.assertNotEqual(baseline["raw_completion"], baseline["executable_code"])
        self.assertTrue(baseline["test_result"]["pass"])
        self.assertEqual(baseline["test_judge"]["decision"], "PASS")
        self.assertEqual(len(sandbox.programs), 1)
        self.assertTrue(sandbox.programs[0].startswith(baseline["executable_code"]))
        self.assertNotIn("Here's a Python function", sandbox.programs[0])
        self.assertIn('assert remove_Occ("hello","l") == "heo"', sandbox.programs[0])

    async def test_invalid_syntax_is_recorded_and_skips_test_execution(self):
        sandbox = FakeSandbox()
        task = benchmark_module.adversarial_code_llm(
            mutation_strategy="disabled",
            use_llm_judge=False,
        )
        solve = task.solver[0]
        state = DummyState(
            prompt="Write broken.",
            test_list=["assert broken() == 1"],
        )

        async def generate_fn(_state):
            return FakeGenerateResponse("def broken(:\n    return 1\n")

        with (
            patch.object(benchmark_module, "sandbox", return_value=sandbox),
            patch.object(
                benchmark_module,
                "validate_python_syntax",
                return_value={
                    "syntax_valid": False,
                    "syntax_error": "tree-sitter detected invalid Python syntax: (module (ERROR))",
                },
            ),
        ):
            solved_state = await solve(state, generate_fn)

        metadata = solved_state.metadata
        baseline = metadata["baseline"]
        self.assertEqual(metadata["stop_reason"], "baseline_invalid_syntax")
        self.assertFalse(baseline["syntax_valid"])
        self.assertEqual(
            baseline["syntax_result"]["syntax_error"],
            "tree-sitter detected invalid Python syntax: (module (ERROR))",
        )
        self.assertIsNone(baseline["test_result"])
        self.assertEqual(len(sandbox.programs), 0)

    async def test_react_attempt_keeps_artifact_aligned_across_record_judge_and_execution(self):
        sandbox = FakeSandbox()
        seen_codes = []
        judge_decisions = [
            SimpleNamespace(decision="FAIL", confidence=0.2, reasoning="baseline fail", vulnerability="none"),
            SimpleNamespace(decision="FAIL", confidence=0.3, reasoning="iteration fail", vulnerability="none"),
        ]
        state = DummyState(
            prompt="Write add_one.",
            test_list=["assert add_one(1) == 3"],
        )
        generated = [
            "Here is the baseline answer:\n```python\ndef add_one(x):\n    return x + 1\n```\n",
            "Here is the attacked candidate:\n```python\ndef add_one(x):\n    return x + 1\n```\n",
        ]

        async def generate_fn(_state):
            return FakeGenerateResponse(generated.pop(0))

        with (
            patch.object(benchmark_module, "sandbox", return_value=sandbox),
            patch.object(
                benchmark_module,
                "validate_python_syntax",
                return_value={"syntax_valid": True, "syntax_error": None},
            ),
            patch.object(benchmark_module, "ReactSelectorPolicy", FakeSelectorPolicy),
            patch.object(
                benchmark_module,
                "LLMJudge",
                lambda backend: FakeJudge(backend, decisions=judge_decisions, seen_codes=seen_codes),
            ),
        ):
            task = benchmark_module.adversarial_code_llm(
                mutation_strategy="react",
                use_llm_judge=True,
                judge_model="same-backend",
                selector_model="same-backend",
                max_iterations=1,
            )
            solve = task.solver[0]
            solved_state = await solve(state, generate_fn)

        attempt = solved_state.metadata["all_attempts"][0]
        self.assertIn("raw_completion", attempt)
        self.assertIn("executable_code", attempt)
        self.assertIn("review_artifact", attempt)
        self.assertIn("artifact_under_review", attempt)
        self.assertIn("trace", attempt)
        self.assertIn("test_result", attempt)
        self.assertIn("test_judge", attempt)
        self.assertIn("llm_judge", attempt)
        self.assertEqual(attempt["review_artifact"], attempt["artifact_under_review"])
        self.assertEqual(attempt["artifact_under_review"], attempt["trace"]["attacked_code_after_tactic"])
        self.assertEqual(attempt["artifact_under_review"], seen_codes[1]["code"])
        self.assertEqual(attempt["trace"]["selector_output"]["policy_mode"], "agent_based_decision")
        self.assertIsNone(attempt["trace"]["selector_output"]["selector_reasoning"])
        self.assertTrue(attempt["executable_code"].startswith("def add_one"))
        self.assertTrue(sandbox.programs[1].startswith(attempt["executable_code"]))
        self.assertNotEqual(attempt["review_artifact"], attempt["executable_code"])

    async def test_agent_based_decision_records_selector_reasoning_in_attempt_shape(self):
        sandbox = FakeSandbox()
        judge_decisions = [
            SimpleNamespace(decision="FAIL", confidence=0.2, reasoning="baseline fail", vulnerability="none"),
            SimpleNamespace(decision="FAIL", confidence=0.3, reasoning="iteration fail", vulnerability="none"),
        ]
        state = DummyState(
            prompt="Write add_one.",
            test_list=["assert add_one(1) == 3"],
        )

        async def generate_fn(_state):
            return FakeGenerateResponse("def add_one(x):\n    return x + 1\n")

        with (
            patch.object(benchmark_module, "sandbox", return_value=sandbox),
            patch.object(
                benchmark_module,
                "validate_python_syntax",
                return_value={"syntax_valid": True, "syntax_error": None},
            ),
            patch.object(benchmark_module, "ReactSelectorPolicy", FakeSelectorPolicyWithReasoning),
            patch.object(
                benchmark_module,
                "LLMJudge",
                lambda backend: FakeJudge(backend, decisions=judge_decisions),
            ),
        ):
            task = benchmark_module.adversarial_code_llm(
                mutation_strategy="react",
                policy_mode="agent_based_decision",
                use_llm_judge=True,
                judge_model="same-backend",
                selector_model="same-backend",
                max_iterations=1,
            )
            solve = task.solver[0]
            solved_state = await solve(state, generate_fn)

        attempt = solved_state.metadata["all_attempts"][0]
        selector_output = attempt["trace"]["selector_output"]
        self.assertEqual(selector_output["tactic_family"], "injection")
        self.assertEqual(
            selector_output["selector_reasoning"],
            "Injection may push the judge toward accepting the broken baseline.",
        )
        self.assertEqual(attempt["test_judge"]["decision"], "FAIL")
        self.assertEqual(attempt["llm_judge"]["decision"], "FAIL")

    async def test_react_stops_early_and_keeps_required_metadata_fields(self):
        sandbox = FakeSandbox()
        judge_decisions = [
            SimpleNamespace(decision="FAIL", confidence=0.2, reasoning="baseline fail", vulnerability="none"),
            SimpleNamespace(decision="PASS", confidence=0.9, reasoning="fooled judge", vulnerability="none"),
        ]
        state = DummyState(
            prompt="Write always_zero.",
            test_list=["assert always_zero() == 1"],
        )
        generate_call_count = 0

        async def generate_fn(_state):
            nonlocal generate_call_count
            generate_call_count += 1
            return FakeGenerateResponse("def always_zero():\n    return 0\n")

        with (
            patch.object(benchmark_module, "sandbox", return_value=sandbox),
            patch.object(
                benchmark_module,
                "validate_python_syntax",
                return_value={"syntax_valid": True, "syntax_error": None},
            ),
            patch.object(benchmark_module, "ReactSelectorPolicy", FakeSelectorPolicy),
            patch.object(
                benchmark_module,
                "LLMJudge",
                lambda backend: FakeJudge(backend, decisions=judge_decisions),
            ),
        ):
            task = benchmark_module.adversarial_code_llm(
                mutation_strategy="react",
                use_llm_judge=True,
                judge_model="same-backend",
                selector_model="same-backend",
                max_iterations=3,
            )
            solve = task.solver[0]
            solved_state = await solve(state, generate_fn)

        metadata = solved_state.metadata
        self.assertEqual(metadata["policy_mode"], "agent_based_decision")
        self.assertTrue(metadata["attack_succeeded"])
        self.assertEqual(metadata["successful_iteration"], 1)
        self.assertEqual(metadata["stop_reason"], "attack_succeeded")
        self.assertEqual(metadata["total_iterations"], 1)
        self.assertEqual(len(metadata["all_attempts"]), 1)
        self.assertEqual(generate_call_count, 2)

        for required_key in (
            "baseline",
            "all_attempts",
            "attack_succeeded",
            "successful_iteration",
            "stop_reason",
            "total_iterations",
            "selector_model",
            "judge_model",
        ):
            self.assertIn(required_key, metadata)

    async def test_one_shot_experiment_mode_limits_adversarial_attempts_to_one(self):
        sandbox = FakeSandbox()
        judge_decisions = [
            SimpleNamespace(decision="FAIL", confidence=0.2, reasoning="baseline fail", vulnerability="none"),
            SimpleNamespace(decision="FAIL", confidence=0.3, reasoning="iteration fail", vulnerability="none"),
        ]
        state = DummyState(
            prompt="Write add_one.",
            test_list=["assert add_one(1) == 3"],
        )
        generate_call_count = 0

        async def generate_fn(_state):
            nonlocal generate_call_count
            generate_call_count += 1
            return FakeGenerateResponse("def add_one(x):\n    return x + 1\n")

        with (
            patch.object(benchmark_module, "sandbox", return_value=sandbox),
            patch.object(
                benchmark_module,
                "validate_python_syntax",
                return_value={"syntax_valid": True, "syntax_error": None},
            ),
            patch.object(benchmark_module, "ReactSelectorPolicy", FakeSelectorPolicy),
            patch.object(
                benchmark_module,
                "LLMJudge",
                lambda backend: FakeJudge(backend, decisions=judge_decisions),
            ),
        ):
            task = benchmark_module.adversarial_code_llm(
                mutation_strategy="react",
                experiment_mode="one_shot",
                use_llm_judge=True,
                judge_model="same-backend",
                selector_model="same-backend",
                max_iterations=3,
            )
            solve = task.solver[0]
            solved_state = await solve(state, generate_fn)

        metadata = solved_state.metadata
        self.assertEqual(metadata["experiment_mode"], "one_shot")
        self.assertEqual(metadata["effective_max_iterations"], 1)
        self.assertEqual(metadata["total_iterations"], 1)
        self.assertEqual(len(metadata["all_attempts"]), 1)
        self.assertEqual(metadata["stop_reason"], "max_iterations_reached")
        self.assertEqual(generate_call_count, 2)

    async def test_iterative_experiment_mode_preserves_configured_iteration_budget(self):
        sandbox = FakeSandbox()
        judge_decisions = [
            SimpleNamespace(decision="FAIL", confidence=0.2, reasoning="baseline fail", vulnerability="none"),
            SimpleNamespace(decision="FAIL", confidence=0.3, reasoning="iteration 1 fail", vulnerability="none"),
            SimpleNamespace(decision="FAIL", confidence=0.4, reasoning="iteration 2 fail", vulnerability="none"),
        ]
        state = DummyState(
            prompt="Write add_one.",
            test_list=["assert add_one(1) == 3"],
        )
        generate_call_count = 0

        async def generate_fn(_state):
            nonlocal generate_call_count
            generate_call_count += 1
            return FakeGenerateResponse("def add_one(x):\n    return x + 1\n")

        with (
            patch.object(benchmark_module, "sandbox", return_value=sandbox),
            patch.object(
                benchmark_module,
                "validate_python_syntax",
                return_value={"syntax_valid": True, "syntax_error": None},
            ),
            patch.object(benchmark_module, "ReactSelectorPolicy", FakeSelectorPolicy),
            patch.object(
                benchmark_module,
                "LLMJudge",
                lambda backend: FakeJudge(backend, decisions=judge_decisions),
            ),
        ):
            task = benchmark_module.adversarial_code_llm(
                mutation_strategy="react",
                experiment_mode="iterative",
                use_llm_judge=True,
                judge_model="same-backend",
                selector_model="same-backend",
                max_iterations=2,
            )
            solve = task.solver[0]
            solved_state = await solve(state, generate_fn)

        metadata = solved_state.metadata
        self.assertEqual(metadata["experiment_mode"], "iterative")
        self.assertEqual(metadata["effective_max_iterations"], 2)
        self.assertEqual(metadata["total_iterations"], 2)
        self.assertEqual(len(metadata["all_attempts"]), 2)
        self.assertEqual(metadata["stop_reason"], "max_iterations_reached")
        self.assertEqual(generate_call_count, 3)

    async def test_results_are_persisted_for_two_runs_and_can_be_read_offline(self):
        temp_dir = Path.cwd() / ".tmp_test_results" / uuid4().hex
        results_dir = str(temp_dir / "results")
        try:
            target_model = FakeModelName("ollama/qwen3.5:0.8b")

            mbpp_sandbox = FakeSandbox()
            mbpp_task = benchmark_module.adversarial_code_llm(
                benchmark="mbpp",
                mutation_strategy="disabled",
                use_llm_judge=False,
                experiment_mode="one_shot",
                results_dir=results_dir,
                target_model=target_model,
            )
            mbpp_state = DummyState(
                prompt="Write add_one.",
                test_list=["assert add_one(1) == 2"],
                metadata={"test_list": ["assert add_one(1) == 2"], "sample_id": "mbpp_sample"},
                target=["assert add_one(1) == 2"],
            )

            async def mbpp_generate_fn(_state):
                return FakeGenerateResponse("def add_one(x):\n    return x + 1\n")

            with (
                patch.object(benchmark_module, "sandbox", return_value=mbpp_sandbox),
                patch.object(
                    benchmark_module,
                    "validate_python_syntax",
                    return_value={"syntax_valid": True, "syntax_error": None},
                ),
            ):
                await mbpp_task.solver[0](mbpp_state, mbpp_generate_fn)

            humaneval_sandbox = FakeSandbox()
            humaneval_task = benchmark_module.adversarial_code_llm(
                benchmark="humaneval",
                mutation_strategy="disabled",
                use_llm_judge=False,
                experiment_mode="iterative",
                results_dir=results_dir,
                target_model=target_model,
            )
            humaneval_state = DummyState(
                prompt="Write increment.",
                metadata={
                    "test": "def check(candidate):\n    assert candidate(1) == 2\n",
                    "entry_point": "increment",
                    "sample_id": "humaneval_sample",
                },
                target="def check(candidate):\n    assert candidate(1) == 2\n",
            )

            async def humaneval_generate_fn(_state):
                return FakeGenerateResponse("def increment(x):\n    return x + 1\n")

            with (
                patch.object(benchmark_module, "sandbox", return_value=humaneval_sandbox),
                patch.object(
                    benchmark_module,
                    "validate_python_syntax",
                    return_value={"syntax_valid": True, "syntax_error": None},
                ),
            ):
                await humaneval_task.solver[0](humaneval_state, humaneval_generate_fn)

            run_dirs = sorted(Path(results_dir).iterdir())
            self.assertEqual(len(run_dirs), 2)
            loaded_runs = load_persisted_runs(results_dir)
            self.assertEqual(len(loaded_runs), 2)

            loaded_configs = [run["run_config"] for run in loaded_runs]
            loaded_summaries = [run["run_summary"] for run in loaded_runs]
            loaded_attempt_counts = [len(run["attempts"]) for run in loaded_runs]

            self.assertEqual(
                {config["benchmark"] for config in loaded_configs},
                {"mbpp", "humaneval"},
            )
            self.assertEqual(
                {config["experiment_mode"] for config in loaded_configs},
                {"one_shot", "iterative"},
            )
            self.assertEqual(
                {config["target_model"] for config in loaded_configs},
                {"ollama/qwen3.5:0.8b"},
            )
            self.assertTrue(all(summary["num_samples"] == 1 for summary in loaded_summaries))
            self.assertTrue(all(summary["run_id"] for summary in loaded_summaries))
            self.assertTrue(all(count >= 1 for count in loaded_attempt_counts))
            self.assertTrue(
                all(
                    attempt["reward"] is not None
                    for run in loaded_runs
                    for attempt in run["attempts"]
                )
            )
            self.assertTrue(
                all(
                    "arm_id" in attempt
                    and "reward_components" in attempt
                    and "reward_rule" in attempt
                    for run in loaded_runs
                    for attempt in run["attempts"]
                )
            )
            self.assertTrue(
                all("average_reward_by_arm" in summary for summary in loaded_summaries)
            )
            self.assertTrue(
                all("cumulative_reward_by_arm" in summary for summary in loaded_summaries)
            )
            self.assertTrue(
                all("pulls_by_arm" in summary for summary in loaded_summaries)
            )
            self.assertTrue(
                all("reward_rule" in summary for summary in loaded_summaries)
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
