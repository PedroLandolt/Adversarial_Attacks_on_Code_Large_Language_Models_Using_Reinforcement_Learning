import asyncio
import subprocess
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


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

        with patch.object(benchmark_module, "sandbox", return_value=sandbox):
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


if __name__ == "__main__":
    unittest.main()
