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


def _install_repo_import_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    v3_root = repo_root / "V3"
    if str(v3_root) not in sys.path:
        sys.path.insert(0, str(v3_root))


_install_stub_modules()
_install_repo_import_path()

import adversarial_attack as benchmark_module


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
    def __init__(self, prompt: str, test_list: list[str]):
        self.messages = [SimpleNamespace(content=prompt)]
        self.input_text = prompt
        self.metadata = {"test_list": test_list}
        self.target = test_list
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
                "def remove_Occ(s, ch):\n"
                "    return s.replace(ch, '', 1)[::-1].replace(ch, '', 1)[::-1]\n"
            )

        with patch.object(benchmark_module, "sandbox", return_value=sandbox):
            solved_state = await solve(state, generate_fn)

        baseline = solved_state.metadata["baseline"]
        self.assertTrue(baseline["test_result"]["pass"])
        self.assertEqual(baseline["test_judge"]["decision"], "PASS")
        self.assertEqual(len(sandbox.programs), 1)
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
            "def add_one(x):\n    return x + 1\n",
            "def add_one(x):\n    return x + 1\n",
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
        self.assertIn("artifact_under_review", attempt)
        self.assertIn("trace", attempt)
        self.assertIn("test_result", attempt)
        self.assertIn("test_judge", attempt)
        self.assertIn("llm_judge", attempt)
        self.assertEqual(attempt["artifact_under_review"], attempt["trace"]["attacked_code_after_tactic"])
        self.assertEqual(attempt["artifact_under_review"], seen_codes[1]["code"])
        self.assertTrue(sandbox.programs[1].startswith(attempt["artifact_under_review"]))

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
