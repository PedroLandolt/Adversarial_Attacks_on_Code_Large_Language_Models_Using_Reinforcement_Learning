import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


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

    class ChatMessageSystem:
        def __init__(self, content):
            self.content = content

    class ChatMessageUser:
        def __init__(self, content):
            self.content = content

    class GenerateConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DummyModel:
        async def generate(self, *args, **kwargs):
            return SimpleNamespace(completion="semantic")

    model_mod.ChatMessageSystem = ChatMessageSystem
    model_mod.ChatMessageUser = ChatMessageUser
    model_mod.GenerateConfig = GenerateConfig
    model_mod.get_model = lambda name: DummyModel()
    sys.modules["inspect_ai.model"] = model_mod

    tool_mod = types.ModuleType("inspect_ai.tool")
    tool_mod.Tool = object
    tool_mod.tool = lambda fn: fn
    sys.modules["inspect_ai.tool"] = tool_mod

    util_mod = types.ModuleType("inspect_ai.util")
    util_mod.sandbox = lambda: None
    sys.modules["inspect_ai.util"] = util_mod

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

from agent.selector_policy import ReactSelectorPolicy, SelectorContext
from agent.tactic_registry import get_supported_tactic_families, get_tactic_entry, get_tactic_registry
from judge.red_teaming_tactics import apply_tactic, get_all_tactics


class TacticRegistryTests(unittest.IsolatedAsyncioTestCase):
    def test_registry_exposes_legacy_and_taxonomy_benchmark_tactics(self):
        entries = get_tactic_registry("benchmark")
        families = [entry.tactic_family for entry in entries]
        for legacy_family in ("injection", "output", "semantic", "cot"):
            self.assertIn(legacy_family, families)
        for taxonomy_family in ("roleplay", "appeal_to_authority", "formatting_smuggling", "recursion_crescendo", "crowding"):
            self.assertIn(taxonomy_family, families)
        for entry in entries:
            self.assertTrue(entry.tactic_id)
            self.assertTrue(entry.tactic_family)
            self.assertTrue(entry.environment_support)
            self.assertTrue(entry.renderer_binding)

    async def test_selector_policy_returns_registry_backed_decision(self):
        class SemanticModel:
            async def generate(self, *args, **kwargs):
                return SimpleNamespace(completion="semantic")

        policy = ReactSelectorPolicy(selector_model=SemanticModel(), environment="benchmark")
        decision = await policy.select(
            SelectorContext(
                problem="Write increment.",
                current_code="def increment(x):\n    return x + 1\n",
                test_judge_decision="FAIL",
                llm_judge_decision="FAIL",
                llm_judge_confidence=0.2,
                iteration=1,
                max_iterations=3,
                previous_attempts=[],
            )
        )

        self.assertEqual(decision.tactic_family, "semantic")
        self.assertEqual(decision.tactic_id, "legacy_semantic")
        self.assertEqual(decision.renderer_binding, "semantic_inconsistency")
        self.assertIn("benchmark", decision.environment_support)

    async def test_non_legacy_taxonomy_action_can_be_selected_and_rendered(self):
        class RoleplayModel:
            async def generate(self, *args, **kwargs):
                return SimpleNamespace(completion="roleplay")

        policy = ReactSelectorPolicy(selector_model=RoleplayModel(), environment="benchmark")
        decision = await policy.select(
            SelectorContext(
                problem="Write increment.",
                current_code="def increment(x):\n    return x + 1\n",
                test_judge_decision="FAIL",
                llm_judge_decision="FAIL",
                llm_judge_confidence=0.2,
                iteration=1,
                max_iterations=3,
                previous_attempts=[],
            )
        )

        rendered = apply_tactic(
            "def increment(x):\n    return x + 1\n",
            decision.tactic_family,
            problem="Write increment.",
        )
        self.assertEqual(decision.tactic_family, "roleplay")
        self.assertEqual(decision.tactic_id, "taxonomy_roleplay")
        self.assertEqual(decision.taxonomy_category, "narrative_contextual")
        self.assertIn("Reviewer simulation mode.", rendered)

    def test_registry_and_renderer_keep_legacy_four_tactics_working(self):
        code = "def add_one(x):\n    return x + 1\n"
        self.assertEqual(get_all_tactics(), get_supported_tactic_families("benchmark"))

        for tactic_family in get_all_tactics():
            entry = get_tactic_entry(tactic_family, "benchmark")
            rendered = apply_tactic(code, entry.tactic_family, problem="Write add_one.")
            self.assertIsInstance(rendered, str)
            self.assertNotEqual(rendered.strip(), "")


if __name__ == "__main__":
    unittest.main()
