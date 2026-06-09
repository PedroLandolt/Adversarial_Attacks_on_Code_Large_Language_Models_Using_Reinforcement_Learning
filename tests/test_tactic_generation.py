"""
Tests for tactic generation prompt quality (Tasks 1 & 2).

RED phase: these tests define what must be true after the fix.
"""

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


def _install_stubs() -> None:
    if "inspect_ai" in sys.modules:
        return

    inspect_ai = types.ModuleType("inspect_ai")

    class Task:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    inspect_ai.Task = Task
    inspect_ai.task = lambda fn: fn
    sys.modules["inspect_ai"] = inspect_ai

    model_mod = types.ModuleType("inspect_ai.model")

    class ChatMessageSystem:
        role = "system"
        def __init__(self, content):
            self.content = content

    class ChatMessageUser:
        role = "user"
        def __init__(self, content):
            self.content = content

    class GenerateConfig:
        def __init__(self, **kw):
            pass

    class DummyModel:
        async def generate(self, *args, **kwargs):
            return SimpleNamespace(completion="semantic")

    model_mod.ChatMessageSystem = ChatMessageSystem
    model_mod.ChatMessageUser = ChatMessageUser
    model_mod.GenerateConfig = GenerateConfig
    model_mod.get_model = lambda name: DummyModel()
    sys.modules["inspect_ai.model"] = model_mod

    solver_mod = types.ModuleType("inspect_ai.solver")
    solver_mod.solver = lambda fn: fn
    solver_mod.TaskState = object
    sys.modules["inspect_ai.solver"] = solver_mod

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


_install_stubs()
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "V3"))

from judge.red_teaming_tactics import _build_dynamic_attack_message
from agent.tactic_registry import get_tactic_registry

_PROMPTS_JSON = Path(__file__).resolve().parents[1] / "V3" / "prompts" / "tactic_generation.json"


class DynamicAttackMessageTests(unittest.TestCase):
    """The FEEDBACK line (test=FAIL / llm=FAIL) inside review artifacts
    contradicts injection-style tactics and causes judge confusion.  It
    must be removed from the dynamic attack message."""

    def test_feedback_line_absent_from_dynamic_message(self):
        msg = _build_dynamic_attack_message(
            "injection",
            "def foo(): return 1",
            metadata={"test_decision": "FAIL", "llm_decision": "FAIL", "llm_confidence": 0.0},
        )
        self.assertNotIn("# FEEDBACK:", msg)

    def test_attempt_context_nonce_still_present(self):
        msg = _build_dynamic_attack_message("injection", "def foo(): return 1")
        self.assertIn("# ATTEMPT_CONTEXT:", msg)

    def test_dynamic_message_is_non_empty(self):
        msg = _build_dynamic_attack_message("crowding", "def foo(): pass")
        self.assertTrue(msg.strip())


class PromptsFileTests(unittest.TestCase):
    """The prompts JSON file must exist and be complete.
    - code_generation_prompt: tactic-independent system message for Step 1 (code generation).
    - artifact_*: per-tactic judge-attack prompts for Step 3."""

    def _load(self) -> dict:
        import json
        self.assertTrue(_PROMPTS_JSON.exists(), f"Prompts file missing: {_PROMPTS_JSON}")
        return json.loads(_PROMPTS_JSON.read_text(encoding="utf-8"))

    def test_prompts_file_exists(self):
        self.assertTrue(_PROMPTS_JSON.exists(), f"Missing: {_PROMPTS_JSON}")

    def test_prompts_file_has_no_tactic_codegen_keys(self):
        """Tactic-specific code-gen keys must be absent — replaced by the single code_generation_prompt."""
        data = self._load()
        self.assertNotIn("template", data, "Tactic-specific 'template' key must be absent")
        self.assertNotIn("default_instruction", data, "Tactic-specific 'default_instruction' key must be absent")
        self.assertNotIn("instructions", data, "Tactic-specific 'instructions' key must be absent")

    def test_prompts_file_has_code_generation_prompt(self):
        """A single tactic-independent prompt steers code generation (Step 1)."""
        data = self._load()
        self.assertIn("code_generation_prompt", data)
        prompt = data["code_generation_prompt"]
        self.assertTrue(prompt.strip(), "code_generation_prompt must not be empty")
        self.assertNotIn("one test", prompt.lower(), "Prompt must not restrict to a single test failure")
        self.assertIn("test suite", prompt.lower(), "Prompt must reference the test suite")

    def test_prompts_file_has_artifact_template(self):
        data = self._load()
        self.assertIn("artifact_template", data)
        self.assertIn("{instruction}", data["artifact_template"])

    def test_prompts_file_has_artifact_default_instruction(self):
        data = self._load()
        self.assertIn("artifact_default_instruction", data)
        self.assertTrue(data["artifact_default_instruction"].strip())

    def test_prompts_file_has_all_renderer_bindings_in_artifact_instructions(self):
        data = self._load()
        artifact_instructions = data.get("artifact_instructions", {})
        for entry in get_tactic_registry("benchmark"):
            self.assertIn(
                entry.renderer_binding,
                artifact_instructions,
                f"renderer_binding '{entry.renderer_binding}' missing from artifact_instructions",
            )

    def test_no_for_example_in_artifact_instructions(self):
        data = self._load()
        for binding, text in data.get("artifact_instructions", {}).items():
            self.assertNotIn(
                "for example",
                text.lower(),
                f"binding={binding}: 'for example' hedging found in artifact_instructions",
            )


if __name__ == "__main__":
    unittest.main()
