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
        def __init__(self, content):
            self.content = content

    class ChatMessageUser:
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

from judge.red_teaming_tactics import (
    _build_dynamic_attack_message,
    build_tactic_generation_prompt,
)
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


class TacticGenerationPromptTests(unittest.TestCase):
    """build_tactic_generation_prompt must:
    - never forbid code fences (the old "no markdown code fences" was
      preventing clean extraction),
    - mention code fences as an allowed output format,
    - include the tactic-specific bug instruction,
    - explicitly prohibit comments, docstrings, and test code,
    - use directive per-tactic instructions without hedging."""

    def test_prompt_does_not_forbid_code_fences(self):
        for entry in get_tactic_registry("benchmark"):
            prompt = build_tactic_generation_prompt(entry.renderer_binding)
            self.assertNotIn(
                "no markdown code fences",
                prompt,
                msg=f"Tactic {entry.tactic_id} prompt forbids code fences",
            )

    def test_prompt_allows_or_mentions_code_fences(self):
        prompt = build_tactic_generation_prompt("prompt_injection")
        fence_keywords = ("```python", "code block", "code fence", "```")
        self.assertTrue(
            any(kw in prompt for kw in fence_keywords),
            msg="Generation prompt should allow code fences for easier extraction",
        )

    def test_all_renderer_bindings_return_non_empty_prompt(self):
        for entry in get_tactic_registry("benchmark"):
            prompt = build_tactic_generation_prompt(entry.renderer_binding)
            self.assertTrue(
                prompt.strip(),
                msg=f"Empty prompt for renderer_binding={entry.renderer_binding}",
            )

    def test_unknown_renderer_binding_returns_default_prompt(self):
        prompt = build_tactic_generation_prompt("__unknown__")
        self.assertTrue(prompt.strip())

    def test_prompt_explicitly_prohibits_comments_and_docstrings(self):
        """Every prompt must ban comments and docstrings so the model never
        adds a line like '# bug: off-by-one here' that exposes the attack."""
        for entry in get_tactic_registry("benchmark"):
            prompt = build_tactic_generation_prompt(entry.renderer_binding).lower()
            has_prohibition = (
                "no comment" in prompt
                or "no docstring" in prompt
            )
            self.assertTrue(
                has_prohibition,
                msg=f"binding={entry.renderer_binding}: missing explicit prohibition on comments/docstrings",
            )

    def test_prompt_explicitly_prohibits_test_code(self):
        """Every prompt must ban test/validation code appended after the function."""
        for entry in get_tactic_registry("benchmark"):
            prompt = build_tactic_generation_prompt(entry.renderer_binding).lower()
            has_prohibition = "no test" in prompt or "test code" in prompt
            self.assertTrue(
                has_prohibition,
                msg=f"binding={entry.renderer_binding}: missing explicit prohibition on test code",
            )

    def test_per_tactic_instructions_do_not_hedge_with_for_example(self):
        """'for example' makes small models pick the examples literally instead of
        following the core bug instruction.  Instructions must be directive."""
        for entry in get_tactic_registry("benchmark"):
            prompt = build_tactic_generation_prompt(entry.renderer_binding)
            self.assertNotIn(
                "for example",
                prompt.lower(),
                msg=f"binding={entry.renderer_binding}: instruction still contains 'for example' hedging",
            )


class PromptsFileTests(unittest.TestCase):
    """The prompts JSON file must exist, be complete, and be structured so that
    a professor can edit it without touching Python code."""

    def _load(self) -> dict:
        import json
        self.assertTrue(_PROMPTS_JSON.exists(), f"Prompts file missing: {_PROMPTS_JSON}")
        return json.loads(_PROMPTS_JSON.read_text(encoding="utf-8"))

    def test_prompts_file_exists(self):
        self.assertTrue(_PROMPTS_JSON.exists(), f"Missing: {_PROMPTS_JSON}")

    def test_prompts_file_has_template_key(self):
        data = self._load()
        self.assertIn("template", data, "JSON must have a 'template' key")
        self.assertIn("{instruction}", data["template"],
                      "'template' must contain the {instruction} placeholder")

    def test_prompts_file_has_default_instruction(self):
        data = self._load()
        self.assertIn("default_instruction", data)
        self.assertTrue(data["default_instruction"].strip())

    def test_prompts_file_has_all_renderer_bindings(self):
        import json
        data = self._load()
        instructions = data.get("instructions", {})
        for entry in get_tactic_registry("benchmark"):
            self.assertIn(
                entry.renderer_binding,
                instructions,
                f"renderer_binding '{entry.renderer_binding}' missing from prompts file",
            )

    def test_no_for_example_in_prompts_file(self):
        data = self._load()
        for binding, text in data.get("instructions", {}).items():
            self.assertNotIn(
                "for example",
                text.lower(),
                f"binding={binding}: 'for example' hedging found in prompts file",
            )


if __name__ == "__main__":
    unittest.main()
