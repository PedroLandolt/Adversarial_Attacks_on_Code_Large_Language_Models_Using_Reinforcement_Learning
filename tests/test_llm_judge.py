"""
Tests for JESTER/judge/llm_judge.py — parse_response behaviour.

llama3.1:8b wraps its JSON output in ```json fences and sometimes leaves
a trailing explanation line.  These tests pin the exact formats seen in
production so regressions are caught before hitting the smoke test.
"""

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


def _install_stub_modules() -> None:
    if "inspect_ai" in sys.modules:
        return

    inspect_ai = types.ModuleType("inspect_ai")
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
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    model_mod.ChatMessageSystem = ChatMessageSystem
    model_mod.ChatMessageUser = ChatMessageUser
    model_mod.GenerateConfig = GenerateConfig
    model_mod.get_model = lambda name: None
    sys.modules["inspect_ai.model"] = model_mod


def _install_repo_import_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    v3_root = repo_root / "V3"
    if str(v3_root) not in sys.path:
        sys.path.insert(0, str(v3_root))


_install_stub_modules()
_install_repo_import_path()

from judge.llm_judge import LLMJudge


class ParseResponseTests(unittest.TestCase):
    """_parse_response must extract decision/confidence regardless of model formatting."""

    def setUp(self):
        self.judge = LLMJudge.__new__(LLMJudge)

    def test_plain_json_parsed(self):
        raw = '{"decision": "PASS", "confidence": 0.9, "reasoning": "ok", "vulnerability": "none"}'
        d = self.judge._parse_response(raw)
        self.assertEqual(d.decision, "PASS")
        self.assertAlmostEqual(d.confidence, 0.9)

    def test_fenced_json_block_parsed(self):
        """Model wraps JSON in ```json fences — must be stripped and parsed."""
        raw = '```json\n{"decision": "FAIL", "confidence": 0.7, "reasoning": "bug", "vulnerability": "none"}\n```'
        d = self.judge._parse_response(raw)
        self.assertEqual(d.decision, "FAIL")
        self.assertAlmostEqual(d.confidence, 0.7)

    def test_think_block_stripped_before_parse(self):
        raw = '<think>reasoning here</think>\n{"decision": "PASS", "confidence": 0.85, "reasoning": "ok", "vulnerability": "none"}'
        d = self.judge._parse_response(raw)
        self.assertEqual(d.decision, "PASS")
        self.assertAlmostEqual(d.confidence, 0.85)

    def test_confidence_non_zero_on_valid_json(self):
        """Parsed confidence must be > 0 when the model provides a value."""
        raw = '{"decision": "FAIL", "confidence": 0.6, "reasoning": "wrong output", "vulnerability": "none"}'
        d = self.judge._parse_response(raw)
        self.assertGreater(d.confidence, 0.0)

    def test_empty_response_returns_parse_error(self):
        d = self.judge._parse_response("")
        self.assertEqual(d.decision, "FAIL")
        self.assertIsNone(d.confidence)
        self.assertIn("parse_error", d.vulnerability)

    def test_json_with_fenced_block_confidence_non_zero(self):
        """End-to-end format llama3.1:8b actually returns — fenced, non-zero confidence."""
        raw = (
            '```json\n'
            '{"decision": "PASS", "confidence": 0.9, '
            '"reasoning": "Correct implementation.", "vulnerability": "none"}\n'
            '```'
        )
        d = self.judge._parse_response(raw)
        self.assertGreater(d.confidence, 0.0)
        self.assertNotIn("parse_error", d.vulnerability)


if __name__ == "__main__":
    unittest.main()
