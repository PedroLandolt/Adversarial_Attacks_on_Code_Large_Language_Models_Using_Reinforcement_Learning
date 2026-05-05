"""
Tests for V3/utils/code_extraction.py.

RED phase: these tests define the required behavior.  All must fail
before the utility module is created, then pass after.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "V3"))

from utils.code_extraction import extract_python_code


class ThinkBlockStrippingTests(unittest.TestCase):
    """Think blocks must be stripped before extraction so reasoning-model
    code fragments inside <think>...</think> are never mistaken for the
    actual function body."""

    def test_code_after_think_block_is_extracted(self):
        completion = (
            "<think>\nLet me think...\n</think>\n\n"
            "def correct(x):\n    return x + 1"
        )
        result = extract_python_code(completion)
        self.assertEqual(result, "def correct(x):\n    return x + 1")

    def test_python_code_inside_think_block_is_extracted_as_fallback(self):
        """When code appears only inside <think>, it must still be returned.

        qwen3.5:0.8b emits the function body exclusively inside the reasoning
        block, so the extractor must fall back to searching inside <think>
        when nothing parseable is found outside.
        """
        completion = "<think>\ndef inside_think():\n    return 'found'\n</think>"
        result = extract_python_code(completion)
        self.assertIn("def inside_think", result)

    def test_code_in_think_block_does_not_shadow_code_after_it(self):
        completion = (
            "<think>\n"
            "def shadow(x):\n"
            "    return 0\n"
            "</think>\n\n"
            "def real_function(x):\n"
            "    return x * 2\n"
        )
        result = extract_python_code(completion)
        self.assertIn("real_function", result)
        self.assertNotIn("shadow", result)

    def test_empty_completion_after_think_strip_returns_empty(self):
        completion = "<think>all content here</think>"
        result = extract_python_code(completion)
        self.assertEqual(result.strip(), "")


class FencedBlockExtractionTests(unittest.TestCase):
    def test_python_fenced_block_is_extracted(self):
        completion = "Here is the solution:\n```python\ndef foo(x):\n    return x\n```\nEnd."
        result = extract_python_code(completion)
        self.assertEqual(result, "def foo(x):\n    return x")

    def test_generic_fenced_block_is_extracted(self):
        completion = "Solution:\n```\ndef bar():\n    pass\n```"
        result = extract_python_code(completion)
        self.assertEqual(result, "def bar():\n    pass")


class ProseExtractionTests(unittest.TestCase):
    def test_def_line_extracted_from_prose(self):
        completion = (
            "Here is my function:\n\n"
            "def add(a, b):\n"
            "    return a + b\n\n"
            "That should work."
        )
        result = extract_python_code(completion)
        self.assertIn("def add(a, b)", result)

    def test_clean_function_returned_as_is(self):
        code = "def increment(x):\n    return x + 1\n"
        result = extract_python_code(code)
        self.assertEqual(result, code.strip())

    def test_empty_string_returns_empty(self):
        self.assertEqual(extract_python_code(""), "")


if __name__ == "__main__":
    unittest.main()
