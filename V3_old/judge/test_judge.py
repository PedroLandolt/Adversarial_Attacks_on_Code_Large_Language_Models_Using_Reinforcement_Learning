"""
Test-based Judge: Evaluates code based on unit test results.

Simple, deterministic, no manipulation possible.
This is the ground truth baseline.
"""

from dataclasses import dataclass


@dataclass
class TestJudgeDecision:
    """Test judge evaluation result."""
    decision: str  # "PASS" or "FAIL"
    confidence: float  # Always 1.0 (deterministic)
    reasoning: str
    vulnerability: str  # Always "none"


def test_judge(test_results: dict) -> TestJudgeDecision:
    """
    Simple test-based judge.

    Args:
        test_results: {"pass": bool, "stdout": str, "stderr": str}

    Returns:
        TestJudgeDecision
    """

    passed = test_results.get("pass", False)

    if passed:
        return TestJudgeDecision(
            decision="PASS",
            confidence=1.0,
            reasoning="All unit tests passed.",
            vulnerability="none",
        )
    else:
        stderr = test_results.get("stderr", "")
        return TestJudgeDecision(
            decision="FAIL",
            confidence=1.0,
            reasoning=f"Unit tests failed. Error: {stderr[:200]}",
            vulnerability="none",
        )