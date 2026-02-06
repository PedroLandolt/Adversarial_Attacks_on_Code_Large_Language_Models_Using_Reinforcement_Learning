from dataclasses import dataclass
from typing import List

@dataclass
class MBPPProblem:
    task_id: str
    prompt: str
    tests: List[str]  # python asserts

@dataclass
class Candidate:
    code: str
    tests_pass: bool
    judge_score: float
    judge_reason: str
    mutated_prompt: str
