import re
from typing import Tuple
from .llm_client import LLMClient

def judge_solution(client: LLMClient, problem_text: str, code: str) -> Tuple[float, str]:
    return client.judge(problem_text, code)

