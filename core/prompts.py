from .types import MBPPProblem

def build_base_prompt(problem: MBPPProblem) -> str:
    return f"""
                You are solving a Python programming problem.

                Problem:
                {problem.prompt}

                Return ONLY Python code. No markdown. No explanation.
            """
