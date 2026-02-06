import traceback
from typing import List, Tuple

def run_tests(code: str, tests: List[str]) -> Tuple[bool, str]:
    env = {}
    try:
        exec(code, env, env)
        for t in tests:
            exec(t, env, env)
        return True, "OK"
    except Exception:
        return False, traceback.format_exc(limit=2)
