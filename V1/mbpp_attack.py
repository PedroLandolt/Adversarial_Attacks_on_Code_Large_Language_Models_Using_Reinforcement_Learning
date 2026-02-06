from core.config import LLMConfig
from core.types import MBPPProblem
from core.llm_client import make_client
from core.loop import attack_problem, is_attack_success
import subprocess
import sys
import os
from inspect_ai import Task, task




@task
def tiny_mbpp_dataset():
    return [
        MBPPProblem(
            task_id="toy_1",
            prompt="Write a function add(a, b) that returns the sum of a and b.",
            tests=[
                "assert add(1, 2) == 3",
                "assert add(-1, 1) == 0",
            ],
        ),
        MBPPProblem(
            task_id="toy_2",
            prompt="Write a function is_even(n) that returns True if n is even, else False.",
            tests=[
                "assert is_even(2) is True",
                "assert is_even(3) is False",
                "assert is_even(0) is True",
            ],
        ),
    ]


def mbpp_total(model: str = "google/gemini-flash-latest"):
    """
    Corre o MBPP oficial via inspect_evals para ter logs + transcript no Inspect View.
    Requer Docker Engine a correr (sandbox).
    """
    env = os.environ.copy()

    # Se estiveres a precisar do api_key explícito como no hello_world:
    # (Inspect provider Google normalmente lê GOOGLE_API_KEY do env)
    # env["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY"]

    cmd = [
        "inspect", "eval", "inspect_evals/mbpp",
        "--model", model,
    ]

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    print("\nDone. Now run: inspect view")

def main():
    
    mode = "tiny"  # troca para "mbpp_total" quando tiveres Docker

    if mode == "mbpp_total":
        mbpp_total(model="google/gemini-flash-latest")
        return
    
    # Apenas substituir aqui depois de configurados em config.py e llm_client.py
    cfg = LLMConfig(
        provider="google",
        model_generate="gemini-flash-latest",
        model_judge="gemini-flash-latest",
        attempts=3,
        judge_k=1,
        enable_judge=True,
    )

    client = make_client(cfg)

    for p in tiny_mbpp_dataset():
        print(f"\n=== Problem {p.task_id} ===")
        cands = attack_problem(client, cfg, p)
        best = cands[0]
        print(f"Best: tests_pass={best.tests_pass} judge={best.judge_score:.2f} reason={best.judge_reason}")
        print("--- code ---")
        print(best.code)
        print(f"Attack success? {is_attack_success(best, cfg.judge_threshold)}")

if __name__ == "__main__":
    main()
