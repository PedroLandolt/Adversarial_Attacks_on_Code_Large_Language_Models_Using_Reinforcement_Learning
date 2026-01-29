from inspect_ai import Task, task
from inspect_ai.dataset import example_dataset
from inspect_ai.agent import react

@task
def humaneval_react():
    return Task(
        dataset=example_dataset("humaneval"),  # se não existir como example_dataset, usas inspect_evals
        solver=[
            react(attempts=2)  # dá 1-2 tentativas antes de submeter
        ],
        # scorer: normalmente é execução de testes do HumanEval (no inspect_evals já vem)
    )
