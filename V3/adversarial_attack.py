from inspect_ai import Task, task
from inspect_ai.solver import generate
from inspect_evals.mbpp import mbpp

from attacks.misleading_comments import misleading_comments


@task
def adversarial_code_llm(
    temperature: float = 0.5,
    use_misleading_comments: bool = False,
) -> Task:
    """
    MBPP with optional adversarial prompt transformations.
    """

    base_task = mbpp(temperature=temperature)

    solver = []

    if use_misleading_comments:
        solver.append(misleading_comments())

    solver.append(generate())

    # Don't pass scorer as a list - pass it directly
    # base_task.scorer should already be the correct scorer object
    return Task(
        dataset=base_task.dataset,
        solver=solver,
        scorer=base_task.scorer,  # Pass the scorer directly, not in a list
        epochs=base_task.epochs,
        config=base_task.config,
        sandbox=base_task.sandbox,
        version="3.0.0",
    )