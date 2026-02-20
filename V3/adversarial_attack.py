from inspect_ai import Task, task
from inspect_ai.solver import solver, TaskState, generate
from inspect_ai.model import ChatMessageUser
from inspect_ai.util import sandbox
from inspect_evals.mbpp import mbpp

from attacks.misleading_comments import misleading_comments
from attacks.variable_renaming import variable_renaming
from attacks.instruction_perturbation import instruction_perturbation

import random
from copy import deepcopy


@task
def adversarial_code_llm(
    temperature: float = 0.5,
    max_iterations: int = 5,
    mutation_strategy: str = "random",
    use_misleading_comments: bool = True,
    use_variable_renaming: bool = True,
    use_instruction_perturbation: bool = False,
) -> Task:
    """
    Adversarial attack with iterative mutations and feedback.
    """

    base_task = mbpp(temperature=temperature)

    # Build list of available mutators
    mutators = []
    if use_misleading_comments:
        mutators.append(("misleading_comments", misleading_comments()))
    if use_variable_renaming:
        mutators.append(("variable_renaming", variable_renaming()))
    if use_instruction_perturbation:
        mutators.append(("instruction_perturbation", instruction_perturbation()))

    @solver
    def iterative_attack():
        async def solve(state: TaskState, generate_fn):
            attempts = []
            original_messages = deepcopy(state.messages)
            
            # Baseline
            baseline_response = await generate_fn(state)
            
            if hasattr(baseline_response, 'output'):
                if hasattr(baseline_response.output, 'completion'):
                    baseline_output = baseline_response.output.completion
                else:
                    baseline_output = str(baseline_response.output)
            else:
                baseline_output = str(baseline_response)
            
            attempts.append({
                "iteration": 0,
                "mutation": "baseline",
                "prompt": str(state.messages[0].content) if state.messages else state.input_text,
                "response": baseline_output,
            })
            
            # Iterative mutations
            for iteration in range(1, max_iterations + 1):
                if mutation_strategy == "random" and mutators:
                    mutation_name, mutator = random.choice(mutators)
                elif mutation_strategy == "sequential" and mutators:
                    mutation_name, mutator = mutators[(iteration - 1) % len(mutators)]
                else:
                    break
                
                state.messages = deepcopy(original_messages)
                
                try:
                    mutated_state = await mutator.solve(state, generate_fn)
                    
                    if hasattr(mutated_state, 'output') and mutated_state.output:
                        if hasattr(mutated_state.output, 'completion'):
                            mutated_output = mutated_state.output.completion
                        else:
                            mutated_output = str(mutated_state.output)
                    else:
                        mutated_output = "[No output generated]"
                    
                    attempts.append({
                        "iteration": iteration,
                        "mutation": mutation_name,
                        "prompt": str(state.messages[0].content) if state.messages else state.input_text,
                        "response": mutated_output,
                    })
                    
                except Exception as e:
                    attempts.append({
                        "iteration": iteration,
                        "mutation": mutation_name,
                        "error": str(e),
                    })
            
            state.metadata = {
                "baseline_output": baseline_output,
                "total_mutations": len(attempts) - 1,
                "all_attempts": attempts,
            }
            
            return state
        
        return solve

    return Task(
        dataset=base_task.dataset,
        solver=[iterative_attack()],
        scorer=base_task.scorer,
        sandbox="docker",  # ✅ Adicionar sandbox Docker
        epochs=base_task.epochs if hasattr(base_task, 'epochs') else 1,
    )