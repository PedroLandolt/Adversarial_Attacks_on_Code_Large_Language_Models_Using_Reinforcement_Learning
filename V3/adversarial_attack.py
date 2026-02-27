from inspect_ai import Task, task
from inspect_ai.solver import solver, TaskState, generate
from inspect_ai.model import ChatMessageUser
from inspect_evals.mbpp import mbpp

from attacks.misleading_comments import misleading_comments
from attacks.variable_renaming import variable_renaming
from attacks.instruction_perturbation import instruction_perturbation
from judge.llm_judge import LLMJudge
from judge.test_judge import test_judge
from judge.red_teaming_tactics import apply_tactic

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
    use_llm_judge: bool = False,
    judge_model: str = "ollama/qwen2.5:7b",
    red_teaming_tactic: str = None,  # "injection" | "output" | "semantic" | "cot"
) -> Task:
    """
    Adversarial attack with iterative mutations and optional LLM-as-judge.
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

    # Initialize LLM judge if enabled
    llm_judge_instance = None
    if use_llm_judge:
        llm_judge_instance = LLMJudge(judge_model)

    @solver
    def iterative_attack():
        async def solve(state: TaskState, generate_fn):
            attempts = []
            original_messages = deepcopy(state.messages)

            # Extract problem from state
            problem_text = str(state.messages[0].content) if state.messages else state.input_text

            # === STEP 1: Baseline ===
            baseline_response = await generate_fn(state)

            if hasattr(baseline_response, 'output'):
                if hasattr(baseline_response.output, 'completion'):
                    baseline_output = baseline_response.output.completion
                else:
                    baseline_output = str(baseline_response.output)
            else:
                baseline_output = str(baseline_response)

            # Get test-based judge result
            test_result = {
                "pass": baseline_response.output.pass_pred if hasattr(baseline_response.output, 'pass_pred') else False,
                "stdout": "",
                "stderr": "",
            }
            test_decision = test_judge(test_result)

            # Get LLM judge result if enabled
            llm_decision = None
            if use_llm_judge:
                # Apply red-teaming tactic if specified BEFORE sending to judge
                judge_code = baseline_output
                if red_teaming_tactic:
                    judge_code = apply_tactic(
                        baseline_output,
                        red_teaming_tactic,
                        problem_text,
                    )
                
                llm_decision = await llm_judge_instance.evaluate(
                    problem_text,
                    judge_code,  # Use potentially manipulated code
                    test_result,
                )

            attempts.append({
                "iteration": 0,
                "mutation": "baseline",
                "prompt": problem_text,
                "response": baseline_output,
                "test_judge": {
                    "decision": test_decision.decision,
                    "confidence": test_decision.confidence,
                },
                "llm_judge": {
                    "decision": llm_decision.decision if llm_decision else None,
                    "confidence": llm_decision.confidence if llm_decision else None,
                    "reasoning": llm_decision.reasoning if llm_decision else None,
                } if use_llm_judge else None,
            })

            # === STEP 2: Iterative mutations ===
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

                    # Apply red-teaming tactic if specified
                    if red_teaming_tactic:
                        mutated_output = apply_tactic(
                            mutated_output,
                            red_teaming_tactic,
                            problem_text,
                        )

                    # Get test-based judge result
                    test_result = {
                        "pass": mutated_state.output.pass_pred if hasattr(mutated_state.output, 'pass_pred') else False,
                        "stdout": "",
                        "stderr": "",
                    }
                    test_decision = test_judge(test_result)

                    # Get LLM judge result if enabled
                    llm_decision = None
                    if use_llm_judge:
                        llm_decision = await llm_judge_instance.evaluate(
                            problem_text,
                            mutated_output,
                            test_result,
                        )

                    attempts.append({
                        "iteration": iteration,
                        "mutation": mutation_name,
                        "red_teaming_tactic": red_teaming_tactic,
                        "prompt": problem_text,
                        "response": mutated_output,
                        "test_judge": {
                            "decision": test_decision.decision,
                            "confidence": test_decision.confidence,
                        },
                        "llm_judge": {
                            "decision": llm_decision.decision if llm_decision else None,
                            "confidence": llm_decision.confidence if llm_decision else None,
                            "reasoning": llm_decision.reasoning if llm_decision else None,
                            "vulnerability": llm_decision.vulnerability if llm_decision else None,
                        } if use_llm_judge else None,
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
                "use_llm_judge": use_llm_judge,
                "red_teaming_tactic": red_teaming_tactic,
            }

            return state

        return solve

    return Task(
        dataset=base_task.dataset,
        solver=[iterative_attack()],
        scorer=base_task.scorer,
        sandbox="docker",
        epochs=base_task.epochs if hasattr(base_task, 'epochs') else 1,
    )