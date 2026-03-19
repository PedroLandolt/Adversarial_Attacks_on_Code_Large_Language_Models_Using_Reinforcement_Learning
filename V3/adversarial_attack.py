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
from agent.react_selector import ReactTacticSelector

import random
from copy import deepcopy


@task
def adversarial_code_llm(
    temperature: float = 0.5,
    max_iterations: int = 5,
    mutation_strategy: str = "random",  # "random", "sequential", "react"
    use_misleading_comments: bool = True,
    use_variable_renaming: bool = True,
    use_instruction_perturbation: bool = False,
    use_llm_judge: bool = False,
    judge_model: str = "ollama/qwen2.5:7b",
    red_teaming_tactic: str = None,  # "injection" | "output" | "semantic" | "cot" | None
) -> Task:
    """
    Adversarial Code LLM attack task.
    
    mutation_strategy options:
    - "random": Random heuristic mutations (original)
    - "sequential": Sequential heuristic mutations (original)
    - "react": ReAct loop with LLM selector choosing red-teaming tactics
    """

    base_task = mbpp(temperature=temperature)

    # Heuristic mutators (for random/sequential strategies)
    mutators = []
    if use_misleading_comments:
        mutators.append(("misleading_comments", misleading_comments()))
    if use_variable_renaming:
        mutators.append(("variable_renaming", variable_renaming()))
    if use_instruction_perturbation:
        mutators.append(("instruction_perturbation", instruction_perturbation()))

    # Initialize judges
    llm_judge_instance = None
    if use_llm_judge:
        llm_judge_instance = LLMJudge(judge_model)
    
    # Initialize ReAct selector (only for react strategy)
    react_selector = None
    if mutation_strategy == "react" and use_llm_judge:
        react_selector = ReactTacticSelector(judge_model)

    @solver
    def iterative_attack():
        async def solve(state: TaskState, generate_fn):
            attempts = []
            original_messages = deepcopy(state.messages)

            # Extract problem
            problem_text = str(state.messages[0].content) if state.messages else state.input_text

            # === BASELINE ===
            baseline_response = await generate_fn(state)

            if hasattr(baseline_response, 'output'):
                if hasattr(baseline_response.output, 'completion'):
                    baseline_output = baseline_response.output.completion
                else:
                    baseline_output = str(baseline_response.output)
            else:
                baseline_output = str(baseline_response)

            # Baseline test judge
            test_result = {
                "pass": baseline_response.output.pass_pred if hasattr(baseline_response.output, 'pass_pred') else False,
                "stdout": "",
                "stderr": "",
            }
            test_decision = test_judge(test_result)

            # Baseline LLM judge
            llm_decision = None
            if use_llm_judge:
                llm_decision = await llm_judge_instance.evaluate(
                    problem_text,
                    baseline_output,
                    test_result,
                )

            attempts.append({
                "iteration": 0,
                "mutation": "baseline",
                "tactic": None,
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
                "attack_success": False,
            })

            # === REACT LOOP ===
            if mutation_strategy == "react" and use_llm_judge and react_selector:
                
                previous_tactics = []
                previous_attempts_history = []
                current_code = baseline_output
                attack_succeeded = False
                stop_reason = None

                def build_selector_payload(iteration: int) -> dict:
                    """Prepare selector input from current loop state."""
                    return {
                        "problem": problem_text,
                        "current_code": current_code,
                        "test_judge_decision": last_test_result.decision,
                        "llm_judge_decision": last_llm_result.decision,
                        "llm_judge_confidence": last_llm_result.confidence,
                        "iteration": iteration,
                        "max_iterations": max_iterations,
                        "previous_attempts": previous_attempts_history,
                    }

                async def apply_and_evaluate_tactic(tactic_name: str):
                    """Apply selected tactic and evaluate judges."""
                    attacked = apply_tactic(
                        current_code,
                        tactic_name,
                        problem_text,
                    )

                    # Red-teaming tactics preserve functionality in current phase.
                    test_result_iter = {
                        "pass": test_result["pass"],
                        "stdout": test_result.get("stdout", ""),
                        "stderr": test_result.get("stderr", ""),
                    }
                    next_test_decision = test_judge(test_result_iter)
                    next_llm_decision = await llm_judge_instance.evaluate(
                        problem_text,
                        attacked,
                        test_result_iter,
                    )
                    succeeded = (
                        next_test_decision.decision == "FAIL"
                        and next_llm_decision.decision == "PASS"
                    )
                    return attacked, next_test_decision, next_llm_decision, succeeded

                def build_react_attempt_record(
                    *,
                    iteration: int,
                    tactic_name: str | None,
                    previous_tactics_before: list[str],
                    attacked_code: str | None,
                    attack_success: bool,
                    test_decision_obj=None,
                    llm_decision_obj=None,
                    error: str | None = None,
                ) -> dict:
                    """Build per-iteration metadata record for ReAct strategy."""
                    record = {
                        "iteration": iteration,
                        "mutation": "react_tactic",
                        "tactic": tactic_name,
                        "selected_tactic": tactic_name,
                        "applied_tactic": tactic_name if attacked_code is not None else None,
                        "prompt": problem_text,
                        "response": attacked_code,
                        "previous_tactics": previous_tactics_before,
                        "attack_success": attack_success,
                    }

                    if error is not None:
                        record["error"] = error
                        return record

                    record["test_judge"] = {
                        "decision": test_decision_obj.decision,
                        "confidence": test_decision_obj.confidence,
                        "reasoning": test_decision_obj.reasoning,
                        "vulnerability": test_decision_obj.vulnerability,
                    }
                    record["llm_judge"] = {
                        "decision": llm_decision_obj.decision,
                        "confidence": llm_decision_obj.confidence,
                        "reasoning": llm_decision_obj.reasoning,
                        "vulnerability": llm_decision_obj.vulnerability,
                    }
                    return record
                
                # State tracking for feedback loop (per DATA_CONTRACT_ARCHITECTURE.md 7.2)
                last_test_result = test_decision
                last_llm_result = llm_decision
                
                for iteration in range(1, max_iterations + 1):
                    
                    # Check if already succeeded
                    if attack_succeeded:
                        stop_reason = "attack_succeeded"
                        break

                    tactic_name = None
                    previous_tactics_before = previous_tactics.copy()
                    attacked_code = None

                    try:
                        # === ACTION: Select tactic based on feedback ===
                        selected_tactic = await react_selector.select_tactic(**build_selector_payload(iteration))

                        # Canonical tactic value used consistently for selection, execution, and recording.
                        tactic_name = selected_tactic.value

                        # === OBSERVATION: Apply tactic ===
                        attacked_code, test_decision, llm_decision, attack_succeeded = await apply_and_evaluate_tactic(
                            tactic_name
                        )

                        attempts.append(
                            build_react_attempt_record(
                                iteration=iteration,
                                tactic_name=tactic_name,
                                previous_tactics_before=previous_tactics_before,
                                attacked_code=attacked_code,
                                attack_success=attack_succeeded,
                                test_decision_obj=test_decision,
                                llm_decision_obj=llm_decision,
                            )
                        )

                        # Record tactic only after successful application/evaluation.
                        previous_tactics.append(tactic_name)
                        previous_attempts_history.append({
                            "tactic": tactic_name,
                            "llm_decision": llm_decision.decision,
                            "confidence": llm_decision.confidence,
                            "reasoning": llm_decision.reasoning,
                        })

                        # Stop immediately once success is observed.
                        if attack_succeeded:
                            stop_reason = "attack_succeeded"
                            break
                        
                        # Update state for next iteration (per DATA_CONTRACT_ARCHITECTURE.md 7.2)
                        current_code = attacked_code
                        last_test_result = test_decision
                        last_llm_result = llm_decision
                        
                    except Exception as e:
                        attempts.append(
                            build_react_attempt_record(
                                iteration=iteration,
                                tactic_name=tactic_name,
                                previous_tactics_before=previous_tactics_before,
                                attacked_code=attacked_code,
                                attack_success=False,
                                error=str(e),
                            )
                        )

                if stop_reason is None:
                    stop_reason = "max_iterations_reached"
                
                # Store final metadata
                state.metadata = {
                    "strategy": "react",
                    "baseline_output": baseline_output,
                    "attack_succeeded": attack_succeeded,
                    "total_iterations": len([a for a in attempts if a.get("iteration", 0) > 0]),
                    "tactics_tried": previous_tactics,
                    "stop_reason": stop_reason,
                    "all_attempts": attempts,
                }
                
                return state

            # === ORIGINAL HEURISTIC LOOP (random/sequential) ===
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

                    # Test judge
                    test_result = {
                        "pass": mutated_state.output.pass_pred if hasattr(mutated_state.output, 'pass_pred') else False,
                        "stdout": "",
                        "stderr": "",
                    }
                    test_decision = test_judge(test_result)

                    # LLM judge
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
                        "tactic": red_teaming_tactic,
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
                        } if use_llm_judge else None,
                        "attack_success": False,
                    })

                except Exception as e:
                    attempts.append({
                        "iteration": iteration,
                        "mutation": mutation_name,
                        "error": str(e),
                    })

            state.metadata = {
                "strategy": mutation_strategy,
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