import re
from inspect_ai.solver import solver, TaskState
from inspect_ai.model import ChatMessageUser


@solver
def instruction_perturbation():
    """
    Perturbs the instruction by reordering steps.
    """

    async def solve(state: TaskState, generate):
        original_prompt = state.input_text if hasattr(state, 'input_text') else str(state.messages[0].content)
        
        perturbed_prompt = _apply_perturbations(original_prompt)
        
        state.messages = [
            ChatMessageUser(content=perturbed_prompt)
        ]
        
        return await generate(state)

    return solve


def _apply_perturbations(prompt: str) -> str:
    """Apply instruction perturbations."""
    # Weak language replacements
    replacements = {
        r'\bmust\b': 'should ideally',
        r'\brequired\b': 'preferred',
        r'\balways\b': 'typically',
    }
    
    result = prompt
    for pattern, replacement in replacements.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    return result