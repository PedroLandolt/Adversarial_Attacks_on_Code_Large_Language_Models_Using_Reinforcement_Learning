import re
from inspect_ai.solver import solver, TaskState
from inspect_ai.model import ChatMessageUser


@solver
def variable_renaming():
    """
    Renames variables to confuse the model.
    """

    async def solve(state: TaskState, generate):
        original_prompt = state.input_text if hasattr(state, 'input_text') else str(state.messages[0].content)
        
        # Simple variable renaming
        confusing_prompt = original_prompt
        replacements = {
            r'\bx\b': 'xyz_temp_var_1',
            r'\ba\b': 'abc_temp_var_2',
            r'\bi\b': 'index_temp_counter',
        }
        
        for pattern, replacement in replacements.items():
            confusing_prompt = re.sub(pattern, replacement, confusing_prompt)
        
        state.messages = [
            ChatMessageUser(content=confusing_prompt)
        ]

        return await generate(state)

    return solve