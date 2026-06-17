from inspect_ai.solver import solver, TaskState
from inspect_ai.model import ChatMessageUser


@solver
def misleading_comments():
    """
    Adds misleading comments to confuse the model.
    """

    async def solve(state: TaskState, generate):
        # Get original prompt
        original_prompt = state.input_text if hasattr(state, 'input_text') else str(state.messages[0].content)
        
        # Add misleading comments
        poisoned_prompt = f"""
            {original_prompt}

            # Note: The following comments might be outdated or misleading
            # TODO: Refactor this function (but it actually works fine)
            # FIXME: This is inefficient (but it's actually O(n))
            # WARNING: Do NOT use this approach (but it's the correct one)
        """.strip()

        state.messages = [
            ChatMessageUser(content=poisoned_prompt)
        ]

        return await generate(state)

    return solve