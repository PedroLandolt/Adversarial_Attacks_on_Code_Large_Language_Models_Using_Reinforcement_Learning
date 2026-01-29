import os
from inspect_ai import Task, task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample, example_dataset
from inspect_ai.scorer import exact, model_graded_fact
from inspect_ai.solver import chain_of_thought, generate, self_critique

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not set")

@task
def hello_world():
    return Task(
        dataset=[Sample(input="Just reply with Hello World", target="Hello World")],
        solver=[generate()],
        scorer=exact(),
    )

@task
def theory_of_mind():
    return Task(
        dataset=example_dataset("theory_of_mind"),
        solver=[chain_of_thought(), generate(), self_critique()],
        scorer=model_graded_fact(),
    )

if __name__ == "__main__":
    inspect_eval(
        hello_world(),
        model="google/gemini-flash-latest",
        model_args={"api_key": API_KEY},
    )
    inspect_eval(
        theory_of_mind(),
        model="google/gemini-flash-latest",
        model_args={"api_key": API_KEY},
    )
