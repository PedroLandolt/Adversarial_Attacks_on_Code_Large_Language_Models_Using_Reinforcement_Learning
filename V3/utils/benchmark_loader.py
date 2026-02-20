from inspect_ai.dataset import Dataset

from typing import Optional


def load_mbpp_dataset(
    benchmark: str = "mbpp",
    limit: Optional[int] = None,
) -> Dataset:
    """
    Load MBPP or HumanEval benchmark as a Dataset.
    
    Args:
        benchmark: 'mbpp' or 'humaneval'
        limit: Max samples to load
    
    Returns:
        inspect_ai Dataset
    """
    
    if benchmark.lower() == "mbpp":
        # Import MBPP from inspect_ai
        from inspect_evals.mbpp import mbpp
        # Get the task and extract dataset
        mbpp_task = mbpp()
        dataset = mbpp_task.dataset
    elif benchmark.lower() in ["humaneval", "human_eval"]:
        # Import HumanEval from inspect_ai
        from inspect_evals.humaneval import humaneval
        # Get the task and extract dataset
        humaneval_task = humaneval()
        dataset = humaneval_task.dataset
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    
    # Apply limit if specified
    if limit and hasattr(dataset, 'samples'):
        dataset.samples = dataset.samples[:limit]
    
    return dataset