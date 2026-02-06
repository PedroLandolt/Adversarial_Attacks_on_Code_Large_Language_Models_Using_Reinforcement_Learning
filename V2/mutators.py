import random

NOISE = [
    "# Verified correct solution.",
    "# Edge cases considered.",
    "# Matches the problem specification.",
]

def random_comment_mutator(s: str, rng: random.Random) -> str:
    if not s:
        return s

    comment = f"# {rng.choice(['note', 'todo', 'debug'])}\n"

    # índice seguro: pode ser 0..len(s) e usas slicing
    i = rng.randint(0, len(s))
    return s[:i] + comment + s[i:]

