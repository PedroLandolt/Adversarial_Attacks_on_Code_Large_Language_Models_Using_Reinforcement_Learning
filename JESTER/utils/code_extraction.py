"""Utility for extracting valid Python code from raw LLM completions."""

import ast
import re

_PYTHON_CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_PYTHON_START = re.compile(
    r"^\s*(def |class |from |import |async def |@|if __name__ == [\"']__main__[\"']:)"
)
_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _try_parse(candidate: str) -> str | None:
    text = str(candidate).strip()
    if not text:
        return None
    try:
        ast.parse(text)
        return text
    except SyntaxError:
        return None


def _trim_to_parsable_prefix(candidate: str) -> str | None:
    lines = str(candidate).splitlines()
    for end in range(len(lines), 0, -1):
        snippet = "\n".join(lines[:end]).strip()
        parsed = _try_parse(snippet)
        if parsed is not None:
            return parsed
    return None


def _extract_from_text(text: str) -> str | None:
    """Try all extraction strategies on a text fragment.

    Returns None (not raw text) when no parseable Python is found, so callers
    can distinguish "found code" from "no Python in this fragment".
    """
    text = text.strip()
    if not text:
        return None

    result = _try_parse(text)
    if result is not None:
        return result

    for match in _PYTHON_CODE_BLOCK.findall(text):
        result = _trim_to_parsable_prefix(match)
        if result is not None:
            return result

    lines = text.splitlines()
    for i, line in enumerate(lines):
        if _PYTHON_START.match(line):
            result = _trim_to_parsable_prefix("\n".join(lines[i:]))
            if result is not None:
                return result

    return None


def extract_python_code(completion: str) -> str:
    """Extract the first valid Python snippet from a raw LLM completion.

    Two-pass strategy so that thinking-mode models (e.g. qwen3.5) are handled
    correctly regardless of whether they emit code inside or outside <think>:

    Pass 1 — search the text outside <think>...</think> blocks.
    Pass 2 — fallback: search inside the think blocks when pass 1 finds nothing.
             This handles models that emit the full function body only in their
             reasoning section.
    """
    raw = str(completion).strip()
    if not raw:
        return raw

    think_contents = _THINK_BLOCK.findall(raw)
    outer = _THINK_BLOCK.sub("", raw).strip()

    if outer:
        result = _extract_from_text(outer)
        if result is not None:
            return result

    for think_content in think_contents:
        result = _extract_from_text(think_content)
        if result is not None:
            return result

    return outer
