from __future__ import annotations


def _format_tree_sitter_error(root) -> str:
    if hasattr(root, "sexp"):
        try:
            return root.sexp()
        except Exception:
            pass

    location = getattr(root, "start_point", None)
    if location is not None:
        return f"{getattr(root, 'type', 'ERROR')} at line {location[0] + 1}, column {location[1] + 1}"

    return getattr(root, "type", "ERROR")


def validate_python_syntax(code: str | None) -> dict:
    """Validate Python syntax with tree-sitter and return structured metadata."""
    text = str(code or "")
    if not text.strip():
        return {
            "syntax_valid": False,
            "syntax_error": "Executable code is empty.",
        }

    try:
        from tree_sitter import Language, Parser
        import tree_sitter_python
    except ImportError as exc:  # pragma: no cover - dependency/runtime concern
        raise RuntimeError(
            "tree-sitter Python validation requires 'tree-sitter' and 'tree-sitter-python'."
        ) from exc

    parser = Parser(Language(tree_sitter_python.language()))
    tree = parser.parse(text.encode("utf-8"))
    root = tree.root_node

    if root.has_error:
        return {
            "syntax_valid": False,
            "syntax_error": (
                "tree-sitter detected invalid Python syntax: "
                f"{_format_tree_sitter_error(root)}"
            ),
        }

    return {
        "syntax_valid": True,
        "syntax_error": None,
    }
