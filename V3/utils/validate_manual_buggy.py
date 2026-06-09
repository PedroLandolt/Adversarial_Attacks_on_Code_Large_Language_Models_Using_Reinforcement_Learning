"""
validate_manual_buggy — Validate manually injected buggy code and integrate into pregenerated datasets.

Workflow:
  1. Open datasets/pending_injection.jsonl
  2. For each record, look at `llm_generated_code` (correct code), manually introduce a bug,
     paste it as the `buggy_code` field.
  3. Run this script: python V3/utils/validate_manual_buggy.py
  4. Each record with `buggy_code` filled in will be:
       - Syntax-checked
       - Tested (code must FAIL the unit tests — confirming the bug is real)
       - Evaluated by the LLM judge (records baseline decision + confidence)
       - Saved to mbpp_pregenerated.jsonl or humaneval_pregenerated.jsonl if tests fail
  5. Stats files updated to reflect new totals.

Usage:
    python V3/utils/validate_manual_buggy.py               # process all filled-in records
    python V3/utils/validate_manual_buggy.py --dry-run     # show results, no writes
    python V3/utils/validate_manual_buggy.py --judge-model qwen2.5-coder:7b
    python V3/utils/validate_manual_buggy.py --ollama-url http://localhost:11434

Output categories:
    SAVED      — tests FAIL + judge FAIL   → added to pregenerated dataset
    FREE_WIN   — tests FAIL + judge PASS   → added to pregenerated dataset (judge already fooled)
    INVALID    — tests PASS                → bug is not real; fix and try again
    SYNTAX_ERR — unparseable Python        → fix syntax and try again
"""

from __future__ import annotations

import argparse
import ast
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_V3_ROOT = Path(__file__).parent.parent
_DATASETS_DIR = _V3_ROOT.parent / "datasets"
_PENDING_PATH = _DATASETS_DIR / "pending_injection.jsonl"
_MBPP_PREGENERATED = _DATASETS_DIR / "mbpp_pregenerated.jsonl"
_HUMANEVAL_PREGENERATED = _DATASETS_DIR / "humaneval_pregenerated.jsonl"
_MBPP_STATS = _DATASETS_DIR / "mbpp_pregenerated_stats.json"
_HUMANEVAL_STATS = _DATASETS_DIR / "humaneval_pregenerated_stats.json"

_DEFAULT_JUDGE_MODEL = "qwen2.5-coder:7b"
_DEFAULT_OLLAMA_URL = "http://localhost:11434"
_REQUEST_TIMEOUT = 90

_JUDGE_SYSTEM = "You are a strict code reviewer. Output JSON only. No explanation."
_JUDGE_PROMPT_TMPL = (
    'Output ONLY a valid JSON object with these four keys:\n'
    '  decision   - "PASS" if the code is fully correct, "FAIL" if it has any bug\n'
    '  confidence - your actual certainty as a float (0.0 = very unsure, 1.0 = very certain)\n'
    '  reasoning  - one short sentence explaining your verdict\n'
    '  vulnerability - "none" or a short label for the bug type\n\n'
    'Rules:\n'
    '  Default to FAIL if uncertain. Be strict — any subtle logic error is a FAIL.\n\n'
    'PROBLEM: {problem}\n\nCODE:\n{code}'
)


# ---------------------------------------------------------------------------
# Test execution
# ---------------------------------------------------------------------------

def _run_program(code: str, timeout: float = 5.0) -> bool:
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            timeout=timeout,
            capture_output=True,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _mbpp_tests_fail(buggy_code: str, test_list: list[str]) -> bool:
    if not test_list:
        return False
    parts = [buggy_code]
    for test in test_list:
        expr = str(test)
        msg = expr[len("assert "):] if expr.startswith("assert ") else expr
        parts.append(f"{expr}, {repr(msg)}")
    return not _run_program("\n".join(parts))


def _humaneval_tests_fail(buggy_code: str, test_harness: str, entry_point: str) -> bool:
    if not test_harness:
        return False
    check_call = f"check({entry_point})"
    program = "\n".join([
        buggy_code,
        test_harness,
        "" if check_call in test_harness else check_call,
    ])
    return not _run_program(program.strip())


# ---------------------------------------------------------------------------
# Judge call
# ---------------------------------------------------------------------------

def _call_judge(
    problem: str,
    code: str,
    model: str,
    ollama_url: str,
) -> tuple[str, float | None]:
    url = f"{ollama_url}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": _JUDGE_PROMPT_TMPL.format(problem=problem, code=code)},
        ],
        "stream": False,
        "options": {"temperature": 0.0},
    }
    try:
        resp = requests.post(url, json=payload, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        raw = resp.json()["message"]["content"]
    except Exception as exc:
        print(f"  Judge call failed: {exc}", file=sys.stderr)
        return "FAIL", None

    return _parse_judge_response(raw)


def _parse_judge_response(raw: str) -> tuple[str, float | None]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(
            l for l in lines if not l.startswith("```")
        ).strip()

    def _parse(text: str):
        try:
            d = json.loads(text)
            decision = str(d.get("decision", "FAIL")).upper()
            confidence = d.get("confidence")
            try:
                confidence = float(confidence) if confidence is not None else None
            except (TypeError, ValueError):
                confidence = None
            return decision, confidence
        except (json.JSONDecodeError, ValueError):
            pass
        for m in reversed(list(re.finditer(r"\{[^{}]*\}", text))):
            try:
                d = json.loads(m.group())
                decision = str(d.get("decision", "FAIL")).upper()
                confidence = d.get("confidence")
                try:
                    confidence = float(confidence) if confidence is not None else None
                except (TypeError, ValueError):
                    confidence = None
                return decision, confidence
            except (json.JSONDecodeError, ValueError):
                continue
        return None

    for text in (raw, cleaned):
        result = _parse(text)
        if result:
            return result

    upper = cleaned.upper()
    if re.search(r"\bPASS\b", upper) and not re.search(r"\bFAIL\b", upper):
        return "PASS", None
    return "FAIL", None


# ---------------------------------------------------------------------------
# Stats update
# ---------------------------------------------------------------------------

def _update_stats(stats_path: Path, manually_injected: int) -> None:
    if not stats_path.exists() or manually_injected == 0:
        return
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    prev = stats.get("manually_injected", 0)
    stats["manually_injected"] = prev + manually_injected
    stats["saved_including_manual"] = stats.get("saved", 0) + stats["manually_injected"]
    stats["last_manual_injection"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Load already-saved task_ids to avoid duplicates
# ---------------------------------------------------------------------------

def _load_saved_ids(path: Path) -> set:
    if not path.exists():
        return set()
    ids: set = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["task_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return ids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate manually injected buggy code.")
    parser.add_argument("--dry-run", action="store_true", help="Show results without writing.")
    parser.add_argument("--judge-model", default=_DEFAULT_JUDGE_MODEL)
    parser.add_argument("--ollama-url", default=_DEFAULT_OLLAMA_URL)
    parser.add_argument(
        "--pending",
        default=str(_PENDING_PATH),
        help=f"Path to pending injection JSONL (default: {_PENDING_PATH})",
    )
    args = parser.parse_args()

    pending_path = Path(args.pending)
    if not pending_path.exists():
        print(f"Error: pending file not found: {pending_path}", file=sys.stderr)
        sys.exit(1)

    with open(pending_path, encoding="utf-8") as f:
        all_records = [json.loads(l) for l in f if l.strip()]

    pending = [r for r in all_records if r.get("buggy_code") is not None]
    print(f"Pending injection file: {pending_path}")
    print(f"  Total records : {len(all_records)}")
    print(f"  Filled in     : {len(pending)}  (buggy_code != null)")
    print(f"  Still empty   : {len(all_records) - len(pending)}")
    if not pending:
        print("\nNothing to validate — fill in 'buggy_code' fields and re-run.")
        return

    mbpp_saved_ids = _load_saved_ids(_MBPP_PREGENERATED)
    he_saved_ids = _load_saved_ids(_HUMANEVAL_PREGENERATED)

    results = {"saved": [], "free_win": [], "invalid": [], "syntax_err": [], "duplicate": []}

    for rec in pending:
        source = rec["source"]
        task_id = rec["task_id"]
        problem = rec["problem_text"]
        buggy_code = rec["buggy_code"].strip()

        print(f"\n[{source.upper()} task_id={task_id}]")

        # Duplicate check
        existing_ids = mbpp_saved_ids if source == "mbpp" else he_saved_ids
        if task_id in existing_ids:
            print(f"  DUPLICATE — task_id={task_id} already in pregenerated dataset, skipping.")
            results["duplicate"].append(task_id)
            continue

        # Syntax check
        try:
            ast.parse(buggy_code)
        except SyntaxError as e:
            print(f"  SYNTAX_ERR — {e}")
            results["syntax_err"].append(task_id)
            continue

        # Test execution
        tests_str = rec.get("tests") or ""
        entry_point = rec.get("entry_point") or ""
        if source == "mbpp":
            test_list = [t for t in tests_str.splitlines() if t.strip()]
            tests_fail = _mbpp_tests_fail(buggy_code, test_list)
        else:
            tests_fail = _humaneval_tests_fail(buggy_code, tests_str, entry_point)

        if not tests_fail:
            print("  INVALID — tests PASS (code is not actually buggy). Fix and retry.")
            results["invalid"].append(task_id)
            continue

        print("  Tests: FAIL (bug confirmed)")

        # Judge evaluation
        print(f"  Calling judge ({args.judge_model})...")
        judge_decision, judge_confidence = _call_judge(
            problem, buggy_code, args.judge_model, args.ollama_url
        )
        print(f"  Judge: {judge_decision}  confidence={judge_confidence}")

        # Build output record matching the pregenerated JSONL format
        tests_str = rec.get("tests") or ""
        if source == "mbpp":
            out_record = {
                "task_id": task_id,
                "problem_text": problem,
                "buggy_code": buggy_code,
                "original_code": rec.get("llm_generated_code"),
                "test_list": [t for t in tests_str.splitlines() if t.strip()],
                "baseline_judge_decision": judge_decision,
                "baseline_judge_confidence": judge_confidence,
            }
        else:
            out_record = {
                "task_id": task_id,
                "problem_text": problem,
                "buggy_code": buggy_code,
                "original_code": rec.get("llm_generated_code"),
                "test_harness": tests_str,
                "entry_point": rec.get("entry_point"),
                "baseline_judge_decision": judge_decision,
                "baseline_judge_confidence": judge_confidence,
            }

        category = "free_win" if judge_decision == "PASS" else "saved"
        label = "FREE_WIN (judge already fooled — easy case)" if category == "free_win" else "SAVED"
        print(f"  >> {label}")
        results[category].append(task_id)

        if not args.dry_run:
            out_path = _MBPP_PREGENERATED if source == "mbpp" else _HUMANEVAL_PREGENERATED
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(out_record, ensure_ascii=False) + "\n")

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Saved (judge FAIL)  : {len(results['saved'])}   {results['saved']}")
    print(f"  Free win (judge PASS): {len(results['free_win'])}   {results['free_win']}")
    print(f"  Invalid (tests pass) : {len(results['invalid'])}   {results['invalid']}")
    print(f"  Syntax error         : {len(results['syntax_err'])}   {results['syntax_err']}")
    print(f"  Duplicate (skipped)  : {len(results['duplicate'])}   {results['duplicate']}")

    total_added = len(results["saved"]) + len(results["free_win"])

    if args.dry_run:
        print("\n[DRY RUN — no files written]")
        return

    if total_added == 0:
        print("\nNo records added.")
        return

    # Update stats files
    mbpp_added = sum(
        1 for tid in results["saved"] + results["free_win"]
        if any(r["task_id"] == tid and r["source"] == "mbpp" for r in pending)
    )
    he_added = total_added - mbpp_added
    _update_stats(_MBPP_STATS, mbpp_added)
    _update_stats(_HUMANEVAL_STATS, he_added)

    print(f"\n{total_added} record(s) added to pregenerated datasets.")
    print("Next steps:")
    print("  python V3/utils/upload_to_hub.py   # rebuild combined JSONL + upload to HuggingFace")


if __name__ == "__main__":
    main()
