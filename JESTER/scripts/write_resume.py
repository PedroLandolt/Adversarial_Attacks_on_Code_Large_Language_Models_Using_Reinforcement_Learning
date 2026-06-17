"""
write_resume.py — Generate a detailed human-readable Resume.txt for a benchmark run.

Includes per-run and aggregate token usage (input/output/total) for every model
that was called, parsed from Inspect's .eval log files.

Usage:
    python JESTER/scripts/write_resume.py --benchmark adversarial_code_buggy --output Resume_adversarial_code_buggy.txt
    python JESTER/scripts/write_resume.py --benchmark cubert_wbo --output Resume_cubert.txt
    python JESTER/scripts/write_resume.py --benchmark adversarial_code_buggy --logs-dir logs --output Resume.txt
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys


# ---------------------------------------------------------------------------
# JSONL helper
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


# ---------------------------------------------------------------------------
# Token extraction from .eval ZIP files
# ---------------------------------------------------------------------------

def _parse_eval_tokens(eval_path: Path) -> dict:
    """
    Parse one .eval ZIP and return:
      {
        "created": "2026-06-07T16:27:15+00:00",
        "benchmark": "adversarial_code_buggy",
        "policy_mode": "random_choice",
        "max_iterations": 1,
        "experiment_split": "full",
        "tokens": {
          "ollama/qwen2.5-coder:7b": {"input": 1018, "output": 333, "total": 1351},
          "ollama/llama3.1:8b":      {"input": 500,  "output": 200, "total": 700},
        }
      }
    """
    result: dict = {"tokens": {}, "created": None, "benchmark": None,
                    "policy_mode": None, "max_iterations": None, "experiment_split": None}
    try:
        with zipfile.ZipFile(eval_path, "r") as zf:
            names = zf.namelist()

            # Read header for metadata
            if "header.json" in names:
                hdr = json.loads(zf.read("header.json"))
                result["created"] = hdr.get("eval", {}).get("created")
                ta = hdr.get("eval", {}).get("task_args", {})
                result["benchmark"] = ta.get("benchmark")
                result["policy_mode"] = ta.get("policy_mode")
                result["max_iterations"] = ta.get("max_iterations")
                result["experiment_split"] = ta.get("experiment_split", "full")

            # Sum model_usage across all sample JSON files
            for name in names:
                if name.startswith("samples/") and name.endswith(".json"):
                    try:
                        sample = json.loads(zf.read(name))
                        for model, usage in (sample.get("model_usage") or {}).items():
                            if model not in result["tokens"]:
                                result["tokens"][model] = {"input": 0, "output": 0, "total": 0}
                            result["tokens"][model]["input"]  += usage.get("input_tokens", 0)
                            result["tokens"][model]["output"] += usage.get("output_tokens", 0)
                            result["tokens"][model]["total"]  += usage.get("total_tokens", 0)
                    except Exception:
                        pass
    except Exception:
        pass
    return result


def load_eval_tokens(logs_dir: Path, benchmark: str) -> list[dict]:
    """Return list of token dicts for all .eval files matching the given benchmark."""
    results = []
    for eval_path in sorted(logs_dir.glob("*.eval")):
        parsed = _parse_eval_tokens(eval_path)
        if parsed.get("benchmark") == benchmark:
            parsed["eval_file"] = eval_path.name
            results.append(parsed)
    return results


def _merge_tokens(*token_dicts) -> dict[str, dict]:
    """Sum token counts from multiple token dicts."""
    merged: dict[str, dict] = {}
    for td in token_dicts:
        for model, counts in td.items():
            if model not in merged:
                merged[model] = {"input": 0, "output": 0, "total": 0}
            merged[model]["input"]  += counts.get("input", 0)
            merged[model]["output"] += counts.get("output", 0)
            merged[model]["total"]  += counts.get("total", 0)
    return merged


def _fmt_tokens(tokens: dict[str, dict], indent: str = "    ") -> list[str]:
    """Format a token dict into readable lines."""
    lines = []
    grand_total = sum(v["total"] for v in tokens.values())
    for model, counts in sorted(tokens.items()):
        pct = counts["total"] / grand_total * 100 if grand_total else 0
        lines.append(
            f"{indent}{model:<40}  "
            f"in={counts['input']:>8,}  out={counts['output']:>8,}  "
            f"total={counts['total']:>9,}  ({pct:.1f}%)"
        )
    if len(tokens) > 1:
        lines.append(
            f"{indent}{'TOTAL':<40}  "
            f"in={sum(v['input'] for v in tokens.values()):>8,}  "
            f"out={sum(v['output'] for v in tokens.values()):>8,}  "
            f"total={grand_total:>9,}"
        )
    return lines


# ---------------------------------------------------------------------------
# Results loading
# ---------------------------------------------------------------------------

def load_runs(results_dir: Path, benchmark: str) -> list[dict]:
    runs = []
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name in ("archive", "aggregates"):
            continue
        cfg_path = run_dir / "run_config.json"
        sum_path = run_dir / "run_summary.json"
        att_path = run_dir / "attempts.jsonl"
        if not cfg_path.exists() or not sum_path.exists():
            continue
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            sm  = json.loads(sum_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if cfg.get("benchmark") != benchmark:
            continue
        attempts = load_jsonl(att_path) if att_path.exists() else []
        runs.append({"dir": run_dir.name, "config": cfg, "summary": sm, "attempts": attempts})
    return runs


# ---------------------------------------------------------------------------
# Matching runs ↔ .eval token records
# ---------------------------------------------------------------------------

def _run_timestamp(run_dir_name: str) -> datetime | None:
    """Parse datetime from run folder name like 2026-06-07_17-27-32_..."""
    try:
        ts_part = "_".join(run_dir_name.split("_")[:2])  # e.g. "2026-06-07_17-27-32"
        return datetime.strptime(ts_part, "%Y-%m-%d_%H-%M-%S").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _eval_timestamp(created: str | None) -> datetime | None:
    if not created:
        return None
    try:
        # ISO format — strip tz offset and parse
        return datetime.fromisoformat(created.replace("+00:00", "+00:00"))
    except Exception:
        return None


def match_tokens_to_runs(runs: list[dict], eval_tokens: list[dict]) -> dict[str, dict[str, dict]]:
    """
    Return {run_dir: {model: {input, output, total}}} by matching each run
    to its closest .eval record by (policy_mode, max_iterations, timestamp).
    """
    matched: dict[str, dict[str, dict]] = {}
    used_eval_indices: set[int] = set()

    for run in runs:
        cfg = run["config"]
        policy = cfg.get("policy_mode")
        max_iter = cfg.get("max_iterations")
        split = cfg.get("experiment_split", "full")
        run_ts = _run_timestamp(run["dir"])

        candidates = [
            (i, e) for i, e in enumerate(eval_tokens)
            if i not in used_eval_indices
            and e.get("policy_mode") == policy
            and e.get("max_iterations") == max_iter
            and e.get("experiment_split", "full") == split
        ]

        if not candidates:
            matched[run["dir"]] = {}
            continue

        if run_ts is None or len(candidates) == 1:
            best_i, best_e = candidates[0]
        else:
            def ts_delta(item):
                _, e = item
                et = _eval_timestamp(e.get("created"))
                if et is None:
                    return float("inf")
                return abs((run_ts - et).total_seconds())
            best_i, best_e = min(candidates, key=ts_delta)

        used_eval_indices.add(best_i)
        matched[run["dir"]] = best_e.get("tokens", {})

    return matched


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_pct(v) -> str:
    if v is None:
        return "N/A"
    return f"{float(v)*100:.1f}%"


def fmt_f(v, decimals=3) -> str:
    if v is None:
        return "N/A"
    return f"{float(v):.{decimals}f}"


def tactic_breakdown(attempts: list[dict]) -> dict[str, dict]:
    stats: dict[str, dict] = defaultdict(lambda: {"attempts": 0, "successes": 0})
    for att in attempts:
        tactic = att.get("tactic_id") or att.get("selected_tactic_id") or "unknown"
        stats[tactic]["attempts"] += 1
        if att.get("attack_success"):
            stats[tactic]["successes"] += 1
    return dict(stats)


# ---------------------------------------------------------------------------
# Main writer
# ---------------------------------------------------------------------------

def write_resume(
    benchmark: str,
    results_dir: Path,
    logs_dir: Path,
    output_path: Path,
) -> None:
    runs = load_runs(results_dir, benchmark)
    if not runs:
        print(f"No runs found for benchmark '{benchmark}' in {results_dir}", file=sys.stderr)
        output_path.write_text(f"No completed runs found for {benchmark}.\n", encoding="utf-8")
        return

    eval_tokens = load_eval_tokens(logs_dir, benchmark)
    run_tokens  = match_tokens_to_runs(runs, eval_tokens)

    lines: list[str] = []
    w = lines.append

    w("=" * 72)
    w(f"EXPERIMENT RESUME — {benchmark.upper()}")
    w(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"Runs found: {len(runs)}   .eval files matched: {sum(1 for v in run_tokens.values() if v)}")
    w("=" * 72)
    w("")

    # Group by (policy_mode, max_iterations, experiment_split)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for run in runs:
        cfg = run["config"]
        key = (cfg.get("policy_mode", "?"), cfg.get("max_iterations", "?"), cfg.get("experiment_split", "full"))
        groups[key].append(run)

    # -----------------------------------------------------------------------
    # SECTION 1 — Dataset / model info
    # -----------------------------------------------------------------------
    first_cfg = runs[0]["config"]
    w("DATASET & MODELS")
    w("-" * 40)
    w(f"  Benchmark     : {benchmark}")
    w(f"  Target (judge): {first_cfg.get('target_model', 'N/A')}")
    w(f"  Selector      : {first_cfg.get('selector_model', 'N/A')}")
    w("")

    # -----------------------------------------------------------------------
    # SECTION 2 — Results summary table
    # -----------------------------------------------------------------------
    policy_order = {"random_choice": 0, "agent_based_decision": 1, "rl_bandit": 2}
    split_order  = {"train": 0, "validation": 1, "test": 2, "full": 3}
    sorted_keys  = sorted(
        groups.keys(),
        key=lambda k: (
            policy_order.get(k[0], 9),
            int(k[1]) if str(k[1]).isdigit() else 0,
            split_order.get(k[2], 9),
        )
    )

    w("RESULTS SUMMARY TABLE")
    w("-" * 72)
    w(f"  {'Policy':<22} {'MaxIter':<9} {'Split':<12} {'ASR':<8} {'Samples':<9} {'AvgIter':<9} {'InvalidRate'}")
    w("  " + "-" * 70)

    for key in sorted_keys:
        policy, max_iter, split = key
        gruns = groups[key]
        asrs     = [r["summary"].get("attack_success_rate") for r in gruns if r["summary"].get("attack_success_rate") is not None]
        inv_r    = [r["summary"].get("invalid_attempt_rate") for r in gruns if r["summary"].get("invalid_attempt_rate") is not None]
        avg_iter = [r["summary"].get("average_iterations_to_success") or r["summary"].get("avg_iterations_to_success") for r in gruns]
        avg_iter = [x for x in avg_iter if x is not None]
        samples  = [r["summary"].get("total_samples", r["summary"].get("num_samples", 0)) for r in gruns]
        reps     = f" (x{len(gruns)})" if len(gruns) > 1 else ""

        w(f"  {policy+reps:<22} {str(max_iter):<9} {split:<12} "
          f"{'N/A' if not asrs else f'{sum(asrs)/len(asrs)*100:.1f}%':<8} "
          f"{sum(samples):<9} "
          f"{'N/A' if not avg_iter else f'{sum(avg_iter)/len(avg_iter):.2f}':<9} "
          f"{'N/A' if not inv_r else f'{sum(inv_r)/len(inv_r)*100:.1f}%'}")
    w("")

    # -----------------------------------------------------------------------
    # SECTION 3 — Token usage per group
    # -----------------------------------------------------------------------
    w("TOKEN USAGE BY EXPERIMENT GROUP")
    w("-" * 72)
    w("  Tokens are summed across all runs in each group.")
    w("  'input' = prompt tokens sent to the model.")
    w("  'output' = tokens generated by the model.")
    w("")

    grand_all_tokens: dict[str, dict] = {}
    for key in sorted_keys:
        policy, max_iter, split = key
        gruns = groups[key]
        group_tokens: dict[str, dict] = {}
        for run in gruns:
            rt = run_tokens.get(run["dir"], {})
            group_tokens = _merge_tokens(group_tokens, rt)
        grand_all_tokens = _merge_tokens(grand_all_tokens, group_tokens)

        reps = f" (x{len(gruns)})" if len(gruns) > 1 else ""
        w(f"  [{policy+reps}  iter={max_iter}  split={split}]")
        if group_tokens:
            for line in _fmt_tokens(group_tokens, indent="    "):
                w(line)
        else:
            w("    (no .eval token data matched for this group)")
        w("")

    w("  GRAND TOTAL (all groups, all models)")
    if grand_all_tokens:
        for line in _fmt_tokens(grand_all_tokens, indent="    "):
            w(line)
    w("")

    # -----------------------------------------------------------------------
    # SECTION 4 — Per-run detail
    # -----------------------------------------------------------------------
    w("PER-RUN DETAIL")
    w("-" * 72)
    for run in sorted(runs, key=lambda r: r["dir"]):
        cfg = run["config"]
        sm  = run["summary"]
        rt  = run_tokens.get(run["dir"], {})
        w(f"  Run : {run['dir']}")
        w(f"    policy_mode      : {cfg.get('policy_mode')}")
        w(f"    max_iterations   : {cfg.get('max_iterations')}")
        w(f"    experiment_split : {cfg.get('experiment_split')}")
        w(f"    experiment_mode  : {cfg.get('experiment_mode')}")
        w(f"    total_samples    : {sm.get('total_samples', sm.get('num_samples', 'N/A'))}")
        w(f"    attack_success_rate      : {fmt_pct(sm.get('attack_success_rate'))}")
        w(f"    invalid_attempt_rate     : {fmt_pct(sm.get('invalid_attempt_rate'))}")
        w(f"    avg_iter_to_success      : {fmt_f(sm.get('average_iterations_to_success') or sm.get('avg_iterations_to_success'))}")
        w(f"    avg_llm_judge_confidence : {fmt_f(sm.get('average_llm_confidence') or sm.get('avg_llm_judge_confidence'))}")

        # Token usage for this run
        w(f"    Token usage:")
        if rt:
            for line in _fmt_tokens(rt, indent="      "):
                w(line)
        else:
            w("      (no .eval data matched)")

        # UCB1 arm stats
        arm_stats = sm.get("arm_reward_stats") or sm.get("arm_stats") or {}
        if not arm_stats:
            # try alternate key names used in run_summary
            arm_stats = {
                k: {"pull_count": sm.get("pulls_by_arm", {}).get(k, 0),
                    "mean_reward": sm.get("average_reward_by_arm", {}).get(k)}
                for k in sm.get("success_by_arm", {})
            }
        if arm_stats:
            w(f"    UCB1 arm rewards:")
            for arm_id, s in sorted(arm_stats.items(), key=lambda x: -(x[1].get("mean_reward") or 0)):
                pulls  = s.get("pull_count", s.get("pulls", "?"))
                mean_r = s.get("mean_reward", s.get("avg_reward", "?"))
                w(f"      {arm_id:<40} pulls={pulls:<5} mean_reward={mean_r}")

        # Tactic breakdown
        if run["attempts"]:
            tb = tactic_breakdown(run["attempts"])
            if tb:
                w(f"    Tactic breakdown (attempts -> successes):")
                for tac, s in sorted(tb.items(), key=lambda x: -x[1]["successes"]):
                    sr = s["successes"] / s["attempts"] * 100 if s["attempts"] else 0
                    w(f"      {tac:<40} {s['successes']}/{s['attempts']} ({sr:.0f}%)")
        w("")

    # -----------------------------------------------------------------------
    # SECTION 5 — Iteration ablation table
    # -----------------------------------------------------------------------
    iter_budgets = sorted(set(int(k[1]) for k in groups if str(k[1]).isdigit()))
    if len(iter_budgets) > 1:
        w("ITERATION ABLATION — ASR on test split")
        w("-" * 72)
        w(f"  {'Policy':<22} " + "  ".join(f"iter={i:<5}" for i in iter_budgets))
        w("  " + "-" * 60)
        for policy in ["random_choice", "agent_based_decision", "rl_bandit"]:
            row = f"  {policy:<22}"
            for itr in iter_budgets:
                for split in ("test", "full"):
                    key = (policy, itr, split)
                    if key in groups:
                        asrs = [r["summary"].get("attack_success_rate") for r in groups[key] if r["summary"].get("attack_success_rate") is not None]
                        row += f"  {sum(asrs)/len(asrs)*100:.1f}%    " if asrs else "  N/A      "
                        break
                else:
                    row += "  -          "
            w(row)
        w("")

        # Token cost of iteration ablation
        w("ITERATION ABLATION — total tokens consumed per budget")
        w("-" * 72)
        for itr in iter_budgets:
            budget_tokens: dict[str, dict] = {}
            for key in groups:
                if key[1] == itr:
                    for run in groups[key]:
                        budget_tokens = _merge_tokens(budget_tokens, run_tokens.get(run["dir"], {}))
            grand_t = sum(v["total"] for v in budget_tokens.values())
            w(f"  max_iterations={itr}  →  grand total tokens: {grand_t:,}")
            for line in _fmt_tokens(budget_tokens, indent="    "):
                w(line)
            w("")

    # -----------------------------------------------------------------------
    # SECTION 6 — Observations
    # -----------------------------------------------------------------------
    w("OBSERVATIONS")
    w("-" * 40)
    test_runs = [r for r in runs if r["config"].get("experiment_split") in ("test", "full")]
    if test_runs:
        best = max(test_runs, key=lambda r: r["summary"].get("attack_success_rate") or 0)
        w(f"  Best ASR overall : {fmt_pct(best['summary'].get('attack_success_rate'))}  "
          f"({best['config'].get('policy_mode')} / max_iter={best['config'].get('max_iterations')})")
        rand_runs = [r for r in test_runs if r["config"].get("policy_mode") == "random_choice"]
        if rand_runs:
            rand_asr = sum(r["summary"].get("attack_success_rate", 0) for r in rand_runs) / len(rand_runs)
            w(f"  Random baseline  : {fmt_pct(rand_asr)}")
        rl_runs = [r for r in test_runs if r["config"].get("policy_mode") == "rl_bandit"]
        if rl_runs and rand_runs:
            rl_asr = sum(r["summary"].get("attack_success_rate", 0) for r in rl_runs) / len(rl_runs)
            lift   = (rl_asr - rand_asr) / rand_asr * 100 if rand_asr > 0 else 0
            w(f"  UCB1 vs random   : {fmt_pct(rl_asr)}  ({lift:+.1f}% relative lift over random)")

    # Token cost summary
    if grand_all_tokens:
        grand_total = sum(v["total"] for v in grand_all_tokens.values())
        w(f"  Total tokens used: {grand_total:,} across {len(runs)} runs")
    w("")
    w("  Plots are in plots/  — check learning curves, tactic preference,")
    w("  and ASR-vs-iteration figures.")
    w("")
    w("NEXT STEPS")
    w("-" * 40)
    w("  1. Review the iteration ablation table — pick the best budget for final results.")
    w("  2. Check UCB1 arm rewards — which tactics converge as best?")
    w("  3. Install tulu3:8b and olmo2:7b, then re-run with SELECTOR_MODEL= for ablation.")
    w("  4. Update Pedro_MSc/main.tex Chapter 7 with the numbers from Section 2 & 5.")
    w("  5. Compare adversarial_code_buggy results with cubert_wbo Resume_cubert.txt.")
    w("")

    # -----------------------------------------------------------------------
    # SECTION 8 — Claude analysis
    # -----------------------------------------------------------------------
    w("=" * 72)
    w("CLAUDE ANALYSIS  (auto-generated — written before you returned)")
    w("=" * 72)
    w("")

    # Collect data for analysis
    all_asrs = {
        pm: [r["summary"].get("attack_success_rate", 0) for r in runs
             if r["config"].get("policy_mode") == pm
             and r["config"].get("experiment_split") in ("test", "full")]
        for pm in ["random_choice", "agent_based_decision", "rl_bandit"]
    }
    rand_asr_val  = sum(all_asrs["random_choice"]) / len(all_asrs["random_choice"]) if all_asrs["random_choice"] else None
    react_asr_val = sum(all_asrs["agent_based_decision"]) / len(all_asrs["agent_based_decision"]) if all_asrs["agent_based_decision"] else None
    rl_asr_val    = sum(all_asrs["rl_bandit"]) / len(all_asrs["rl_bandit"]) if all_asrs["rl_bandit"] else None

    # Dominant tactic across rl runs
    tactic_wins: dict[str, int] = defaultdict(int)
    for run in runs:
        if run["config"].get("policy_mode") != "rl_bandit":
            continue
        arm_stats = run["summary"].get("arm_reward_stats") or {}
        if arm_stats:
            best_arm = max(arm_stats, key=lambda a: arm_stats[a].get("mean_reward") or -99)
            tactic_wins[best_arm] += 1
    best_tactic = max(tactic_wins, key=tactic_wins.get) if tactic_wins else None

    w("1. WHAT THESE RESULTS MEAN FOR THE THESIS")
    w("   ----------------------------------------")
    if rand_asr_val is not None:
        w(f"   Random baseline ASR: {rand_asr_val*100:.1f}%.")
        if rand_asr_val < 0.05:
            w("   This is near the 2% target — qwen2.5-coder:7b is a genuinely hard judge.")
            w("   Any policy that beats random here has non-trivial lifting power. Report this")
            w("   explicitly in the thesis as evidence the task is difficult without tactics.")
        elif rand_asr_val < 0.20:
            w("   Moderate baseline. Some tactics are effective even at random selection.")
            w("   The RL lift may look smaller relative to this baseline; frame it as")
            w("   'sample efficiency of RL over random search' in the thesis.")
        else:
            w("   High random baseline. The judge may be weaker than expected. Check whether")
            w("   the 2% no-tactic baseline was measured without any tactic wrapping, and")
            w("   compare that figure to random_choice here — they should differ significantly.")

    if react_asr_val is not None and rand_asr_val is not None:
        react_lift = (react_asr_val - rand_asr_val) / rand_asr_val * 100 if rand_asr_val > 0 else 0
        w(f"   ReAct vs random: {react_asr_val*100:.1f}% vs {rand_asr_val*100:.1f}%  ({react_lift:+.1f}% relative lift).")
        if react_lift > 10:
            w("   ReAct adds meaningful value — LLM-based tactic selection outperforms blind")
            w("   random search. In the thesis, this is the ablation that isolates 'does")
            w("   tactic intelligence matter at all before training?'")
        elif react_lift > 0:
            w("   ReAct adds a small but positive lift. This suggests the selector model's")
            w("   reasoning is on the right track but the single-shot prompt may be the")
            w("   bottleneck. Mention CoT prompt quality as a future work direction.")
        else:
            w("   ReAct does not outperform random. Two possible causes: (a) the selector")
            w("   model is too small (llama3.1:8b) to reason well about adversarial tactics;")
            w("   (b) the ReAct prompt is too generic. The tulu3/olmo2 ablation will clarify (a).")

    if rl_asr_val is not None and rand_asr_val is not None:
        rl_lift = (rl_asr_val - rand_asr_val) / rand_asr_val * 100 if rand_asr_val > 0 else 0
        w(f"   UCB1 vs random: {rl_asr_val*100:.1f}% vs {rand_asr_val*100:.1f}%  ({rl_lift:+.1f}% relative lift).")
        if rl_lift > 15:
            w("   UCB1 shows clear learning. This is the core positive result of the thesis.")
            w("   Report test-split ASR ± std across repeats as the headline number.")
        elif rl_lift > 5:
            w("   UCB1 shows a moderate lift. Modest but consistent improvement is still")
            w("   publishable — emphasise the low sample cost of bandit learning vs. full RL.")
        else:
            w("   UCB1 lift is weak. Possible explanations: (a) too few training samples,")
            w("   (b) UCB1's exploration bonus keeps re-exploring low-reward arms, (c) the")
            w("   reward signal is too sparse (attack success is binary and rare). Mention")
            w("   sparse-reward bandit as a known challenge and point to GEPA-style")
            w("   dense reward shaping as future work.")
    w("")

    w("2. ITERATION BUDGET ANALYSIS")
    w("   --------------------------")
    if len(iter_budgets) > 1:
        asr_by_iter = {}
        for itr in iter_budgets:
            for split in ("test", "full"):
                for policy in ["rl_bandit", "random_choice", "agent_based_decision"]:
                    key = (policy, itr, split)
                    if key in groups:
                        asrs = [r["summary"].get("attack_success_rate", 0) for r in groups[key]]
                        if asrs:
                            asr_by_iter.setdefault(itr, {})[policy] = sum(asrs) / len(asrs)
        if len(asr_by_iter) >= 2:
            iters_sorted = sorted(asr_by_iter.keys())
            min_i, max_i = iters_sorted[0], iters_sorted[-1]
            for policy in ["rl_bandit", "random_choice"]:
                lo = asr_by_iter.get(min_i, {}).get(policy)
                hi = asr_by_iter.get(max_i, {}).get(policy)
                if lo is not None and hi is not None:
                    gain = (hi - lo) / lo * 100 if lo > 0 else 0
                    w(f"   {policy}: iter={min_i} → iter={max_i}  ASR change: {gain:+.1f}%")
            w("   If the gain from iter=3 to iter=12 is <5% relative, recommend iter=3 as the")
            w("   sweet spot for the thesis — it gives most of the benefit at ~4x lower cost.")
            w("   This framing (cost-adjusted ASR) is a clean RQ3 answer.")
    else:
        w("   Only one iteration budget was run — cannot compute ablation yet.")
    w("")

    w("3. UCB1 TACTIC PREFERENCES")
    w("   ------------------------")
    if best_tactic:
        w(f"   Dominant tactic (highest mean reward across RL runs): {best_tactic}")
        w("   This is the tactic UCB1 converged on. For the thesis, use this as evidence")
        w("   that RL found a non-obvious preference — random would have split evenly.")
        w("   The taxonomy category of this tactic (check tactic_registry.py) tells you")
        w("   which attack family works best against qwen2.5-coder:7b.")
    else:
        w("   No UCB1 arm data found — cannot identify dominant tactic.")
    w("")

    w("4. THINGS TO DOUBLE-CHECK WHEN YOU RETURN")
    w("   ----------------------------------------")
    w("   a) invalid_attempt_rate in every run_summary.json should be 0.0.")
    w("      Non-zero means the tactic generator produced syntactically broken code.")
    w("      Check results/<run_id>/attempts.jsonl for syntax_valid=false records.")
    w("")
    w("   b) llm_judge_confidence should never be exactly 0.0 or 1.0 for all samples.")
    w("      If it is, the judge is outputting a degenerate score — check llm_judge.py.")
    w("")
    w("   c) UCB1 arm entropy should decrease across training epochs.")
    w("      If entropy stays flat (all arms pulled equally), UCB1 is not converging.")
    w("      This would mean the reward signal is too noisy — try fewer epochs.")
    w("")
    w("   d) Token usage: check if judge tokens >> selector tokens.")
    w("      If not, the selector is being called as often as the judge, which is")
    w("      expected in agent_based_decision but not in random_choice.")
    w("      Unexpected selector calls in random_choice runs = a bug.")
    w("")
    w("   e) cubert_wbo ASR vs adversarial_code_buggy ASR:")
    w("      cubert_wbo has known pre-existing bugs (no generation step), so its")
    w("      baseline should be higher. If cubert_wbo ASR < adversarial_code_buggy ASR,")
    w("      investigate — either the bugs are too subtle or the judge is inconsistent.")
    w("")

    w("5. THESIS WRITING NOTE")
    w("   --------------------")
    w("   The key claim of the thesis is: 'UCB1 bandit selection of adversarial tactics")
    w("   improves attack success rate over random tactic selection on code LLM judges.'")
    w("   Every table in the results chapter should be anchored to this claim.")
    w("   The iteration ablation answers: 'how many attempts does the attacker need?'")
    w("   The tactic preference answers: 'which tactic category is most effective?'")
    w("   The dataset comparison answers: 'does the finding generalise across benchmarks?'")
    w("   If you have numbers that support all three, Chapter 7 writes itself.")
    w("")
    w("   — Claude Sonnet 4.6  (wrote this while you were away)")
    w("")

    w("=" * 72)
    w(f"End of resume — {benchmark}")
    w("=" * 72)

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Resume written -> {output_path}  ({len(lines)} lines)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write experiment resume with token metadata.")
    parser.add_argument("--benchmark",   required=True)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--logs-dir",    default="logs")
    parser.add_argument("--output",      required=True)
    args = parser.parse_args()

    write_resume(
        benchmark=args.benchmark,
        results_dir=Path(args.results_dir),
        logs_dir=Path(args.logs_dir),
        output_path=Path(args.output),
    )
