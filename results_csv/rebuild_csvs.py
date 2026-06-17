"""Re-builds both result CSVs from stored_results/. Run after new experiments complete."""
import os, json, csv
from collections import defaultdict

BASE = os.path.join(os.path.dirname(__file__), "..", "stored_results")
OUT  = os.path.dirname(__file__)


def load_by_sample(run_dir):
    results_path = os.path.join(run_dir, "results")
    if not os.path.isdir(results_path):
        return {}
    by_sample = defaultdict(list)
    for sub in os.listdir(results_path):
        jl = os.path.join(results_path, sub, "attempts.jsonl")
        if os.path.exists(jl):
            with open(jl) as f:
                for line in f:
                    try:
                        r = json.loads(line.strip())
                        by_sample[r.get("sample_id", sub)].append(r)
                    except Exception:
                        pass
    return dict(by_sample)


def asr_at(by_sample, max_iter):
    if not by_sample:
        return None
    wins = sum(
        1 for atts in by_sample.values()
        if any(a.get("attack_success") and a.get("iteration", 99) <= max_iter for a in atts)
    )
    return round(100 * wins / len(by_sample), 1)


# ── CSV 1: asr_by_policy_judge_corpus.csv ─────────────────────────────────────
POLICY_RUNS = [
    ("section-1", "qwen",      "qwen2.5-coder:7b",    "random",   "random-llama31-qwen-{c}"),
    ("section-1", "qwen",      "qwen2.5-coder:7b",    "react",    "react-llama31-qwen-{c}"),
    ("section-1", "qwen",      "qwen2.5-coder:7b",    "ucb1",     "ucb1-eval-llama31-qwen-{c}"),
    ("section-1", "qwen",      "qwen2.5-coder:7b",    "thompson", "thompson-eval-llama31-qwen-{c}"),
    ("section-1", "qwen",      "qwen2.5-coder:7b",    "kl-ucb",   "klucb-eval-llama31-qwen-{c}"),
    ("section-1", "qwen",      "qwen2.5-coder:7b",    "exp3",     "exp3-eval-llama31-qwen-{c}"),
    ("section-2", "deepseek",  "deepseek-coder:6.7b", "random",   "random-llama31-deepseek-{c}"),
    ("section-2", "deepseek",  "deepseek-coder:6.7b", "react",    "react-llama31-deepseek-{c}"),
    ("section-2", "deepseek",  "deepseek-coder:6.7b", "ucb1",     "ucb1-eval-llama31-deepseek-{c}"),
    ("section-2", "deepseek",  "deepseek-coder:6.7b", "thompson", "thompson-eval-llama31-deepseek-{c}"),
    ("section-2", "deepseek",  "deepseek-coder:6.7b", "kl-ucb",   "klucb-eval-llama31-deepseek-{c}"),
    ("section-2", "deepseek",  "deepseek-coder:6.7b", "exp3",     "exp3-eval-llama31-deepseek-{c}"),
    ("section-3", "codellama", "codellama:7b",        "random",   "random-llama31-codellama-{c}"),
    ("section-3", "codellama", "codellama:7b",        "react",    "react-llama31-codellama-{c}"),
    ("section-3", "codellama", "codellama:7b",        "ucb1",     "ucb1-eval-llama31-codellama-{c}"),
    ("section-3", "codellama", "codellama:7b",        "thompson", "thompson-eval-llama31-codellama-{c}"),
    ("section-3", "codellama", "codellama:7b",        "kl-ucb",   "klucb-eval-llama31-codellama-{c}"),
    ("section-3", "codellama", "codellama:7b",        "exp3",     "exp3-eval-llama31-codellama-{c}"),
    ("section-4", "starcoder", "starcoder2:7b",       "random",   "random-llama31-starcoder-{c}"),
    ("section-4", "starcoder", "starcoder2:7b",       "react",    "react-llama31-starcoder-{c}"),
]

CORPORA = [
    ("adversarial_code_buggy", "adversarial-code-buggy"),
    ("cubert_wbo",             "cubert_wbo"),
]

rows1 = []
for section, judge_tag, judge_label, policy, pattern in POLICY_RUNS:
    for corpus_dir, corpus_label in CORPORA:
        run_dir = os.path.join(BASE, section, pattern.format(c=corpus_dir))
        bs = load_by_sample(run_dir)
        rows1.append({
            "judge":        judge_label,
            "judge_tag":    judge_tag,
            "policy":       policy,
            "corpus":       corpus_label,
            "n":            len(bs) if bs else None,
            "baseline_asr": asr_at(bs, 0),
            "asr_1shot":    asr_at(bs, 1),
            "asr_3shot":    asr_at(bs, 3),
            "asr_6shot":    asr_at(bs, 6),
        })

out1 = os.path.join(OUT, "asr_by_policy_judge_corpus.csv")
with open(out1, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["judge","judge_tag","policy","corpus","n",
                                       "baseline_asr","asr_1shot","asr_3shot","asr_6shot"])
    w.writeheader()
    w.writerows(rows1)
print(f"Written: {out1}  ({len(rows1)} rows)")


# ── CSV 2: rq2_tactic_effectiveness.csv ───────────────────────────────────────
TACTICS = [
    "legacy_injection", "legacy_output", "legacy_semantic", "legacy_cot",
    "taxonomy_roleplay", "taxonomy_appeal_to_authority",
    "taxonomy_formatting_smuggling", "taxonomy_recursion_crescendo", "taxonomy_crowding",
]
JUDGES = [
    ("qwen",      "qwen2.5-coder:7b"),
    ("deepseek",  "deepseek-coder:6.7b"),
    ("codellama", "codellama:7b"),
    ("starcoder", "starcoder2:7b"),
]

rows2 = []
rq2_base = os.path.join(BASE, "rq2")
for tactic in TACTICS:
    for judge_tag, judge_label in JUDGES:
        for corpus_dir, corpus_label in CORPORA:
            run_dir = os.path.join(rq2_base, f"{tactic}-llama31-{judge_tag}-{corpus_dir}")
            bs = load_by_sample(run_dir)
            if not bs:
                rows2.append({"tactic": tactic, "judge": judge_label, "judge_tag": judge_tag,
                               "corpus": corpus_label, "n": None,
                               "baseline_asr": None, "tactic_asr": None, "combined_asr": None})
                continue
            total = len(bs)
            b_wins = t_wins = c_wins = 0
            for atts in bs.values():
                i0 = any(a.get("attack_success") and a.get("iteration") == 0 for a in atts)
                i1 = any(a.get("attack_success") and a.get("iteration") == 1 for a in atts)
                if i0: b_wins += 1
                if i1: t_wins += 1
                if i0 or i1: c_wins += 1
            rows2.append({
                "tactic":       tactic,
                "judge":        judge_label,
                "judge_tag":    judge_tag,
                "corpus":       corpus_label,
                "n":            total,
                "baseline_asr": round(100 * b_wins / total, 1),
                "tactic_asr":   round(100 * t_wins / total, 1),
                "combined_asr": round(100 * c_wins / total, 1),
            })

out2 = os.path.join(OUT, "rq2_tactic_effectiveness.csv")
with open(out2, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["tactic","judge","judge_tag","corpus","n",
                                       "baseline_asr","tactic_asr","combined_asr"])
    w.writeheader()
    w.writerows(rows2)

missing = [r for r in rows2 if r["n"] is None]
print(f"Written: {out2}  ({len(rows2)} rows, {len(missing)} still NULL)")
if missing:
    print("NULL rows (runs not yet complete):")
    for r in missing:
        print(f"  {r['tactic']} / {r['judge_tag']} / {r['corpus']}")
