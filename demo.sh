#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# demo.sh  ─  Adversarial Attacks on Code LLMs  ─  Professor Evaluation Demo
# ─────────────────────────────────────────────────────────────────────────────
#
# Runs four self-contained experiment suites, then aggregates and plots
# everything. The professor only needs to press play.
#
#   [1] Tactic Isolation   ─ each of the 9 tactics applied individually
#   [2] Policy Comparison  ─ random-choice vs. CoT-agent vs. RL bandit
#   [3] Iteration Depth    ─ 1 vs. 3 vs. 5 adaptive iterations
#   [4] CoT Ablation       ─ LLM selector with vs. without chain-of-thought
#
# ─── Usage ───────────────────────────────────────────────────────────────────
#
#   bash demo.sh                                    # defaults
#   bash demo.sh --model ollama/qwen2.5-coder:32b   # different model
#   bash demo.sh --samples 30                       # more samples
#   bash demo.sh --quick                            # 3 samples, pipeline check
#
# ─── Options ─────────────────────────────────────────────────────────────────
#
#   --model  MODEL    Code-generation target model  (default: ollama/llama3.1:70b)
#   --judge  MODEL    Judge / selector model        (default: same as --model)
#   --benchmark BENCH mbpp | humaneval              (default: mbpp)
#   --samples N       Samples per experiment        (default: 20)
#   --iterations N    Max iterations for iterative  (default: 5)
#   --weights PATH    Pre-trained UCB1 weights      (default: weights/mbpp_ucb1.json)
#   --output DIR      Output root directory         (default: demo_results)
#   --quick           3 samples only — fast smoke test
#   --no-prereq       Skip prerequisite checks
#   -h / --help       Show this message
#
# ─── Output layout ───────────────────────────────────────────────────────────
#
#   demo_results/<timestamp>/
#     SUMMARY.md                    ← key numbers, ready for slides / paper
#     run.log                       ← full Inspect output
#     results/
#       1_tactic_isolation/         ← 9 runs (one per tactic)
#       2_policy_comparison/        ← 3 runs (random / agent / rl_bandit)
#       3_iteration_depth/          ← 3 runs (1, 3, 5 iterations)
#       4_cot_ablation/             ← 2 runs (CoT on / off)
#       combined/                   ← flat copy of all 17 runs for cross-suite plots
#     aggregates/
#       1_tactic_isolation/         ← grouped_summary.json, aggregated_runs.csv, …
#       2_policy_comparison/
#       3_iteration_depth/
#       4_cot_ablation/
#       combined/
#     plots/
#       1_tactic_isolation/         ← tactic_win_rate, success_rate, …
#       2_policy_comparison/        ← policy bar charts, success_breakdown, …
#       3_iteration_depth/          ← one_shot_vs_iterative, iterations_to_success, …
#       4_cot_ablation/             ← cot comparison charts
#       combined/                   ← full cross-suite view (all 17 runs)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Terminal colours ──────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; RESET='\033[0m'
BLUE='\033[0;34m'

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL="ollama/llama3.1:70b"
JUDGE_MODEL=""
BENCHMARK="mbpp"
SAMPLES=20
MAX_ITER=5
WEIGHTS_PATH="weights/mbpp_ucb1.json"
OUTPUT_ROOT="demo_results"
QUICK=false
SKIP_PREREQ=false

# ── Argument parsing ──────────────────────────────────────────────────────────
usage() {
    sed -n '/^# ─── Usage/,/^# ─── Output/{ /^# ─── Output/d; s/^# \{0,3\}//; p }' "$0"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL="$2";        shift 2 ;;
        --judge)      JUDGE_MODEL="$2";  shift 2 ;;
        --benchmark)  BENCHMARK="$2";   shift 2 ;;
        --samples)    SAMPLES="$2";     shift 2 ;;
        --iterations) MAX_ITER="$2";    shift 2 ;;
        --weights)    WEIGHTS_PATH="$2"; shift 2 ;;
        --output)     OUTPUT_ROOT="$2"; shift 2 ;;
        --quick)      QUICK=true;       shift   ;;
        --no-prereq)  SKIP_PREREQ=true; shift   ;;
        -h|--help)    usage; exit 0     ;;
        *) echo -e "${RED}Unknown option: $1${RESET}" >&2; usage >&2; exit 1 ;;
    esac
done

[[ "$QUICK" == true ]] && SAMPLES=3

JUDGE_MODEL="${JUDGE_MODEL:-$MODEL}"
SELECTOR_MODEL="$JUDGE_MODEL"
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
SESSION_DIR="${OUTPUT_ROOT}/${TIMESTAMP}"
RESULTS_DIR="${SESSION_DIR}/results"
PLOTS_DIR="${SESSION_DIR}/plots"
AGGREGATES_DIR="${SESSION_DIR}/aggregates"
LOG_FILE="${SESSION_DIR}/run.log"

# ── Locate project root (this script lives there) ─────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Find the virtual-environment binaries ─────────────────────────────────────
_find_bin() {
    local name="$1"
    for candidate in \
        ".venv/Scripts/${name}.exe" \
        ".venv/Scripts/${name}" \
        ".venv/bin/${name}" \
        "../.venv/Scripts/${name}.exe" \
        "../.venv/Scripts/${name}" \
        "../.venv/bin/${name}"; do
        [[ -f "$candidate" ]] && { echo "$candidate"; return; }
    done
    echo "$name"   # fall back to PATH
}

INSPECT_BIN="$(_find_bin inspect)"
PYTHON_BIN="$(_find_bin python)"

# ── Create output skeleton ────────────────────────────────────────────────────
mkdir -p \
    "${RESULTS_DIR}/1_tactic_isolation" \
    "${RESULTS_DIR}/2_policy_comparison" \
    "${RESULTS_DIR}/3_iteration_depth" \
    "${RESULTS_DIR}/4_cot_ablation" \
    "${RESULTS_DIR}/combined" \
    "${PLOTS_DIR}/1_tactic_isolation" \
    "${PLOTS_DIR}/2_policy_comparison" \
    "${PLOTS_DIR}/3_iteration_depth" \
    "${PLOTS_DIR}/4_cot_ablation" \
    "${PLOTS_DIR}/combined" \
    "${AGGREGATES_DIR}/1_tactic_isolation" \
    "${AGGREGATES_DIR}/2_policy_comparison" \
    "${AGGREGATES_DIR}/3_iteration_depth" \
    "${AGGREGATES_DIR}/4_cot_ablation" \
    "${AGGREGATES_DIR}/combined"

touch "$LOG_FILE"

# ── Logging helpers ───────────────────────────────────────────────────────────
_ts() { date '+%H:%M:%S'; }
log()  { printf '[%s] %s\n' "$(_ts)" "$*" >> "$LOG_FILE"; }
ok()   { echo -e "  ${GREEN}✓${RESET} $*"; log "OK: $*"; }
warn() { echo -e "  ${YELLOW}⚠${RESET} $*"; log "WARN: $*"; }
fail() { echo -e "  ${RED}✗${RESET} $*"; log "FAIL: $*"; }
step() { echo -e "  ${BLUE}→${RESET} $*"; log "STEP: $*"; }

section() {
    local title="$1" subtitle="${2:-}"
    local line='────────────────────────────────────────────────────────────'
    echo ""
    echo -e "${BOLD}${CYAN}${line}${RESET}"
    echo -e "${BOLD}${CYAN}  ${title}${RESET}"
    [[ -n "$subtitle" ]] && echo -e "${DIM}  ${subtitle}${RESET}"
    echo -e "${BOLD}${CYAN}${line}${RESET}"
    log "=== ${title} ==="
}

# ── Absolute-path helper (Inspect changes cwd internally) ─────────────────────
abspath() {
    "$PYTHON_BIN" -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "$1"
}

# ── Run one Inspect experiment ────────────────────────────────────────────────
# Usage: run_exp <suite_subdir> <human label> [extra -T flags ...]
run_exp() {
    local SUITE_SUBDIR="$1"; shift
    local LABEL="$1";         shift
    local ABS_DEST
    ABS_DEST="$(abspath "${RESULTS_DIR}/${SUITE_SUBDIR}")"

    step "$LABEL"
    log "  results_dir → $ABS_DEST"

    if PYTHONPATH=V3 "$INSPECT_BIN" eval JESTER/adversarial_attack.py@adversarial_code_llm \
        --model "$MODEL" \
        -T benchmark="$BENCHMARK" \
        -T mutation_strategy=react \
        -T use_llm_judge=True \
        -T judge_model="$JUDGE_MODEL" \
        -T selector_model="$SELECTOR_MODEL" \
        -T results_dir="$ABS_DEST" \
        --max-samples "$SAMPLES" \
        "$@" >> "$LOG_FILE" 2>&1; then
        : # success — fall through to stats
    else
        warn "Run returned non-zero exit (see log). Partial results may still be available."
    fi

    # Print a quick success count from the written summaries
    local STATS
    STATS="$("$PYTHON_BIN" - "$ABS_DEST" <<'PYEOF'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
runs = [p for p in d.iterdir() if p.is_dir() and (p/"run_summary.json").exists()]
succ = sum(
    1 for p in runs
    if json.loads((p/"run_summary.json").read_text()).get("attack_success_rate", 0) == 1.0
)
print(f"{succ}/{len(runs)} succeeded")
PYEOF
    2>/dev/null || echo "?")"
    echo -e "    ${DIM}${STATS}${RESET}"
}

# ─────────────────────────────────────────────────────────────────────────────
# PREREQUISITES
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$SKIP_PREREQ" == false ]]; then
    section "Prerequisites"

    # Python environment
    if "$PYTHON_BIN" -c "import inspect_ai" &>/dev/null; then
        ok "Python environment  ($PYTHON_BIN)"
    else
        fail "inspect_ai not found in $PYTHON_BIN"
        echo "  → Run:  pip install -r requirements.txt"
        exit 1
    fi

    # Docker
    if docker info &>/dev/null 2>&1; then
        ok "Docker running"
    else
        fail "Docker not running — deterministic test sandbox unavailable"
        echo "  → Start Docker Desktop and retry."
        exit 1
    fi

    # Ollama — only checked when the model string starts with "ollama/"
    if [[ "$MODEL" == ollama/* ]]; then
        if curl -sf http://localhost:11434/api/tags &>/dev/null; then
            ok "Ollama running (localhost:11434)"
            # Check whether the specific model is already pulled
            MODEL_TAG="${MODEL#ollama/}"
            if curl -sf http://localhost:11434/api/tags \
                | "$PYTHON_BIN" -c "
import sys, json
d = json.load(sys.stdin)
tag = sys.argv[1].split(':')[0]
sys.exit(0 if any(m['name'].startswith(tag) for m in d.get('models', [])) else 1)
" "$MODEL_TAG" &>/dev/null; then
                ok "Model pulled: $MODEL_TAG"
            else
                warn "Model '$MODEL_TAG' not pulled yet."
                if [[ -t 0 ]]; then
                    read -rp "  Pull it now? [y/N] " _ans
                    if [[ "$_ans" =~ ^[Yy]$ ]]; then
                        ollama pull "$MODEL_TAG"
                    else
                        warn "Skipping pull — run will fail if model is missing."
                    fi
                else
                    warn "Non-interactive mode — run 'ollama pull $MODEL_TAG' manually."
                fi
            fi
        else
            fail "Ollama not reachable on localhost:11434"
            echo "  → Start Ollama:  ollama serve"
            exit 1
        fi
    else
        # Non-Ollama provider — check that the relevant API key env var is set
        _PROVIDER="${MODEL%%/*}"
        case "$_PROVIDER" in
            openai)    _KEY_VAR="OPENAI_API_KEY" ;;
            anthropic) _KEY_VAR="ANTHROPIC_API_KEY" ;;
            google)    _KEY_VAR="GOOGLE_API_KEY" ;;
            mistral)   _KEY_VAR="MISTRAL_API_KEY" ;;
            groq)      _KEY_VAR="GROQ_API_KEY" ;;
            together)  _KEY_VAR="TOGETHER_API_KEY" ;;
            azure)     _KEY_VAR="AZURE_OPENAI_API_KEY" ;;
            *)         _KEY_VAR="" ;;
        esac
        if [[ -n "$_KEY_VAR" ]]; then
            if [[ -n "${!_KEY_VAR:-}" ]]; then
                ok "API key set: \$$_KEY_VAR"
            else
                warn "\$$_KEY_VAR is not set — model calls will fail."
                echo "  → Export it:  export $_KEY_VAR=<your-key>"
                echo "  → Or add it to a .env file in the project root."
            fi
        else
            ok "Provider: $_PROVIDER (no standard key check — verify manually)"
        fi
    fi

    # Pre-trained RL weights
    HAVE_WEIGHTS=false
    if [[ -f "$WEIGHTS_PATH" ]]; then
        ok "RL weights: $WEIGHTS_PATH"
        HAVE_WEIGHTS=true
    else
        warn "RL weights not found at '$WEIGHTS_PATH'."
        warn "Suite [2] rl_bandit will run with uninitialised (cold-start) weights."
        warn "For best results, copy your trained weights to '$WEIGHTS_PATH'."
    fi
fi

ABS_WEIGHTS="$(abspath "$WEIGHTS_PATH" 2>/dev/null || echo "$WEIGHTS_PATH")"

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
TACTIC_COUNT=9
POLICY_COUNT=3
ITER_LEVELS=3    # 1, 3, MAX_ITER
COT_COUNT=2
TOTAL_RUNS=$(( TACTIC_COUNT + POLICY_COUNT + ITER_LEVELS + COT_COUNT ))
TOTAL_SAMPLES=$(( TOTAL_RUNS * SAMPLES ))

section "Demo Configuration"
echo -e "  Model            : ${BOLD}$MODEL${RESET}"
echo -e "  Judge / Selector : $JUDGE_MODEL"
echo -e "  Benchmark        : $BENCHMARK"
echo -e "  Samples / run    : $SAMPLES"
echo -e "  Max iterations   : $MAX_ITER  (iterative experiments)"
echo -e "  RL weights       : $WEIGHTS_PATH"
echo ""
echo -e "  ${BOLD}Experiment plan${RESET}"
printf "  %-35s %3d runs × %2d samples = %4d samples\n" \
    "[1] Tactic Isolation" "$TACTIC_COUNT" "$SAMPLES" $(( TACTIC_COUNT * SAMPLES ))
printf "  %-35s %3d runs × %2d samples = %4d samples\n" \
    "[2] Policy Comparison" "$POLICY_COUNT" "$SAMPLES" $(( POLICY_COUNT * SAMPLES ))
printf "  %-35s %3d runs × %2d samples = %4d samples\n" \
    "[3] Iteration Depth (1/3/$MAX_ITER)" "$ITER_LEVELS" "$SAMPLES" $(( ITER_LEVELS * SAMPLES ))
printf "  %-35s %3d runs × %2d samples = %4d samples\n" \
    "[4] CoT Ablation" "$COT_COUNT" "$SAMPLES" $(( COT_COUNT * SAMPLES ))
echo "  ──────────────────────────────────────────────────────"
printf "  %-35s %3d runs × %2d samples = %4d samples total\n" \
    "Total" "$TOTAL_RUNS" "$SAMPLES" "$TOTAL_SAMPLES"

echo ""
echo -e "  Output: ${BOLD}${SESSION_DIR}/${RESET}"
echo -e "  Log:    $LOG_FILE"
[[ "$QUICK" == true ]] && echo -e "  ${YELLOW}QUICK MODE — $SAMPLES samples per run${RESET}"

# Estimated runtime: ~90 s/sample for 70b-class models, less for smaller models
_EST_SECS=$(( TOTAL_SAMPLES * 90 ))
_EST_H=$(( _EST_SECS / 3600 ))
_EST_M=$(( (_EST_SECS % 3600) / 60 ))
echo -e "  ${DIM}Estimated runtime: ~${_EST_H}h ${_EST_M}m  (@ 90 s/sample — varies by model and hardware)${RESET}"
echo ""

log "========================================================"
log "Demo configuration"
log "  model:      $MODEL"
log "  judge:      $JUDGE_MODEL"
log "  benchmark:  $BENCHMARK"
log "  samples:    $SAMPLES"
log "  max_iter:   $MAX_ITER"
log "  weights:    $WEIGHTS_PATH"
log "  output:     $SESSION_DIR"
log "========================================================"

# ─────────────────────────────────────────────────────────────────────────────
# SUITE 1 — TACTIC ISOLATION
# ─────────────────────────────────────────────────────────────────────────────
# Each of the 9 tactics is pinned with forced_tactic. One-shot mode strips
# out any policy effects — we measure the raw attack capability of each tactic.
# ─────────────────────────────────────────────────────────────────────────────
section "[1] Tactic Isolation" \
    "One-shot, each tactic pinned individually  (9 runs × $SAMPLES samples)"

TACTICS=(
    "legacy_injection"
    "legacy_output"
    "legacy_semantic"
    "legacy_cot"
    "taxonomy_roleplay"
    "taxonomy_appeal_to_authority"
    "taxonomy_formatting_smuggling"
    "taxonomy_recursion_crescendo"
    "taxonomy_crowding"
)

for TACTIC in "${TACTICS[@]}"; do
    run_exp "1_tactic_isolation" \
        "$TACTIC" \
        -T policy_mode=random_choice \
        -T experiment_mode=one_shot \
        -T max_iterations=1 \
        -T forced_tactic="$TACTIC" \
        -T experiment_split=full
done

# ─────────────────────────────────────────────────────────────────────────────
# SUITE 2 — POLICY COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
# All three selector policies run on the same first-N samples of the benchmark.
# This is the core thesis comparison: does learned tactic selection (RL bandit)
# outperform random selection and LLM chain-of-thought reasoning?
# ─────────────────────────────────────────────────────────────────────────────
section "[2] Policy Comparison" \
    "Iterative ($MAX_ITER iterations), same samples  (3 runs × $SAMPLES samples)"

run_exp "2_policy_comparison" \
    "random_choice  (random tactic selection)" \
    -T policy_mode=random_choice \
    -T experiment_mode=iterative \
    -T max_iterations="$MAX_ITER" \
    -T experiment_split=full

run_exp "2_policy_comparison" \
    "agent_based_decision  (LLM selects tactic via CoT)" \
    -T policy_mode=agent_based_decision \
    -T selector_use_cot=True \
    -T experiment_mode=iterative \
    -T max_iterations="$MAX_ITER" \
    -T experiment_split=full

run_exp "2_policy_comparison" \
    "rl_bandit  (UCB1, frozen weights)" \
    -T policy_mode=rl_bandit \
    -T bandit_algorithm=ucb1 \
    -T bandit_weights_path="$ABS_WEIGHTS" \
    -T bandit_freeze_weights=True \
    -T experiment_mode=iterative \
    -T max_iterations="$MAX_ITER" \
    -T experiment_split=full

# ─────────────────────────────────────────────────────────────────────────────
# SUITE 3 — ITERATION DEPTH
# ─────────────────────────────────────────────────────────────────────────────
# Identical policy (agent_based_decision + CoT) run with different iteration
# budgets: 1, 3, and MAX_ITER. Shows whether adaptive multi-step attacks
# outperform a single attempt and whether there are diminishing returns.
# ─────────────────────────────────────────────────────────────────────────────
section "[3] Iteration Depth" \
    "agent_based_decision, budget = 1 / 3 / $MAX_ITER  (3 runs × $SAMPLES samples)"

for N_ITER in 1 3 "$MAX_ITER"; do
    MODE="iterative"
    [[ "$N_ITER" -eq 1 ]] && MODE="one_shot"
    run_exp "3_iteration_depth" \
        "max_iterations = $N_ITER  ($MODE)" \
        -T policy_mode=agent_based_decision \
        -T selector_use_cot=True \
        -T experiment_mode="$MODE" \
        -T max_iterations="$N_ITER" \
        -T experiment_split=full
done

# ─────────────────────────────────────────────────────────────────────────────
# SUITE 4 — CoT ABLATION
# ─────────────────────────────────────────────────────────────────────────────
# Same policy and iterations, with chain-of-thought reasoning in the selector
# enabled vs. disabled. Isolates the contribution of the LLM's reasoning step
# to tactic selection quality.
# ─────────────────────────────────────────────────────────────────────────────
section "[4] CoT Ablation" \
    "agent_based_decision, CoT on vs. off  (2 runs × $SAMPLES samples)"

run_exp "4_cot_ablation" \
    "CoT ENABLED   (selector reasons before choosing tactic)" \
    -T policy_mode=agent_based_decision \
    -T selector_use_cot=True \
    -T experiment_mode=iterative \
    -T max_iterations="$MAX_ITER" \
    -T experiment_split=full

run_exp "4_cot_ablation" \
    "CoT DISABLED  (selector picks tactic without reasoning)" \
    -T policy_mode=agent_based_decision \
    -T selector_use_cot=False \
    -T experiment_mode=iterative \
    -T max_iterations="$MAX_ITER" \
    -T experiment_split=full

# ─────────────────────────────────────────────────────────────────────────────
# COLLECT ALL RUNS INTO combined/ FOR CROSS-SUITE PLOTS
# ─────────────────────────────────────────────────────────────────────────────
section "Collecting runs for combined view"

"$PYTHON_BIN" - "${RESULTS_DIR}" <<'PYEOF'
import json, shutil, pathlib, sys
results = pathlib.Path(sys.argv[1])
combined = results / "combined"
combined.mkdir(exist_ok=True)
copied = 0
for suite in sorted(results.iterdir()):
    if not suite.is_dir() or suite.name == "combined":
        continue
    for run in sorted(suite.iterdir()):
        if run.is_dir() and (run / "run_config.json").exists():
            dest = combined / run.name
            if not dest.exists():
                shutil.copytree(str(run), str(dest))
                copied += 1
print(f"  Collected {copied} runs into combined/")
PYEOF

# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATE
# ─────────────────────────────────────────────────────────────────────────────
section "Aggregating Results"

for SUITE in 1_tactic_isolation 2_policy_comparison 3_iteration_depth 4_cot_ablation combined; do
    step "$SUITE"
    "$PYTHON_BIN" JESTER/scripts/aggregate_results.py \
        --results-dir  "${RESULTS_DIR}/${SUITE}" \
        --output-dir   "${AGGREGATES_DIR}/${SUITE}" \
        >> "$LOG_FILE" 2>&1 || warn "Aggregation failed for $SUITE (see log)"
done

# ─────────────────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────────────────
section "Generating Plots"

_plot() {
    local SUITE="$1"; shift
    step "$SUITE"
    "$PYTHON_BIN" plot.py \
        --results-dir "${RESULTS_DIR}/${SUITE}" \
        --output-dir  "${PLOTS_DIR}/${SUITE}" \
        "$@" >> "$LOG_FILE" 2>&1 || warn "Plotting failed for $SUITE (see log)"
}

_plot "1_tactic_isolation"
_plot "2_policy_comparison"
_plot "3_iteration_depth"
_plot "4_cot_ablation"
_plot "combined"

# ─────────────────────────────────────────────────────────────────────────────
# GENERATE SUMMARY.md
# ─────────────────────────────────────────────────────────────────────────────
section "Generating Summary Report"

"$PYTHON_BIN" - \
    "${RESULTS_DIR}" \
    "${SESSION_DIR}/SUMMARY.md" \
    "$MODEL" \
    "$BENCHMARK" \
    "$SAMPLES" \
    "$MAX_ITER" \
    "$TIMESTAMP" \
<<'PYEOF'
import json, pathlib, sys
from collections import defaultdict

results_dir  = pathlib.Path(sys.argv[1])
summary_path = pathlib.Path(sys.argv[2])
model        = sys.argv[3]
benchmark    = sys.argv[4]
samples_per  = int(sys.argv[5])
max_iter     = int(sys.argv[6])
timestamp    = sys.argv[7]


def load_runs(subdir):
    """Return one dict per persisted run (one Inspect sample = one run dir)."""
    d = results_dir / subdir
    rows = []
    if not d.exists():
        return rows
    for p in sorted(d.iterdir()):
        s = p / "run_summary.json"
        c = p / "run_config.json"
        if s.exists() and c.exists():
            # run_summary fields override run_config where keys overlap
            rows.append({**json.loads(c.read_text()), **json.loads(s.read_text())})
    return rows


def infer_tactic(row):
    """Best-effort tactic ID from a per-sample run row."""
    pulls = row.get("pulls_by_arm") or {}
    if pulls:
        return max(pulls, key=lambda k: pulls[k])
    return row.get("forced_tactic") or "unknown"


def aggregate(rows, key_fn):
    """Group rows by key_fn(row) and compute aggregate success metrics."""
    groups = defaultdict(list)
    for r in rows:
        groups[key_fn(r)].append(r)
    out = []
    for key, grp in groups.items():
        total    = len(grp)
        succ     = sum(1 for r in grp if (r.get("attack_success_rate") or 0) >= 1.0)
        baseline = sum(1 for r in grp if r.get("baseline_success"))
        tactic   = sum(1 for r in grp if r.get("tactic_driven_success"))
        rate     = succ / total if total else None
        out.append({"key": key, "total": total, "succ": succ,
                    "baseline": baseline, "tactic": tactic, "rate": rate})
    return out


def pct(v):  return f"{v * 100:.1f}%" if v is not None else "N/A"
def bar(v, w=15):
    if v is None: return "─" * w
    return "█" * round(v * w) + "░" * (w - round(v * w))


lines = []
_ = lines.append

_("# Adversarial Attacks on Code LLMs — Demo Results")
_("")
_("| Field | Value |")
_("| ----- | ----- |")
_(f"| Model | `{model}` |")
_(f"| Benchmark | {benchmark.upper()} |")
_(f"| Samples per experiment | {samples_per} |")
_(f"| Max iterations (iterative) | {max_iter} |")
_(f"| Generated | {timestamp.replace('_', ' ')} |")
_("")

# ── [1] Tactic Isolation ──────────────────────────────────────────────────────
_("---")
_("## [1] Tactic Isolation")
_("")
_("Each of the 9 tactics pinned individually (one-shot, no policy influence).  ")
_(f"N = {samples_per} samples per tactic.")
_("")
_("| Tactic | Successes | Rate | Baseline | Tactic-Driven |")
_("|--------|-----------|------|----------|---------------|")

suite1 = load_runs("1_tactic_isolation")
rows1  = aggregate(suite1, infer_tactic)
rows1.sort(key=lambda r: -(r["rate"] or 0))

for r in rows1:
    _(f"| `{r['key']}` | {r['succ']}/{r['total']} | **{pct(r['rate'])}** "
      f"| {r['baseline']} | {r['tactic']} |")

if not rows1:
    _("| — | — | — | — | No data |")
_("")

# ── [2] Policy Comparison ─────────────────────────────────────────────────────
_("---")
_("## [2] Policy Comparison")
_("")
_(f"Three selector policies on the same {samples_per} samples.  ")
_(f"Iterative mode, up to {max_iter} adaptive iterations.")
_("")
_("| Policy | Successes | Rate | Baseline | Tactic-Driven |")
_("|--------|-----------|------|----------|---------------|")

suite2  = load_runs("2_policy_comparison")
rows2   = aggregate(suite2, lambda r: r.get("policy_mode") or "unknown")
policy_order = ["random_choice", "agent_based_decision", "rl_bandit"]
rows2.sort(key=lambda r: (
    policy_order.index(r["key"]) if r["key"] in policy_order else 99
))

for r in rows2:
    _(f"| `{r['key']}` | {r['succ']}/{r['total']} | **{pct(r['rate'])}** "
      f"| {r['baseline']} | {r['tactic']} |")

if not rows2:
    _("| — | — | — | — | No data |")

_("")
_("> **Baseline** = raw model output fools the judge with no tactic applied.  ")
_("> **Tactic-Driven** = a tactic iteration caused the attack to succeed.")
_("")

# ── [3] Iteration Depth ───────────────────────────────────────────────────────
_("---")
_("## [3] Iteration Depth")
_("")
_(f"agent_based_decision (CoT) with iteration budgets 1, 3, and {max_iter}.  ")
_(f"N = {samples_per} samples per level.")
_("")
_("| Max Iterations | Successes | Rate | Distribution |")
_("|----------------|-----------|------|--------------|")

suite3 = load_runs("3_iteration_depth")
rows3  = aggregate(suite3, lambda r: int(r.get("max_iterations") or 0))
rows3.sort(key=lambda r: r["key"])

for r in rows3:
    _(f"| {r['key']} | {r['succ']}/{r['total']} | **{pct(r['rate'])}** "
      f"| `{bar(r['rate'])}` |")

if not rows3:
    _("| — | — | — | No data |")
_("")

# ── [4] CoT Ablation ──────────────────────────────────────────────────────────
_("---")
_("## [4] CoT Ablation")
_("")
_(f"agent_based_decision with chain-of-thought reasoning enabled vs. disabled.  ")
_(f"N = {samples_per} samples per condition.")
_("")
_("| Selector CoT | Successes | Rate | Distribution |")
_("|--------------|-----------|------|--------------|")

suite4 = load_runs("4_cot_ablation")

def _cot_key(r):
    v = r.get("selector_cot_enabled")
    if v is True:  return "Enabled"
    if v is False: return "Disabled"
    return "Unknown"

rows4 = aggregate(suite4, _cot_key)
rows4.sort(key=lambda r: r["key"] == "Disabled")  # Enabled first

for r in rows4:
    _(f"| {r['key']} | {r['succ']}/{r['total']} | **{pct(r['rate'])}** "
      f"| `{bar(r['rate'])}` |")

if not rows4:
    _("| — | — | — | No data |")
_("")

# ── Key Observations ──────────────────────────────────────────────────────────
_("---")
_("## Key Observations")
_("")

if rows1:
    best = rows1[0]
    worst = rows1[-1]
    _(f"- **Best tactic**: `{best['key']}` — {pct(best['rate'])} success rate")
    _(f"- **Worst tactic**: `{worst['key']}` — {pct(worst['rate'])} success rate")

if rows2:
    rows2_sorted = sorted(rows2, key=lambda r: -(r["rate"] or 0))
    ranking = " > ".join(f"`{r['key']}`" for r in rows2_sorted)
    _(f"- **Policy ranking**: {ranking}")
    if len(rows2_sorted) >= 2:
        gap = (rows2_sorted[0]["rate"] or 0) - (rows2_sorted[-1]["rate"] or 0)
        _(f"  (best vs. worst: {gap*100:+.1f} pp)")

if len(rows3) >= 2:
    r_min = min(rows3, key=lambda r: r["key"])
    r_max = max(rows3, key=lambda r: r["key"])
    delta = (r_max["rate"] or 0) - (r_min["rate"] or 0)
    _(f"- **Iteration benefit**: {r_max['key']} vs {r_min['key']} iterations → {delta*100:+.1f} pp")

cot_on  = next((r for r in rows4 if r["key"] == "Enabled"),  None)
cot_off = next((r for r in rows4 if r["key"] == "Disabled"), None)
if cot_on and cot_off:
    delta = (cot_on["rate"] or 0) - (cot_off["rate"] or 0)
    _(f"- **CoT effect**: Enabled vs Disabled → {delta*100:+.1f} pp")

_("")
_("---")
_("*Plots: `plots/` — Raw data: `results/` — Aggregated metrics: `aggregates/`*")

summary_path.write_text("\n".join(lines), encoding="utf-8")
print(f"  Written: {summary_path}")
PYEOF

# ─────────────────────────────────────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────────────────────────────────────
section "Done"

echo ""
echo -e "${BOLD}  Output directory:${RESET}  ${SESSION_DIR}/"
echo ""
echo -e "  ${BOLD}plots/${RESET}"
echo -e "  ├── ${CYAN}1_tactic_isolation/${RESET}    tactic_win_rate, attack_success_rate"
echo -e "  ├── ${CYAN}2_policy_comparison/${RESET}   policy bar charts, success_breakdown"
echo -e "  ├── ${CYAN}3_iteration_depth/${RESET}     one_shot_vs_iterative, iterations_to_success"
echo -e "  ├── ${CYAN}4_cot_ablation/${RESET}        CoT comparison"
echo -e "  └── ${CYAN}combined/${RESET}              full cross-suite view"
echo ""
echo -e "  ${BOLD}SUMMARY.md${RESET} — key numbers, ready for slides"
echo -e "  ${BOLD}run.log${RESET}    — full experiment log"
echo ""
echo -e "${DIM}  Model: $MODEL  |  Benchmark: $BENCHMARK  |  $SAMPLES samples/run${RESET}"

log "========================================================"
log "Demo complete.  Output: $SESSION_DIR"
log "========================================================"
