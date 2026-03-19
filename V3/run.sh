# source venv/Scripts/activate
#!/bin/bash

# =============================================================================
# Adversarial Attacks on Code LLMs - ReAct Agent Evaluation Suite
# =============================================================================
#
# This script evaluates the ReAct agent loop which:
#   - Dynamically selects red-teaming tactics based on judge feedback
#   - Implements action → observation → reasoning → next action loop
#   - Stops early when attack succeeds (test=FAIL + llm=PASS)
#
# Usage:
#   bash V3/run.sh          # Run all tests
#   bash V3/run.sh --quick  # Run quick subset (2 samples each)
#
# =============================================================================

set -e  # Exit on error

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

MODEL="ollama/qwen3.5:9b" #qwen2.5:7b
JUDGE_MODEL="ollama/qwen3.5:9b" # qwen3.5:9b
SAMPLES=5        # Number of samples per test
QUICK_SAMPLES=2  # Samples for quick mode

# Parse arguments
QUICK_MODE=false

for arg in "$@"; do
    case $arg in
        --quick)
            QUICK_MODE=true
            SAMPLES=$QUICK_SAMPLES
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

print_header() {
    echo ""
    echo "============================================================================="
    echo "$1"
    echo "============================================================================="
    echo ""
}

print_test() {
    echo "-----------------------------------------------------------------------------"
    echo "[$1] $2"
    echo "-----------------------------------------------------------------------------"
}

run_react() {
    inspect eval \
        V3/adversarial_attack.py@adversarial_code_llm \
        --model "$MODEL" \
        -T mutation_strategy=react \
        -T use_llm_judge=True \
        -T judge_model="$JUDGE_MODEL" \
        "$@"
}

# -----------------------------------------------------------------------------
# Start
# -----------------------------------------------------------------------------

print_header "REACT AGENT - ADVERSARIAL ATTACK EVALUATION"

echo "Configuration:"
echo "  Model:        $MODEL"
echo "  Judge Model:  $JUDGE_MODEL"
echo "  Samples:      $SAMPLES"
echo "  Quick Mode:   $QUICK_MODE"
echo ""
echo "ReAct Loop:"
echo "  - LLM selector chooses tactics dynamically"
echo "  - Feedback from judges drives next action"
echo "  - Early stop when attack succeeds"
echo ""

START_TIME=$(date +%s)

# =============================================================================
# TEST 1: BASELINE - ReAct Basic
# =============================================================================

# print_header "TEST 1: BASELINE"

# print_test "1.1" "ReAct baseline (3 iterations)"
# run_react \
#     -T max_iterations=3 \
#     --limit "$SAMPLES"

# =============================================================================
# TEST 2: ITERATION ABLATION
# =============================================================================

# print_header "TEST 2: ITERATION ABLATION"

# print_test "2.1" "ReAct with 1 iteration (minimal)"
# run_react \
#     -T max_iterations=1 \
#     --limit "$SAMPLES"

# print_test "2.2" "ReAct with 3 iterations (standard)"
# run_react \
#     -T max_iterations=3 \
#     --limit "$SAMPLES"

print_test "2.3" "ReAct with 5 iterations (extended)"
run_react \
    -T max_iterations=5 \
    --limit "$SAMPLES"

# print_test "2.4" "ReAct with 10 iterations (aggressive)"
# run_react \
#     -T max_iterations=10 \
#     --limit "$SAMPLES"

# =============================================================================
# TEST 3: SAMPLE SIZE SCALING
# =============================================================================

# print_header "TEST 3: SAMPLE SIZE SCALING"

# print_test "3.1" "ReAct with 5 samples"
# run_react \
#     -T max_iterations=3 \
#     --limit 5

# print_test "3.2" "ReAct with 10 samples"
# run_react \
#     -T max_iterations=3 \
#     --limit 10

# print_test "3.3" "ReAct with 20 samples"
# run_react \
#     -T max_iterations=3 \
#     --limit 20

# =============================================================================
# TEST 4: CONVERGENCE ANALYSIS
# =============================================================================

# print_header "TEST 4: CONVERGENCE ANALYSIS"

# print_test "4.1" "Convergence test - How many iterations until success? (max 15)"
# run_react \
#     -T max_iterations=15 \
#     --limit 10

# =============================================================================
# TEST 5: STRESS TEST
# =============================================================================

# print_header "TEST 5: STRESS TEST"

# print_test "5.1" "Stress test - Many samples, many iterations"
# run_react \
#     -T max_iterations=5 \
#     --limit 30

# =============================================================================
# SUMMARY
# =============================================================================

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

print_header "EVALUATION COMPLETE"

echo "Summary:"
echo "  Total time:     ${DURATION}s"
echo "  Logs location:  ./logs/"
echo ""
echo "Tests executed:"
echo ""
echo "  [Test 1] Baseline"
echo "    1.1 ReAct baseline (3 iter, $SAMPLES samples)"
echo ""
echo "  [Test 2] Iteration Ablation"
echo "    2.1 max_iterations=1  (minimal)"
echo "    2.2 max_iterations=3  (standard)"
echo "    2.3 max_iterations=5  (extended)"
echo "    2.4 max_iterations=10 (aggressive)"
echo ""
echo "  [Test 3] Sample Scaling"
echo "    3.1 5 samples"
echo "    3.2 10 samples"
echo "    3.3 20 samples"
echo ""
echo "  [Test 4] Convergence"
echo "    4.1 Track iterations to success (max 15 iter)"
echo ""
echo "  [Test 5] Stress Test"
echo "    5.1 30 samples, 5 iterations"
echo ""
echo "Key metrics to analyze:"
echo "  - Attack Success Rate (ASR) per iteration count"
echo "  - Average iterations to success"
echo "  - Tactic distribution (which tactics worked?)"
echo "  - Judge disagreement rate"
echo ""
echo "To analyze results:"
echo "  inspect view"
echo ""
echo "============================================================================="
echo "Done!"
echo "============================================================================="