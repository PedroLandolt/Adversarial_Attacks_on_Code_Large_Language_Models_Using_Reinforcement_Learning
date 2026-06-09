# run_policy_comparison.ps1
# Compares three selector policies against the strong judge (qwen2.5-coder:7b).
# Run from the project root:
#   .\V3\scripts\run_policy_comparison.ps1
#
# Policy   | Selector LLM         | Tactic selection method
# ---------|----------------------|------------------------
# random   | llama3.1:8b (gen)    | Uniform random
# agent    | llama3.1:8b (ReAct)  | Chain-of-thought reasoning
# bandit   | llama3.1:8b (gen)    | UCB1 multi-armed bandit

$INSPECT = "C:\Users\pedro\Desktop\Pedro\Tese\.venv\Scripts\inspect"
$env:PYTHONPATH = "V3"

$COMMON_FLAGS = @(
    "V3/adversarial_attack.py@adversarial_code_llm",
    "--model", "ollama/llama3.1:8b",
    "-T", "benchmark=mbpp",
    "-T", "mutation_strategy=react",
    "-T", "experiment_mode=one_shot",
    "-T", "use_llm_judge=True",
    "-T", "target_model=ollama/qwen2.5-coder:7b",
    "-T", "selector_model=ollama/llama3.1:8b",
    "-T", "max_iterations=1",
    "--max-samples", "50",
    "--limit", "50"
)

Write-Host "=== RUN 1/2: agent_based_decision (ReAct) ===" -ForegroundColor Cyan
& $INSPECT eval @COMMON_FLAGS -T policy_mode=agent_based_decision

Write-Host ""
Write-Host "=== RUN 2/2: rl_bandit (UCB1) ===" -ForegroundColor Cyan
& $INSPECT eval @COMMON_FLAGS -T policy_mode=rl_bandit

Write-Host ""
Write-Host "=== ALL RUNS COMPLETE ===" -ForegroundColor Green
Write-Host "Baseline (random, previous run): 34% ASR"
Write-Host "Check results/ for agent and bandit ASR numbers."
