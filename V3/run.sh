# source venv/Scripts/activate
#!/bin/bash

# ==============================================================================
# Adversarial Attack Evaluation - Complete Pipeline
# ==============================================================================

set -e  # Exit on error

echo "=========================================="
echo "Starting Adversarial Attack Evaluations"
echo "=========================================="
echo ""

# ==============================================================================
# Phase 1: Baseline & Mutation Strategies
# ==============================================================================

echo "[1/11] Baseline (no mutations)..."
inspect eval \
  V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen2.5:7b \
  -T max_iterations=0 \
  --limit 5

echo ""
echo "[2/11] Misleading comments attack (3 iterations)..."
inspect eval \
  V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen2.5:7b \
  -T max_iterations=3 \
  -T mutation_strategy=random \
  -T use_misleading_comments=True \
  -T use_variable_renaming=False \
  -T use_instruction_perturbation=False \
  --limit 5

echo ""
echo "[3/11] Variable renaming attack (3 iterations)..."
inspect eval \
  V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen2.5:7b \
  -T max_iterations=3 \
  -T mutation_strategy=random \
  -T use_misleading_comments=False \
  -T use_variable_renaming=True \
  -T use_instruction_perturbation=False \
  --limit 5

echo ""
echo "[4/11] Full random mutations (5 iterations)..."
inspect eval \
  V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen2.5:7b \
  -T max_iterations=5 \
  -T mutation_strategy=random \
  -T use_misleading_comments=True \
  -T use_variable_renaming=True \
  -T use_instruction_perturbation=True \
  --limit 10

echo ""
echo "[5/11] Full sequential mutations (5 iterations)..."
inspect eval \
  V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen2.5:7b \
  -T max_iterations=5 \
  -T mutation_strategy=sequential \
  -T use_misleading_comments=True \
  -T use_variable_renaming=True \
  -T use_instruction_perturbation=True \
  --limit 10

# ==============================================================================
# Phase 2: LLM-as-Judge Integration
# ==============================================================================

echo ""
echo "[6/11] LLM-as-Judge (baseline, no red-teaming)..."
inspect eval \
  V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen2.5:7b \
  -T use_llm_judge=True \
  -T judge_model=ollama/qwen2.5:7b \
  -T max_iterations=2 \
  --limit 3

# ==============================================================================
# Phase 3: Red-teaming Tactics
# ==============================================================================

echo ""
echo "[7/11] Red-teaming Tactic 1: Prompt Injection..."
inspect eval \
  V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen2.5:7b \
  -T use_llm_judge=True \
  -T judge_model=ollama/qwen2.5:7b \
  -T red_teaming_tactic=injection \
  -T max_iterations=2 \
  --limit 3

echo ""
echo "[8/11] Red-teaming Tactic 2: Output Manipulation..."
inspect eval \
  V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen2.5:7b \
  -T use_llm_judge=True \
  -T judge_model=ollama/qwen2.5:7b \
  -T red_teaming_tactic=output \
  -T max_iterations=2 \
  --limit 3

echo ""
echo "[9/11] Red-teaming Tactic 3: Semantic Inconsistency..."
inspect eval \
  V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen2.5:7b \
  -T use_llm_judge=True \
  -T judge_model=ollama/qwen2.5:7b \
  -T red_teaming_tactic=semantic \
  -T max_iterations=2 \
  --limit 3

echo ""
echo "[10/11] Red-teaming Tactic 4: Chain-of-Thought Poisoning..."
inspect eval \
  V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen2.5:7b \
  -T use_llm_judge=True \
  -T judge_model=ollama/qwen2.5:7b \
  -T red_teaming_tactic=cot \
  -T max_iterations=2 \
  --limit 3

# ==============================================================================
# Phase 4: Quick Combined Test
# ==============================================================================

echo ""
echo "[11/11] Quick test: Mutations + LLM Judge + Red-teaming..."
inspect eval \
  V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen2.5:7b \
  -T max_iterations=2 \
  -T mutation_strategy=random \
  -T use_misleading_comments=True \
  -T use_variable_renaming=True \
  -T use_llm_judge=True \
  -T judge_model=ollama/qwen2.5:7b \
  -T red_teaming_tactic=injection \
  --limit 2

# ==============================================================================
# Summary
# ==============================================================================

echo ""
echo "=========================================="
echo "✅ All evaluations complete!"
echo "=========================================="
echo "Check logs/ folder for detailed results"
echo ""