#!/bin/bash
# source venv/Scripts/activate

# ==============================================================================
# Adversarial Attack Evaluation Scripts
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Baseline (no mutations) - Just original MBPP
# ------------------------------------------------------------------------------
echo "Running baseline (no mutations)..."
inspect eval \
  V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen2.5:7b \
  -T max_iterations=0 \
  --limit 5

# ------------------------------------------------------------------------------
# 2. Misleading comments only (3 iterations)
# ------------------------------------------------------------------------------
echo "Running misleading comments attack..."
inspect eval \
  V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen2.5:7b \
  -T max_iterations=3 \
  -T mutation_strategy=random \
  -T use_misleading_comments=True \
  -T use_variable_renaming=False \
  -T use_instruction_perturbation=False \
  --limit 5

# ------------------------------------------------------------------------------
# 3. Variable renaming only (3 iterations)
# ------------------------------------------------------------------------------
echo "Running variable renaming attack..."
inspect eval \
  V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen2.5:7b \
  -T max_iterations=3 \
  -T mutation_strategy=random \
  -T use_misleading_comments=False \
  -T use_variable_renaming=True \
  -T use_instruction_perturbation=False \
  --limit 5

# ------------------------------------------------------------------------------
# 4. Full attack - Random mutations (5 iterations)
# ------------------------------------------------------------------------------
echo "Running full random attack..."
inspect eval \
  V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen2.5:7b \
  -T max_iterations=5 \
  -T mutation_strategy=random \
  -T use_misleading_comments=True \
  -T use_variable_renaming=True \
  -T use_instruction_perturbation=True \
  --limit 10

# ------------------------------------------------------------------------------
# 5. Full attack - Sequential mutations (5 iterations)
# ------------------------------------------------------------------------------
echo "Running full sequential attack..."
inspect eval \
  V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen2.5:7b \
  -T max_iterations=5 \
  -T mutation_strategy=sequential \
  -T use_misleading_comments=True \
  -T use_variable_renaming=True \
  -T use_instruction_perturbation=True \
  --limit 10

# ------------------------------------------------------------------------------
# 6. Heavy attack - 10 iterations random
# ------------------------------------------------------------------------------
echo "Running heavy attack (10 iterations)..."
inspect eval \
  V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen2.5:7b \
  -T max_iterations=10 \
  -T mutation_strategy=random \
  -T use_misleading_comments=True \
  -T use_variable_renaming=True \
  -T use_instruction_perturbation=True \
  --limit 20

# ------------------------------------------------------------------------------
# 7. Quick test - 2 samples, 2 iterations
# ------------------------------------------------------------------------------
echo "Running quick test..."
inspect eval \
  V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen2.5:7b \
  -T max_iterations=2 \
  -T mutation_strategy=random \
  -T use_misleading_comments=True \
  -T use_variable_renaming=True \
  --limit 2

echo "All evaluations complete! Check logs/ folder for results."