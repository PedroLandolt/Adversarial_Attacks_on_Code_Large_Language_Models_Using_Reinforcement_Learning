# Baseline test (no mutations)
inspect eval \
  V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen2.5:7b \
  --limit 5

# Full attack (misleading comments + variable renaming)
inspect eval \
  V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen2.5:7b \
  -T max_iterations=5 \
  -T use_misleading_comments=True \
  -T use_variable_renaming=True \
  --limit 10