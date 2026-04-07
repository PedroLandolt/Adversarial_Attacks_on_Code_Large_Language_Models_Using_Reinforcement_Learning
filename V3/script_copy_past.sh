inspect eval V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen3.5:0.8b \
  -T benchmark=mbpp \
  -T mutation_strategy=react \
  -T policy_mode=agent_based_decision \
  -T use_llm_judge=True \
  -T judge_model=ollama/qwen3.5:0.8b \
  -T selector_model=ollama/qwen3.5:0.8b \
  -T max_iterations=1 \
  --max-samples 1 \
  --limit 1

# ---------------------------------------------------------------- 

inspect eval V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen3.5:0.8b \
  -T benchmark=mbpp \
  -T mutation_strategy=react \
  -T policy_mode=random_choice \
  -T use_llm_judge=True \
  -T judge_model=ollama/qwen3.5:0.8b \
  -T selector_model=ollama/qwen3.5:0.8b \
  -T max_iterations=1 \
  --max-samples 1 \
  --limit 1