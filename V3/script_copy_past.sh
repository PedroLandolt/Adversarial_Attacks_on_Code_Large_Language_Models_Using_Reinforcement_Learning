# ----------------------------------------------------------------
# MBPP
# ----------------------------------------------------------------

inspect eval V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen3.5:0.8b \
  -T benchmark=mbpp \
  -T mutation_strategy=react \
  -T policy_mode=agent_based_decision \
  -T experiment_mode=one_shot \
  -T use_llm_judge=True \
  -T judge_model=ollama/qwen3.5:0.8b \
  -T selector_model=ollama/qwen3.5:0.8b \
  -T max_iterations=3 \
  --max-samples 1 \
  --limit 1

# ---------------------------------------------------------------- 

inspect eval V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen3.5:0.8b \
  -T benchmark=mbpp \
  -T mutation_strategy=react \
  -T policy_mode=agent_based_decision \
  -T experiment_mode=iterative \
  -T use_llm_judge=True \
  -T judge_model=ollama/qwen3.5:0.8b \
  -T selector_model=ollama/qwen3.5:0.8b \
  -T max_iterations=3 \
  --max-samples 1 \
  --limit 1


# ----------------------------------------------------------------

inspect eval V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen3.5:0.8b \
  -T benchmark=mbpp \
  -T mutation_strategy=react \
  -T policy_mode=random_choice \
  -T experiment_mode=one_shot \
  -T use_llm_judge=True \
  -T judge_model=ollama/qwen3.5:0.8b \
  -T selector_model=ollama/qwen3.5:0.8b \
  -T max_iterations=3 \
  --max-samples 1 \
  --limit 1

# ----------------------------------------------------------------

inspect eval V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen3.5:0.8b \
  -T benchmark=mbpp \
  -T mutation_strategy=react \
  -T policy_mode=random_choice \
  -T experiment_mode=iterative \
  -T use_llm_judge=True \
  -T judge_model=ollama/qwen3.5:0.8b \
  -T selector_model=ollama/qwen3.5:0.8b \
  -T max_iterations=3 \
  --max-samples 1 \
  --limit 1

# ----------------------------------------------------------------
# Humaneval
# ----------------------------------------------------------------

inspect eval V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen3.5:0.8b \
  -T benchmark=humaneval \
  -T mutation_strategy=react \
  -T policy_mode=agent_based_decision \
  -T experiment_mode=one_shot \
  -T use_llm_judge=True \
  -T judge_model=ollama/qwen3.5:0.8b \
  -T selector_model=ollama/qwen3.5:0.8b \
  -T max_iterations=3 \
  --max-samples 1 \
  --limit 1

# ---------------------------------------------------------------- 

inspect eval V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen3.5:0.8b \
  -T benchmark=humaneval \
  -T mutation_strategy=react \
  -T policy_mode=agent_based_decision \
  -T experiment_mode=iterative \
  -T use_llm_judge=True \
  -T judge_model=ollama/qwen3.5:0.8b \
  -T selector_model=ollama/qwen3.5:0.8b \
  -T max_iterations=3 \
  --max-samples 1 \
  --limit 1


# ----------------------------------------------------------------

inspect eval V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen3.5:0.8b \
  -T benchmark=humaneval \
  -T mutation_strategy=react \
  -T policy_mode=random_choice \
  -T experiment_mode=one_shot \
  -T use_llm_judge=True \
  -T judge_model=ollama/qwen3.5:0.8b \
  -T selector_model=ollama/qwen3.5:0.8b \
  -T max_iterations=3 \
  --max-samples 1 \
  --limit 1

# ----------------------------------------------------------------

inspect eval V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen3.5:0.8b \
  -T benchmark=humaneval \
  -T mutation_strategy=react \
  -T policy_mode=random_choice \
  -T experiment_mode=iterative \
  -T use_llm_judge=True \
  -T judge_model=ollama/qwen3.5:0.8b \
  -T selector_model=ollama/qwen3.5:0.8b \
  -T max_iterations=3 \
  --max-samples 1 \
  --limit 1

# ----------------------------------------------------------------
# RL_Bandit UCB1 - MBPP
# ----------------------------------------------------------------

inspect eval V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen3.5:0.8b \
  -T benchmark=mbpp \
  -T mutation_strategy=react \
  -T policy_mode=rl_bandit \
  -T bandit_algorithm=ucb1 \
  -T experiment_mode=iterative \
  -T use_llm_judge=True \
  -T judge_model=ollama/qwen3.5:0.8b \
  -T selector_model=ollama/qwen3.5:0.8b \
  -T max_iterations=3 \
  --max-samples 1 \
  --limit 1

# ----------------------------------------------------------------

inspect eval V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen3.5:0.8b \
  -T benchmark=mbpp \
  -T mutation_strategy=react \
  -T policy_mode=rl_bandit \
  -T bandit_algorithm=ucb1 \
  -T experiment_mode=iterative \
  -T use_llm_judge=True \
  -T judge_model=ollama/qwen3.5:0.8b \
  -T selector_model=ollama/qwen3.5:0.8b \
  -T max_iterations=500 \
  --max-samples 10 \
  --limit 10


# ----------------------------------------------------------------

inspect eval V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen3.5:0.8b \
  -T benchmark=mbpp \
  -T mutation_strategy=react \
  -T policy_mode=rl_bandit \
  -T bandit_algorithm=ucb1 \
  -T experiment_mode=iterative \
  -T use_llm_judge=True \
  -T judge_model=ollama/qwen3.5:0.8b \
  -T selector_model=ollama/qwen3.5:0.8b \
  -T max_iterations=90 \
  --max-samples 5 \
  --limit 5