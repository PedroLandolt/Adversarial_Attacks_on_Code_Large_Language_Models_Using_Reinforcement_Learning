# #!/bin/bash

# echo "Testing ReAct Loop - Phase 3"
# echo "===================================="
# echo ""

# # Test 1: ReAct loop com LLM judge
# echo "[1/2] ReAct loop with LLM judge selector..."
# inspect eval \
#   V3/adversarial_attack.py@adversarial_code_llm \
#   --model ollama/qwen2.5:7b \
#   -T use_llm_judge=True \
#   -T judge_model=ollama/qwen2.5:7b \
#   -T use_react_loop=True \
#   -T max_iterations=3 \
#   --limit 2

# echo ""
# echo "[2/2] ReAct loop with random tactic (fallback test)..."
# inspect eval \
#   V3/adversarial_attack.py@adversarial_code_llm \
#   --model ollama/qwen2.5:7b \
#   -T use_llm_judge=True \
#   -T judge_model=ollama/qwen2.5:7b \
#   -T use_react_loop=True \
#   -T max_iterations=2 \
#   --limit 1

# echo ""
# echo "✅ ReAct tests complete!"

#!/bin/bash

#!/bin/bash

echo "Testing ReAct Strategy"
echo "======================"

# ReAct loop with LLM selector
inspect eval \
  V3/adversarial_attack.py@adversarial_code_llm \
  --model ollama/qwen2.5:7b \
  -T mutation_strategy=react \
  -T use_llm_judge=True \
  -T judge_model=ollama/qwen2.5:7b \
  -T max_iterations=3 \
  --limit 2

echo ""
echo "Done!"