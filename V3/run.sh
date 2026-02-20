inspect eval V3/tasks/mbpp_attack.py@mbpp_attack \
  --model google/gemini-flash-latest \
  -T temperature=0.0 \
  --limit 10


inspect eval V3/tasks/mbpp_attack.py@mbpp_attack \
  --model google/gemini-flash-latest \
  -T temperature=0.0 \
  -T use_misleading_comments=True \
  --limit 10