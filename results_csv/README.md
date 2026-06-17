# results_csv — Pre-computed thesis result tables

Generated from `stored_results/` attempts.jsonl files. Re-run `rebuild_csvs.py` to refresh after new runs complete.

## Files

### asr_by_policy_judge_corpus.csv
Columns: `judge, judge_tag, policy, corpus, n, baseline_asr, asr_1shot, asr_3shot, asr_6shot`

Covers all policies × all 4 judges × both corpora. Use this for RQ1, RQ3, RQ4, RQ5.

- `baseline_asr` = iteration 0 wins (no tactic at all)
- `asr_1shot` = wins within 1 tactic attempt (iterations 0–1)
- `asr_3shot` = wins within 3 tactic attempts (iterations 0–3)
- `asr_6shot` = wins within 6 tactic attempts (iterations 0–6) = full run

**Primary thesis metric is `asr_1shot`.** 3-shot and 6-shot are secondary convergence analysis.

Bandit rows (ucb1/thompson/kl-ucb/exp3) use eval split (frozen weights after training).
Starcoder bandit rows are absent — only random and react were run for starcoder.

### rq2_tactic_effectiveness.csv
Columns: `tactic, judge, judge_tag, corpus, n, baseline_asr, tactic_asr, combined_asr`

Covers all 9 tactics × 4 judges × 2 corpora. Use this for RQ2 and Appendix B.

- `baseline_asr` = iteration 0 wins (judge approves buggy code with no attack)
- `tactic_asr` = iteration 1 wins only (pure tactic contribution, where baseline failed)
- `combined_asr` = baseline + tactic wins (total ASR for that run)

**Use `tactic_asr` for tactic effectiveness claims. Use `combined_asr` for overall run ASR.**

8 starcoder-cubert rows are NULL — those runs finish ~2 PM Jun 16. Re-run `rebuild_csvs.py` after.

## To refresh after starcoder-cubert runs complete

```bash
cd Adversarial_Attacks_on_Code_Large_Language_Models_Using_Reinforcement_Learning
python results_csv/rebuild_csvs.py
```
