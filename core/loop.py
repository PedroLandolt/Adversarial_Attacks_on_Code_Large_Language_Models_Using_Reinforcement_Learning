import random
from typing import List, Optional

from .config import LLMConfig
from .types import MBPPProblem, Candidate
from .prompts import build_base_prompt
from .mutators import Mutator, DEFAULT_MUTATORS
from .tests_runner import run_tests
from .judge import judge_solution
from .llm_client import LLMClient
from .rate_limit import with_rate_limit_retry

def is_attack_success(c: Candidate, threshold: float) -> bool:
    return (c.judge_score >= threshold) and (not c.tests_pass)

def attack_problem(
    client: LLMClient,
    cfg: LLMConfig,
    problem: MBPPProblem,
    mutators: Optional[List[Mutator]] = None,
) -> List[Candidate]:
    rng = random.Random(cfg.seed)
    mutators = mutators or DEFAULT_MUTATORS
    base_prompt = build_base_prompt(problem)

    # 1) gerar candidatos + testar
    candidates: List[Candidate] = []
    for _ in range(cfg.attempts):
        mut = rng.choice(mutators)
        mutated_prompt = mut(base_prompt, rng)

        code = with_rate_limit_retry(
            lambda: client.generate_code(mutated_prompt),
            cfg=cfg
        )

        tests_pass, _ = run_tests(code, problem.tests)

        candidates.append(Candidate(
            code=code,
            tests_pass=tests_pass,
            judge_score=0.0,
            judge_reason="(not judged)",
            mutated_prompt=mutated_prompt
        ))

    # 2) julgar só K que falharam (poupa quota)
    if cfg.enable_judge:
        failed = [c for c in candidates if not c.tests_pass]
        to_judge = failed[:cfg.judge_k]

        for c in to_judge:
            score, reason = with_rate_limit_retry(
                lambda: judge_solution(client, problem.prompt, c.code),
                cfg=cfg
            )
            c.judge_score = score
            c.judge_reason = reason

    # 3) ordenar: queremos “engano” em cima
    candidates.sort(key=lambda c: (c.tests_pass, -c.judge_score))
    return candidates
