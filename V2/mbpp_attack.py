"""
MBPP adversarial attack task (Inspect AI)

Objetivo:
- Carregar o dataset MBPP (Hugging Face).
- Para cada problema, gerar código com o LLM.
- Opcionalmente "mutar" o prompt (ex.: inserir comentários) para atacar / alterar comportamento.
- Avaliar o código executando test_setup_code + test_list.

Nota importante (Windows):
- Timeouts seguros exigem subprocesso.
- No Windows, multiprocessing usa "spawn" -> o target do Process TEM de ser picklable.
- Portanto: NUNCA definir o worker dentro de outra função.
"""

import random
import multiprocessing as mp
from typing import Any, Dict, List, Optional, Tuple

from inspect_ai import Task, task
from inspect_ai.dataset import hf_dataset, FieldSpec
from inspect_ai.scorer import Score, Target, accuracy, scorer
from inspect_ai.solver import Generate, TaskState, solver

from config import AttackConfig
from mutators import random_comment_mutator


# =============================================================================
# 1) Helpers para mensagens (compatibilidade entre versões do inspect_ai)
# =============================================================================
def _make_user_message(content: str):
    """
    Cria uma mensagem 'user' de forma compatível com diferentes versões do inspect_ai.
    Algumas versões têm ChatMessageUser, outras só têm ChatMessage(role=...).
    """
    # Tenta o tipo dedicado (se existir)
    try:
        from inspect_ai.model import ChatMessageUser  # type: ignore
        return ChatMessageUser(content=content)
    except Exception:
        pass

    # Fallback: ChatMessage genérica
    from inspect_ai.model import ChatMessage  # type: ignore
    try:
        # Em várias versões, ChatMessage aceita role="user"
        return ChatMessage(role="user", content=content)
    except TypeError:
        # Último fallback possível (caso o construtor seja diferente)
        # Mantém compatibilidade sem rebentar no import.
        return ChatMessage(content=content)


# =============================================================================
# 2) Execução de código MBPP (subprocess + timeout)
# =============================================================================
def _run_mbpp_in_subprocess(code: str, setup: str, tests: List[str]) -> Tuple[bool, str]:
    """
    Executa (code + setup + tests).
    Esta função corre DENTRO do subprocesso.
    Retorna: (passou, mensagem).
    """
    # Builtins mínimos (para reduzir surface area).
    # Nota: isto não é um "sandbox" perfeito, mas limita o básico.
    safe_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "pow": pow,
        "range": range,
        "reversed": reversed,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }

    glb: Dict[str, Any] = {"__builtins__": safe_builtins}
    loc: Dict[str, Any] = {}

    try:
        if code:
            exec(code, glb, loc)
        if setup:
            exec(setup, glb, loc)

        for t in tests:
            exec(t, glb, loc)

        return True, "all tests passed"

    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _mbpp_worker_entry(q: "mp.Queue", code: str, setup: str, tests: List[str]) -> None:
    """
    Entry point do subprocesso (TEM de ser top-level para Windows spawn).
    Corre o runner e devolve (ok, msg) via Queue.
    """
    ok, msg = _run_mbpp_in_subprocess(code, setup, tests)
    q.put((ok, msg))


def _run_with_timeout(code: str, setup: str, tests: List[str], timeout_s: float = 5.0) -> Tuple[bool, str]:
    """
    Executa a validação num subprocesso e aplica timeout.
    - Usa mp.get_context("spawn") explicitamente (mais consistente em Windows).
    - Worker é top-level (picklable).
    """
    ctx = mp.get_context("spawn")
    q: "mp.Queue" = ctx.Queue()

    p = ctx.Process(
        target=_mbpp_worker_entry,
        args=(q, code, setup, tests),
        daemon=True,
    )

    p.start()
    p.join(timeout_s)

    # Timeout: mata o processo
    if p.is_alive():
        p.terminate()
        p.join(1.0)
        return False, f"TimeoutError: exceeded {timeout_s}s"

    # Processo terminou mas não devolveu nada (anormal)
    if q.empty():
        return False, "RuntimeError: no result from worker"

    return q.get()


# =============================================================================
# 3) Scorer: executa output do modelo e calcula score
# =============================================================================
@scorer(metrics=[accuracy()])
def mbpp_exec_scorer(timeout_s: float = 5.0):
    """
    Scorer custom:
    - Lê state.output.completion (código gerado pelo modelo)
    - Puxa setup/tests a partir do metadata do sample
    - Executa em subprocesso com timeout
    - Score = 1.0 se passar, senão 0.0
    """

    async def score(state: TaskState, target: Target) -> Score:
        # Output do modelo (código). Se vier None, usa string vazia.
        code = (state.output.completion or "").strip()

        # Metadata do dataset MBPP (definido no FieldSpec)
        setup = state.metadata.get("test_setup_code") or ""
        tests = state.metadata.get("test_list") or []

        # Defensive: às vezes pode vir como string única
        if isinstance(tests, str):
            tests = [tests]

        passed, msg = _run_with_timeout(code, setup, tests, timeout_s=timeout_s)

        return Score(
            value=1.0 if passed else 0.0,
            answer=code[:5000],          # evita logs gigantes
            explanation=None if passed else msg,
        )

    return score


# =============================================================================
# 4) Solver: aplica "attack" ao prompt antes de chamar o generate()
# =============================================================================
@solver
def attack_solver(cfg: AttackConfig):
    """
    Solver:
    - Lê o prompt original (input) que vem do dataset.
    - Faz N tentativas (cfg.attempts).
    - Se cfg.use_mutation=True, aplica mutator (ex.: inserir comentários).
    - Chama generate(state) para obter completion do modelo.
    """

    async def solve(state: TaskState, generate: Generate):
        rng = random.Random(cfg.seed)

        # Prompt original: em geral está acessível em state.input_text.
        # Se a tua versão tiver outra API, troca só esta linha.
        base_prompt = getattr(state, "input_text", None) or ""

        best_state: Optional[TaskState] = None

        for _ in range(cfg.attempts):
            prompt = base_prompt

            if cfg.use_mutation:
                prompt = random_comment_mutator(prompt, rng)

            # Atualiza mensagens do state para forçar o "user prompt".
            # Isto evita depender de métodos .replace/.with_input que não existem
            # em certas versões.
            state.messages = [_make_user_message(prompt)]

            # Chama o LLM
            out_state = await generate(state)
            best_state = out_state

        return best_state or state

    return solve


# =============================================================================
# 5) Task: dataset + solver + scorer
# =============================================================================
@task
def mbpp_attack(
    attempts: int = 1,
    use_mutation: bool = False,
):
    """
    Task MBPP Attack:
    - dataset: hf_dataset MBPP split test
    - solver: attack_solver
    - scorer: mbpp_exec_scorer (exec + timeout)
    """
    cfg = AttackConfig(
        attempts=attempts,
        use_mutation=use_mutation,
    )

    dataset = hf_dataset(
        path="google-research-datasets/mbpp",
        split="test",
        sample_fields=FieldSpec(
            input="text",
            id="task_id",
            metadata=["test_list", "test_setup_code"],
        ),
    )

    return Task(
        dataset=dataset,
        solver=attack_solver(cfg),
        scorer=mbpp_exec_scorer(timeout_s=5.0),
    )
