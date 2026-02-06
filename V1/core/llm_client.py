import os
import re
from dataclasses import dataclass
from typing import Protocol, Tuple

from .config import LLMConfig

# Providers (imports só aqui)
from google import genai as google_genai


class LLMClient(Protocol):
    def generate_code(self, prompt: str) -> str: ...
    def judge(self, problem: str, code: str) -> Tuple[float, str]: ...


def _strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:python)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


@dataclass
class GoogleLLMClient:
    cfg: LLMConfig
    client: google_genai.Client

    def generate_code(self, prompt: str) -> str:
        resp = self.client.models.generate_content(
            model=self.cfg.model_generate,
            contents=prompt
        )
        return _strip_code_fences(resp.text)

    def judge(self, problem: str, code: str):
        judge_prompt = f"""
                            You are a strict reviewer evaluating whether a candidate Python solution is correct.

                            Score from 0 to 1 how likely the solution is correct.
                            Return exactly two lines:
                            SCORE: <number between 0 and 1>
                            REASON: <one short sentence>

                            Problem:
                            {problem}

                            Candidate code:
                            {code}
                        """
        resp = self.client.models.generate_content(
            model=self.cfg.model_judge,
            contents=judge_prompt
        )
        out = (resp.text or "").strip()

        score = 0.0
        reason = out.replace("\n", " ")[:200]

        m = re.search(r"SCORE:\s*([0-1](?:\.\d+)?)", out)
        if m:
            try:
                score = float(m.group(1))
            except Exception:
                score = 0.0

        m2 = re.search(r"REASON:\s*(.+)", out)
        if m2:
            reason = m2.group(1).strip()[:200]

        score = max(0.0, min(1.0, score))
        return score, reason



def make_client(cfg: LLMConfig) -> LLMClient:
    if cfg.provider == "google":
        key = os.getenv(cfg.google_api_key_env)
        if not key:
            raise RuntimeError(f"{cfg.google_api_key_env} not set")
        return GoogleLLMClient(cfg=cfg, client=google_genai.Client(api_key=key))

    # if cfg.provider == "openai":

    raise RuntimeError(f"Unknown provider: {cfg.provider}")
