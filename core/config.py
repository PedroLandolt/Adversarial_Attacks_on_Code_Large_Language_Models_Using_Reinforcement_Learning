from dataclasses import dataclass
from typing import Optional

@dataclass
class LLMConfig:
    # provider: "google" ou "openai" (podes adicionar outros depois)
    provider: str = "google"

    # modelos (muda aqui e o resto não mexe)
    model_generate: str = "gemini-flash-latest"
    model_judge: str = "gemini-flash-latest"

    # chaves (lidas do env por omissão)
    google_api_key_env: str = "GOOGLE_API_KEY"

    # throttling/custos
    attempts: int = 4
    judge_k: int = 1
    seed: int = 0
    judge_threshold: float = 0.80

    # rate limit handling
    max_retries: int = 5
    default_wait_seconds: float = 12.0

    # opcional: se quiseres desligar o judge para poupar
    enable_judge: bool = True
