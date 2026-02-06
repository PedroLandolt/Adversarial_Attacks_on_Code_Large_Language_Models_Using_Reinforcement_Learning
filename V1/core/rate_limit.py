import time
import re

def with_rate_limit_retry(fn, cfg, max_retries: int | None = None):
    max_retries = max_retries or cfg.max_retries

    for _ in range(max_retries):
        try:
            return fn()
        except Exception as e:
            msg = str(e)

            # quota diária free-tier -> não adianta retry
            if "GenerateRequestsPerDayPerProjectPerModel-FreeTier" in msg:
                raise RuntimeError(
                    "Daily free-tier quota exhausted for this model/provider. "
                    "Wait for reset, switch project/model, or enable billing."
                ) from e

            # rate-limit temporário
            if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                m = re.search(r"Please retry in ([0-9.]+)s", msg)
                wait_s = float(m.group(1)) if m else cfg.default_wait_seconds
                time.sleep(wait_s)
                continue

            raise
    return fn()
