"""DeepSeek API helper for teacher generation with clean output-training terms.

The DeepSeek model licence is MIT and its platform terms explicitly permit
using outputs to train other models. Keys are read from the environment;
nothing is stored in the repository.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

PROVIDERS = {
    "deepseek": {
        "url": "https://api.deepseek.com/chat/completions",
        "env": "DEEPSEEK_API_KEY",
        "models": ["deepseek-chat", "deepseek-reasoner"],
    },
}


def call_api(
    provider: str,
    model: str,
    key: str,
    prompt: str,
    *,
    temperature: float = 0.7,
    max_tokens: int = 900,
    timeout: int = 300,
) -> tuple[str, dict[str, int]]:
    """Return the assistant text and token usage for one chat completion."""

    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider}")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    request = urllib.request.Request(
        PROVIDERS[provider]["url"],
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            choices = body.get("choices") or []
            if not choices:
                raise ValueError("API response contained no choices")
            usage = body.get("usage", {}) or {}
            return str(choices[0]["message"]["content"]).strip(), {
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
            }
        except Exception as exc:  # noqa: BLE001 - retry once, then report
            last_error = exc
            time.sleep(2 + attempt * 3)
    raise RuntimeError(f"API call failed after retries: {last_error}")
