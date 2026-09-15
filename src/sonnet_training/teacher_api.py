"""Provider API helpers for teacher generation with clean output-training terms.

Keys are read from the environment; nothing is stored in the repository.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

PROVIDERS = {
    "deepseek": {
        "url": "https://api.deepseek.com/chat/completions",
        "env": "DEEPSEEK_API_KEY",
        "models": ["deepseek-chat", "deepseek-reasoner"],
    },
    "mistral": {
        "url": "https://api.mistral.ai/v1/chat/completions",
        "env": "MISTRAL_API_KEY",
        "models": ["mistral-large-latest", "mistral-medium-latest"],
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
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    usage = body.get("usage", {}) or {}
    return str(body["choices"][0]["message"]["content"]).strip(), {
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
    }
