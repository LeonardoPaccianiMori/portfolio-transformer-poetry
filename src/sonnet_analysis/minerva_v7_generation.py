"""Matched, resumable BF16 generation across the seven frozen V7 states."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


GENERATION_VERSION = "minerva_7b_v7_matched_generation_v1"


def completed_non_empty_line_count(text: str) -> int:
    """Count only newline-terminated non-empty continuation lines."""

    if text == "":
        return 0
    completed = text if text.endswith(("\n", "\r")) else "\n".join(text.splitlines()[:-1])
    return sum(bool(line.strip()) for line in completed.splitlines())


def build_sonnet_candidate_prompt(tokenizer: Any, opening_line: str) -> str:
    """Render the frozen sonnet instruction without importing legacy training code."""

    if not opening_line.strip() or "\n" in opening_line or "\r" in opening_line:
        raise ValueError("opening_line must contain exactly one non-empty line")
    user_message = (
        "Componi un sonetto in italiano classico di esattamente quattordici "
        "versi. Usa come primo verso esattamente quello indicato, mantieni un "
        "tema coerente e una sintassi grammaticale, ed evita ripetizioni. "
        "Restituisci soltanto il sonetto, senza titolo, spiegazioni o commenti.\n\n"
        f"Primo verso: {opening_line}"
    )
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("Minerva chat template must render a non-empty string")
    return f"{rendered}{opening_line}\n"


def prepare_minerva_sampling_logits(
    logits: torch.Tensor,
    *,
    generated_token_ids: Sequence[int],
    temperature: float,
    top_k: int | None,
    top_p: float,
    repetition_penalty: float,
) -> torch.Tensor:
    """Apply the frozen continuation-only sampling filters."""

    if logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError("Minerva sampling logits must have shape (1, vocabulary)")
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be greater than 0 when provided")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in the interval (0, 1]")
    if repetition_penalty < 1:
        raise ValueError("repetition_penalty must be at least 1")

    filtered = logits.clone()
    if repetition_penalty != 1 and generated_token_ids:
        repeated_ids = sorted({
            token_id
            for token_id in generated_token_ids
            if 0 <= token_id < filtered.shape[-1]
        })
        if repeated_ids:
            repeated_logits = filtered[:, repeated_ids]
            filtered[:, repeated_ids] = torch.where(
                repeated_logits < 0,
                repeated_logits * repetition_penalty,
                repeated_logits / repetition_penalty,
            )
    filtered = filtered / temperature
    if top_k is not None:
        retained_count = min(top_k, filtered.shape[-1])
        threshold = torch.topk(filtered, retained_count, dim=-1).values[:, -1:]
        filtered = filtered.masked_fill(filtered < threshold, -torch.inf)
    if top_p < 1:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        sorted_remove = cumulative > top_p
        sorted_remove[:, 1:] = sorted_remove[:, :-1].clone()
        sorted_remove[:, 0] = False
        remove = torch.zeros_like(sorted_remove).scatter(
            dim=-1, index=sorted_indices, src=sorted_remove
        )
        filtered = filtered.masked_fill(remove, -torch.inf)
    if not torch.isfinite(filtered).any(dim=-1).all():
        raise RuntimeError("Minerva sampling filters removed every token")
    return filtered


def generate_matched_continuation(
    *,
    model: Any,
    tokenizer: Any,
    opening_line: str,
    device: torch.device | str,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    continuation_line_target: int,
) -> dict[str, Any]:
    """Generate one matched output with cached decoding and exact token evidence."""

    if no_repeat_ngram_size <= 0:
        raise ValueError("no_repeat_ngram_size must be positive")
    prompt = build_sonnet_candidate_prompt(tokenizer, opening_line)
    encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    resolved_device = torch.device(device)
    input_ids = encoded["input_ids"].to(resolved_device)
    attention_mask = torch.ones_like(input_ids)
    generated: list[int] = []
    current = input_ids
    past = None
    continuation = ""
    stop_reason = "max_new_tokens"
    generator = torch.Generator(device=resolved_device).manual_seed(seed)
    special_ids = {
        int(value) for value in getattr(tokenizer, "all_special_ids", [])
        if isinstance(value, int) and value >= 0
    }
    model.eval()
    started = time.monotonic()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            if completed_non_empty_line_count(continuation) >= continuation_line_target:
                stop_reason = "target_lines"
                break
            outputs = model(
                input_ids=current,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            logits = outputs.logits[:, -1, :].float()
            past = outputs.past_key_values
            if special_ids:
                logits[:, list(special_ids)] = -torch.inf
            banned = banned_next_tokens(generated, no_repeat_ngram_size)
            if banned:
                logits[:, list(banned)] = -torch.inf
            logits = prepare_minerva_sampling_logits(
                logits,
                generated_token_ids=generated,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            probabilities = torch.softmax(logits, dim=-1)
            if not torch.isfinite(probabilities).all():
                raise RuntimeError("matched generation produced invalid probabilities")
            token = torch.multinomial(probabilities, 1, generator=generator)
            generated.append(int(token.item()))
            continuation = tokenizer.decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            current = token
            attention_mask = torch.cat([attention_mask, torch.ones_like(token)], dim=1)
    if completed_non_empty_line_count(continuation) >= continuation_line_target:
        stop_reason = "target_lines"
    return {
        "text": f"{opening_line}\n{continuation}",
        "opening_line": opening_line,
        "conditioning_prompt": prompt,
        "conditioning_input_ids": input_ids[0].cpu().tolist(),
        "generated_token_ids": generated,
        "seed": seed,
        "stop_reason": stop_reason,
        "generated_new_tokens": len(generated),
        "completed_continuation_lines": completed_non_empty_line_count(continuation),
        "elapsed_seconds": time.monotonic() - started,
    }


def banned_next_tokens(generated: Sequence[int], n: int) -> set[int]:
    """Return tokens that would repeat an already generated n-gram."""

    if n <= 0:
        raise ValueError("n must be positive")
    if len(generated) + 1 < n:
        return set()
    prefix = tuple(generated[-(n - 1) :]) if n > 1 else ()
    banned = set()
    for start in range(0, len(generated) - n + 1):
        ngram = tuple(generated[start : start + n])
        if ngram[:-1] == prefix:
            banned.add(ngram[-1])
    return banned


def generate_state_outputs(
    *,
    model: Any,
    tokenizer: Any,
    state_id: str,
    state_identity_sha256: str,
    prompts: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    recipe: Mapping[str, Any],
    output_dir: Path,
    device: torch.device | str,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Generate or resume the exact prompt/seed grid for one verified state."""

    if len(set(seeds)) != len(seeds) or not prompts:
        raise ValueError("prompts and unique seeds are required")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    total = len(prompts) * len(seeds)
    started = time.monotonic()
    for prompt in prompts:
        for seed in seeds:
            identity = hashlib.sha256(
                f"{GENERATION_VERSION}|{state_id}|{prompt['id']}|{seed}".encode("utf-8")
            ).hexdigest()[:16]
            path = output_dir / f"{identity}.json"
            if path.is_file():
                row = json.loads(path.read_text(encoding="utf-8"))
                _validate_generation_row(
                    row, state_id=state_id, state_identity_sha256=state_identity_sha256,
                    prompt=prompt, seed=seed, recipe=recipe,
                )
                status = "reused"
            else:
                result = generate_matched_continuation(
                    model=model,
                    tokenizer=tokenizer,
                    opening_line=str(prompt["opening_line"]),
                    device=device,
                    seed=seed,
                    max_new_tokens=int(recipe["max_new_tokens"]),
                    temperature=float(recipe["temperature"]),
                    top_k=recipe["top_k"],
                    top_p=float(recipe["top_p"]),
                    repetition_penalty=float(recipe["repetition_penalty"]),
                    no_repeat_ngram_size=int(recipe["no_repeat_ngram_size"]),
                    continuation_line_target=int(recipe["continuation_line_target"]),
                )
                row = {
                    "generation_version": GENERATION_VERSION,
                    "state_id": state_id,
                    "state_identity_sha256": state_identity_sha256,
                    "prompt": dict(prompt),
                    "seed_role": "confirmatory" if seed == int(recipe["confirmatory_seed"]) else "exploratory_replication",
                    "recipe": dict(recipe),
                    **result,
                    "v7_test_accessed": False,
                }
                temporary = path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                temporary.rename(path)
                status = "written"
            rows.append(
                {
                    "path": path.name,
                    "sha256": _sha256(path),
                    "prompt_id": prompt["id"],
                    "seed": seed,
                }
            )
            if progress:
                elapsed = time.monotonic() - started
                progress(
                    f"output={len(rows)}/{total} progress={100 * len(rows) / total:.1f}% "
                    f"status={status} elapsed={elapsed:.1f}s eta={elapsed/len(rows)*(total-len(rows)):.1f}s"
                )
    completion = {
        "generation_version": GENERATION_VERSION,
        "state_id": state_id,
        "state_identity_sha256": state_identity_sha256,
        "prompt_count": len(prompts),
        "seeds": list(seeds),
        "output_count": len(rows),
        "completion_scope": (
            "authoritative_24_prompt_3_seed_grid"
            if len(prompts) == 24 and list(seeds) == [4099, 4100, 4101]
            else "bounded_non_authoritative_run"
        ),
        "outputs": rows,
        "v7_test_accessed": False,
        "causal_experiments_performed": False,
    }
    (output_dir / "complete.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return completion


def _validate_generation_row(
    row: Mapping[str, Any], *, state_id: str, state_identity_sha256: str,
    prompt: Mapping[str, Any], seed: int, recipe: Mapping[str, Any],
) -> None:
    expected = {
        "generation_version": GENERATION_VERSION,
        "state_id": state_id,
        "state_identity_sha256": state_identity_sha256,
        "prompt": dict(prompt),
        "seed": seed,
        "recipe": dict(recipe),
        "v7_test_accessed": False,
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"existing matched generation mismatch: {key}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest
