"""Fixed held-out generation for untouched and QLoRA-adapted Minerva 3B."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

from sonnet_evaluation.generation import (
    completed_non_empty_line_count,
    safe_prompt_filename,
)
from sonnet_evaluation.task_generation import (
    ACCEPTANCE_SEEDS,
    ACCEPTANCE_TEMPERATURE,
    ACCEPTANCE_TOP_K,
    TASK_CONTINUATION_LINE_TARGET,
    TASK_FORMAT_VERSION,
)
from sonnet_training.minerva_qlora import (
    MINERVA_3B_MODEL_ID,
    MINERVA_3B_REVISION,
)


MINERVA_GENERATION_FORMAT = "task_format_opening_line_continuation"
MINERVA_BASE_VARIANT = "minerva_3b_base"
MINERVA_QLORA_VARIANT = "minerva_3b_qlora_best"
MINERVA_MAX_NEW_TOKENS = 900


def generate_minerva_continuation(
    *,
    model: Any,
    tokenizer: Any,
    opening_line: str,
    max_new_tokens: int,
    device: torch.device | str,
    seed: int,
    temperature: float = ACCEPTANCE_TEMPERATURE,
    top_k: int | None = ACCEPTANCE_TOP_K,
    continuation_line_target: int = TASK_CONTINUATION_LINE_TARGET,
    conditioning_prompt: str | None = None,
) -> dict[str, Any]:
    """Generate one visible continuation with cached causal decoding."""
    if not opening_line.strip() or "\n" in opening_line or "\r" in opening_line:
        raise ValueError("opening_line must contain exactly one non-empty line")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than 0")
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be greater than 0 when provided")
    if continuation_line_target <= 0:
        raise ValueError("continuation_line_target must be greater than 0")

    resolved_device = torch.device(device)
    visible_prompt = f"{opening_line}\n"
    prompt = (
        conditioning_prompt
        if conditioning_prompt is not None
        else visible_prompt
    )
    if not prompt.strip():
        raise ValueError("conditioning_prompt must not be empty")
    if not prompt.endswith(visible_prompt):
        raise ValueError(
            "conditioning_prompt must end with the exact visible opening line"
        )
    encoded = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )
    if not hasattr(encoded, "__getitem__"):
        raise ValueError("Minerva tokenizer must return indexable input_ids")
    input_ids = encoded["input_ids"].to(resolved_device)
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Minerva prompt input_ids must have shape (1, tokens)")

    special_token_ids = {
        token_id
        for token_id in getattr(tokenizer, "all_special_ids", [])
        if isinstance(token_id, int) and token_id >= 0
    }
    generator = torch.Generator(device=resolved_device).manual_seed(seed)
    attention_mask = torch.ones_like(input_ids)
    current_input_ids = input_ids
    past_key_values = None
    generated_ids: list[int] = []
    continuation_text = ""
    stop_reason = "max_new_tokens"

    model.eval()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            if (
                completed_non_empty_line_count(continuation_text)
                >= continuation_line_target
            ):
                stop_reason = "target_lines"
                break

            outputs = model(
                input_ids=current_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :].float() / temperature
            past_key_values = outputs.past_key_values
            if special_token_ids:
                logits[:, list(special_token_ids)] = -torch.inf
            if top_k is not None:
                retained_count = min(top_k, logits.shape[-1])
                threshold = torch.topk(logits, retained_count, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < threshold, -torch.inf)

            probabilities = torch.softmax(logits, dim=-1)
            if not torch.isfinite(probabilities).all():
                raise RuntimeError("Minerva generation produced invalid probabilities")
            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
                generator=generator,
            )
            next_token_id = int(next_token.item())
            generated_ids.append(next_token_id)
            continuation_text = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            current_input_ids = next_token
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token)],
                dim=1,
            )

    if (
        completed_non_empty_line_count(continuation_text)
        >= continuation_line_target
    ):
        stop_reason = "target_lines"

    return {
        "text": f"{opening_line}\n{continuation_text}",
        "opening_line": opening_line,
        "prompt": prompt,
        "conditioning_prompt": prompt,
        "stop_reason": stop_reason,
        "generated_new_tokens": len(generated_ids),
        "completed_continuation_lines": completed_non_empty_line_count(
            continuation_text
        ),
    }


def generate_minerva_variant_for_prompts(
    *,
    model: Any,
    tokenizer: Any,
    prompts: Sequence[dict[str, str]],
    output_dir: Path,
    model_variant: str,
    max_new_tokens: int,
    seeds: Sequence[int],
    device: torch.device | str,
    temperature: float = ACCEPTANCE_TEMPERATURE,
    top_k: int | None = ACCEPTANCE_TOP_K,
    continuation_line_target: int = TASK_CONTINUATION_LINE_TARGET,
    adapter_checkpoint_path: Path | None = None,
    prompt_config_path: Path | None = None,
    conditioning_prompt_builder: Callable[[str], str] | None = None,
    conditioning_format: str = "opening_line_newline",
    adapter_scale: float | None = None,
    adapter_epoch: int | None = None,
    model_id: str = MINERVA_3B_MODEL_ID,
    revision: str = MINERVA_3B_REVISION,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Generate and persist one Minerva variant's fixed prompt/seed set."""
    if not prompts:
        raise ValueError("prompts must not be empty")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []
    total_outputs = len(prompts) * len(seeds)
    started_at = time.monotonic()
    for prompt_index, prompt in enumerate(prompts):
        for seed_index, seed in enumerate(seeds):
            output_index = prompt_index * len(seeds) + seed_index + 1
            generation_id = f"{prompt['id']}__seed_{seed}"
            _report(
                progress,
                f"{model_variant} output {output_index}/{total_outputs}: "
                f"{generation_id}",
            )
            result = generate_minerva_continuation(
                model=model,
                tokenizer=tokenizer,
                opening_line=prompt["opening_line"],
                max_new_tokens=max_new_tokens,
                device=device,
                seed=seed,
                temperature=temperature,
                top_k=top_k,
                continuation_line_target=continuation_line_target,
                conditioning_prompt=(
                    conditioning_prompt_builder(prompt["opening_line"])
                    if conditioning_prompt_builder is not None
                    else None
                ),
            )
            output_path = output_dir / safe_prompt_filename(generation_id)
            output_path.write_text(result["text"], encoding="utf-8")
            generated_files.append({
                "prompt_id": generation_id,
                "source_prompt_id": prompt["id"],
                "poem_id": prompt["poem_id"],
                "author": prompt.get("author", ""),
                "prompt_text": prompt["opening_line"],
                "opening_line": prompt["opening_line"],
                "path": str(output_path),
                "seed": seed,
                "stop_reason": result["stop_reason"],
                "generated_new_tokens": result["generated_new_tokens"],
                "completed_continuation_lines": result[
                    "completed_continuation_lines"
                ],
                "conditioning_format": conditioning_format,
                "conditioning_prompt": result["conditioning_prompt"],
            })
            completed_outputs = len(generated_files)
            elapsed_seconds = time.monotonic() - started_at
            estimated_total_seconds = (
                elapsed_seconds / completed_outputs * total_outputs
            )
            _report(
                progress,
                f"{model_variant} wrote {completed_outputs}/{total_outputs} "
                f"elapsed={_format_duration(elapsed_seconds)} "
                f"eta={_format_duration(estimated_total_seconds - elapsed_seconds)}",
            )

    metadata = {
        "generation_format": MINERVA_GENERATION_FORMAT,
        "task_format_version": TASK_FORMAT_VERSION,
        "model_variant": model_variant,
        "model_id": model_id,
        "revision": revision,
        "adapter_checkpoint_path": (
            str(adapter_checkpoint_path) if adapter_checkpoint_path else None
        ),
        "adapter_scale": adapter_scale,
        "adapter_epoch": adapter_epoch,
        "conditioning_format": conditioning_format,
        "prompt_config_path": str(prompt_config_path) if prompt_config_path else None,
        "output_dir": str(output_dir),
        "max_new_tokens": max_new_tokens,
        "seeds": list(seeds),
        "device": str(device),
        "temperature": temperature,
        "top_k": top_k,
        "stop_text": getattr(tokenizer, "eos_token", None),
        "continuation_line_target": continuation_line_target,
        "total_line_target": continuation_line_target + 1,
        "suppressed_control_tokens": list(
            getattr(tokenizer, "all_special_tokens", [])
        ),
        "generated_files": generated_files,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def generate_fixed_minerva_comparison(
    *,
    repo_root: Path,
    adapter_checkpoint_path: Path,
    output_root: Path,
    prompts: Sequence[dict[str, str]],
    prompt_config_path: Path,
    max_new_tokens: int,
    device: torch.device | str,
    cache_dir: Path,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Load Minerva once and generate the fixed base and adapter comparison."""
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda":
        raise ValueError("fixed Minerva generation requires a CUDA device")
    if max_new_tokens != MINERVA_MAX_NEW_TOKENS:
        raise ValueError(
            "fixed Minerva generation requires max_new_tokens "
            f"{MINERVA_MAX_NEW_TOKENS}"
        )
    checkpoint_path = _resolve_path(repo_root, adapter_checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    _validate_adapter_checkpoint(checkpoint)
    dependencies = _load_dependencies()

    _report(progress, "loading Minerva tokenizer")
    tokenizer = dependencies["AutoTokenizer"].from_pretrained(
        MINERVA_3B_MODEL_ID,
        revision=MINERVA_3B_REVISION,
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _report(progress, "loading Minerva 3B in 4-bit NF4")
    quantization = dependencies["BitsAndBytesConfig"](
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = dependencies["AutoModelForCausalLM"].from_pretrained(
        MINERVA_3B_MODEL_ID,
        revision=MINERVA_3B_REVISION,
        cache_dir=cache_dir,
        quantization_config=quantization,
        torch_dtype=torch.float16,
        device_map={"": resolved_device.index or 0},
    )
    recipe = checkpoint["recipe_config"]
    model = dependencies["get_peft_model"](
        model,
        dependencies["LoraConfig"](
            task_type="CAUSAL_LM",
            r=recipe["lora_rank"],
            lora_alpha=recipe["lora_alpha"],
            lora_dropout=recipe["lora_dropout"],
            bias="none",
            target_modules=list(recipe["target_modules"]),
        ),
    )
    dependencies["set_peft_model_state_dict"](
        model,
        checkpoint["adapter_state_dict"],
    )
    model.eval()
    model.config.use_cache = True

    output_root.mkdir(parents=True, exist_ok=True)
    _report(progress, "generating untouched-base comparison outputs")
    with model.disable_adapter():
        base_metadata = generate_minerva_variant_for_prompts(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            output_dir=output_root / "base",
            model_variant=MINERVA_BASE_VARIANT,
            max_new_tokens=max_new_tokens,
            seeds=ACCEPTANCE_SEEDS,
            device=resolved_device,
            adapter_checkpoint_path=None,
            prompt_config_path=prompt_config_path,
            progress=progress,
        )

    _report(progress, "generating selected QLoRA-adapter outputs")
    qlora_metadata = generate_minerva_variant_for_prompts(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        output_dir=output_root / "qlora",
        model_variant=MINERVA_QLORA_VARIANT,
        max_new_tokens=max_new_tokens,
        seeds=ACCEPTANCE_SEEDS,
        device=resolved_device,
        adapter_checkpoint_path=checkpoint_path,
        prompt_config_path=prompt_config_path,
        progress=progress,
    )

    comparison_metadata = {
        "comparison": "minerva_3b_base_vs_qlora_best",
        "model_id": MINERVA_3B_MODEL_ID,
        "revision": MINERVA_3B_REVISION,
        "selected_adapter": {
            "path": str(checkpoint_path),
            "epoch": checkpoint["epoch"],
            "step": checkpoint["step"],
            "best_validation_row": checkpoint["best_validation_row"],
        },
        "base_output_dir": str(output_root / "base"),
        "qlora_output_dir": str(output_root / "qlora"),
        "base_output_count": len(base_metadata["generated_files"]),
        "qlora_output_count": len(qlora_metadata["generated_files"]),
    }
    (output_root / "comparison_metadata.json").write_text(
        json.dumps(comparison_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return comparison_metadata


def _validate_adapter_checkpoint(
    checkpoint: Any,
    *,
    require_selected: bool = True,
) -> None:
    if not isinstance(checkpoint, dict):
        raise ValueError("Minerva adapter checkpoint must contain a dictionary")
    if checkpoint.get("checkpoint_type") != "minerva_qlora_adapter":
        raise ValueError("checkpoint is not a Minerva QLoRA adapter")
    if checkpoint.get("model_id") != MINERVA_3B_MODEL_ID:
        raise ValueError("adapter checkpoint model_id does not match Minerva 3B")
    if checkpoint.get("revision") != MINERVA_3B_REVISION:
        raise ValueError("adapter checkpoint revision does not match")
    if not isinstance(checkpoint.get("adapter_state_dict"), dict):
        raise ValueError("adapter checkpoint is missing adapter_state_dict")
    best_row = checkpoint.get("best_validation_row")
    if not isinstance(best_row, dict):
        raise ValueError("adapter checkpoint is missing best_validation_row")
    if require_selected and (
        checkpoint.get("epoch") != best_row.get("epoch")
        or checkpoint.get("step") != best_row.get("step")
    ):
        raise ValueError("adapter checkpoint is not the selected best checkpoint")
    recipe = checkpoint.get("recipe_config")
    if not isinstance(recipe, dict):
        raise ValueError("adapter checkpoint is missing recipe_config")
    for field in ("lora_rank", "lora_alpha", "lora_dropout", "target_modules"):
        if field not in recipe:
            raise ValueError(f"adapter recipe is missing {field}")


def _load_dependencies() -> dict[str, Any]:
    try:
        from peft import (
            LoraConfig,
            get_peft_model,
            set_peft_model_state_dict,
        )
        from peft.helpers import rescale_adapter_scale
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
    except ImportError as error:
        raise RuntimeError(
            "Minerva generation dependencies are missing; use the project .venv"
        ) from error
    return {
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "set_peft_model_state_dict": set_peft_model_state_dict,
        "rescale_adapter_scale": rescale_adapter_scale,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
    }


def _resolve_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _report(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _format_duration(seconds: float) -> str:
    whole_seconds = max(0, round(seconds))
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, final_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{final_seconds:02d}s"
    if minutes:
        return f"{minutes}m{final_seconds:02d}s"
    return f"{final_seconds}s"
