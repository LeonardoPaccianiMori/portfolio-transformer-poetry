"""Frozen validation generation for the strongest Minerva 7B Stage B adapters."""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

from sonnet_evaluation.metrics import score_generation_directory
from sonnet_evaluation.minerva_generation import (
    generate_minerva_variant_for_prompts,
)
from sonnet_evaluation.task_acceptance import (
    score_task_format_acceptance_directory,
)
from sonnet_training.minerva_7b_qlora import (
    MINERVA_7B_INSTRUCT_MODEL_ID,
    MINERVA_7B_INSTRUCT_REVISION,
)
from sonnet_training.minerva_7b_sonnet_lora import (
    SELECTED_STAGE_A_SHA256,
    SONNET_RUN_VERSION,
    SONNET_TASK_FORMAT_VERSION,
    V6_MANIFEST_SHA256,
    build_sonnet_user_message,
)


CANDIDATE_GENERATION_VERSION = "minerva_7b_v6_candidate_generation_v1"
CANDIDATE_SEEDS = (4242,)
CANDIDATE_PROMPT_COUNT = 8
CANDIDATE_MAX_NEW_TOKENS = 512


def build_sonnet_candidate_prompt(tokenizer: Any, opening_line: str) -> str:
    """Render the training chat prompt and prefill the required first verse."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": build_sonnet_user_message(opening_line)}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("Minerva chat template must render a non-empty string")
    return f"{rendered}{opening_line}\n"


def validate_candidate_checkpoint(
    checkpoint: dict[str, Any], *, expected_epoch: int
) -> None:
    expected = {
        "checkpoint_type": "minerva_7b_v6_sonnet_lora_adapter",
        "run_version": SONNET_RUN_VERSION,
        "model_id": MINERVA_7B_INSTRUCT_MODEL_ID,
        "revision": MINERVA_7B_INSTRUCT_REVISION,
        "task_format_version": SONNET_TASK_FORMAT_VERSION,
        "selected_stage_a_sha256": SELECTED_STAGE_A_SHA256,
        "manifest_sha256": V6_MANIFEST_SHA256,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Stage B candidate checkpoint mismatch: {key}")
    row = checkpoint.get("row")
    if not isinstance(row, dict) or row.get("preservation_gate_passed") is not True:
        raise ValueError("Stage B candidate did not pass preservation")
    if row.get("epoch") != expected_epoch:
        raise ValueError("Stage B candidate epoch does not match result metadata")


def load_candidate_rows(run_dir: Path) -> list[dict[str, Any]]:
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    if result.get("run_version") != SONNET_RUN_VERSION:
        raise ValueError("Stage B result version does not match")
    if result.get("final_test_used") is not False:
        raise ValueError("Stage B candidate selection must precede final-test use")
    rows = result.get("top_qualifying_candidates")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("Stage B result must retain exactly three candidates")
    return rows


def generate_minerva_7b_sonnet_candidates(
    *,
    repo_root: Path,
    run_dir: Path,
    output_root: Path,
    prompts: Sequence[dict[str, str]],
    prompt_config_path: Path,
    device: torch.device | str,
    cache_dir: Path,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Generate all three validation candidates with one unquantized model load."""
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda":
        raise ValueError("Minerva 7B candidate generation requires CUDA")
    if len(prompts) != CANDIDATE_PROMPT_COUNT:
        raise ValueError("candidate generation requires eight validation prompts")
    run_dir = _resolve(repo_root, run_dir)
    rows = load_candidate_rows(run_dir)
    dependencies = _load_dependencies()

    _report(progress, "loading pinned Minerva tokenizer")
    tokenizer = dependencies["AutoTokenizer"].from_pretrained(
        MINERVA_7B_INSTRUCT_MODEL_ID,
        revision=MINERVA_7B_INSTRUCT_REVISION,
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _report(progress, "loading unquantized Minerva 7B weights in FP16")
    model = dependencies["AutoModelForCausalLM"].from_pretrained(
        MINERVA_7B_INSTRUCT_MODEL_ID,
        revision=MINERVA_7B_INSTRUCT_REVISION,
        cache_dir=cache_dir,
        dtype=torch.float16,
        device_map={"": resolved_device.index or 0},
        low_cpu_mem_usage=True,
    )
    first_checkpoint_path = (
        run_dir / "checkpoints" / f"adapter_epoch_{int(rows[0]['epoch']):02d}.pt"
    )
    first_checkpoint = torch.load(
        first_checkpoint_path, map_location="cpu", weights_only=True
    )
    validate_candidate_checkpoint(
        first_checkpoint, expected_epoch=int(rows[0]["epoch"])
    )
    recipe = first_checkpoint["recipe_config"]
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
    model.config.use_cache = True
    model.eval()

    output_root = _resolve(repo_root, output_root)
    conditions = []
    for candidate_index, row in enumerate(rows, start=1):
        epoch = int(row["epoch"])
        checkpoint_path = (
            run_dir / "checkpoints" / f"adapter_epoch_{epoch:02d}.pt"
        )
        checkpoint = (
            first_checkpoint
            if checkpoint_path == first_checkpoint_path
            else torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        )
        validate_candidate_checkpoint(checkpoint, expected_epoch=epoch)
        dependencies["set_peft_model_state_dict"](
            model, checkpoint["adapter_state_dict"]
        )
        condition_id = f"candidate_{candidate_index}_epoch_{epoch:02d}"
        _report(
            progress,
            f"generating {condition_id} ({candidate_index}/{len(rows)})",
        )
        output_dir = output_root / condition_id
        generate_minerva_variant_for_prompts(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            output_dir=output_dir,
            model_variant=f"minerva_7b_v6_epoch_{epoch:02d}",
            max_new_tokens=CANDIDATE_MAX_NEW_TOKENS,
            seeds=CANDIDATE_SEEDS,
            device=resolved_device,
            adapter_checkpoint_path=checkpoint_path,
            prompt_config_path=prompt_config_path,
            conditioning_prompt_builder=lambda opening: build_sonnet_candidate_prompt(
                tokenizer, opening
            ),
            conditioning_format=SONNET_TASK_FORMAT_VERSION,
            adapter_epoch=epoch,
            model_id=MINERVA_7B_INSTRUCT_MODEL_ID,
            revision=MINERVA_7B_INSTRUCT_REVISION,
            progress=progress,
        )
        metric_rows = score_generation_directory(output_dir)
        control_rows = score_task_format_acceptance_directory(output_dir)
        conditions.append({
            "condition_id": condition_id,
            "epoch": epoch,
            "checkpoint_path": str(checkpoint_path),
            "validation_loss": row["validation_loss"],
            "modern_validation_loss": row["modern_validation_loss"],
            "instruction_validation_loss": row["instruction_validation_loss"],
            "output_dir": str(output_dir),
            "controlled_sonnet_count": sum(
                result["automatic_control_pass"] for result in control_rows
            ),
            "mean_repetition_ratio": statistics.fmean(
                result["repetition_ratio"] for result in metric_rows
            ),
            "output_count": len(metric_rows),
        })

    metadata = {
        "generation_version": CANDIDATE_GENERATION_VERSION,
        "stage_b_run_dir": str(run_dir),
        "prompt_config_path": str(prompt_config_path),
        "prompt_count": len(prompts),
        "seeds": list(CANDIDATE_SEEDS),
        "conditions": conditions,
        "final_test_used": False,
        "selection_frozen": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "candidate_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def _load_dependencies() -> dict[str, Any]:
    try:
        from peft import (
            LoraConfig,
            get_peft_model,
            set_peft_model_state_dict,
        )
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Minerva generation dependencies are missing") from error
    return {
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "set_peft_model_state_dict": set_peft_model_state_dict,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
    }


def _resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _report(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
