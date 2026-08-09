"""Final-test evaluation for one validation-selected Minerva 7B adapter."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

from sonnet_corpus.task_format import build_sonnet_continuation_examples
from sonnet_evaluation.minerva_7b_sonnet_candidates import (
    build_sonnet_candidate_prompt,
    validate_candidate_checkpoint,
)
from sonnet_evaluation.minerva_generation import generate_minerva_variant_for_prompts
from sonnet_evaluation.task_acceptance import score_task_format_acceptance_directory
from sonnet_training.minerva_7b_qlora import (
    MINERVA_7B_INSTRUCT_MODEL_ID,
    MINERVA_7B_INSTRUCT_REVISION,
)
from sonnet_training.minerva_7b_sonnet_lora import (
    SONNET_RUN_VERSION,
    V6_MANIFEST_SHA256,
    evaluate_sonnet_loss,
    tokenize_sonnet_chat_example,
)


FINAL_EVALUATION_VERSION = "minerva_7b_v6_final_evaluation_v1"
SELECTION_VERSION = "minerva_7b_v6_validation_selection_v1"
FINAL_SEEDS = (1337, 1338)
FINAL_PROMPT_COUNT = 10
FINAL_MAX_NEW_TOKENS = 512


def validate_frozen_selection(
    *, selection: dict[str, Any], run_dir: Path, candidate_summary_path: Path
) -> tuple[int, Path]:
    """Require a validation-only frozen decision before exposing final-test data."""
    expected = {
        "selection_version": SELECTION_VERSION,
        "stage_b_run_version": SONNET_RUN_VERSION,
        "selection_frozen": True,
        "final_test_used": False,
        "candidate_summary_sha256": _sha256(candidate_summary_path),
    }
    for key, value in expected.items():
        if selection.get(key) != value:
            raise ValueError(f"frozen selection mismatch: {key}")
    epoch = selection.get("selected_epoch")
    if not isinstance(epoch, int) or epoch <= 0:
        raise ValueError("frozen selection requires a positive selected_epoch")
    checkpoint_path = run_dir / "checkpoints" / f"adapter_epoch_{epoch:02d}.pt"
    if selection.get("selected_checkpoint_sha256") != _sha256(checkpoint_path):
        raise ValueError("frozen selection checkpoint hash does not match")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    validate_candidate_checkpoint(checkpoint, expected_epoch=epoch)
    summary = json.loads(candidate_summary_path.read_text(encoding="utf-8"))
    selected_ids = {
        condition["epoch"] for condition in summary.get("conditions", [])
    }
    if epoch not in selected_ids:
        raise ValueError("frozen selection is not one of the reviewed candidates")
    return epoch, checkpoint_path


def evaluate_minerva_7b_sonnet_final(
    *,
    repo_root: Path,
    run_dir: Path,
    selection_path: Path,
    candidate_summary_path: Path,
    manifest_path: Path,
    output_dir: Path,
    prompts: Sequence[dict[str, str]],
    prompt_config_path: Path,
    device: torch.device | str,
    cache_dir: Path,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Measure final-test loss and generation only after selection is frozen."""
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda":
        raise ValueError("Minerva final evaluation requires CUDA")
    if len(prompts) != FINAL_PROMPT_COUNT:
        raise ValueError("Minerva final evaluation requires ten final-test prompts")
    run_dir = _resolve(repo_root, run_dir)
    selection_path = _resolve(repo_root, selection_path)
    candidate_summary_path = _resolve(repo_root, candidate_summary_path)
    manifest_path = _resolve(repo_root, manifest_path)
    if _sha256(manifest_path) != V6_MANIFEST_SHA256:
        raise ValueError("final evaluation requires the frozen V6 manifest")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    epoch, checkpoint_path = validate_frozen_selection(
        selection=selection,
        run_dir=run_dir,
        candidate_summary_path=candidate_summary_path,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    dependencies = _load_dependencies()

    _report(progress, "loading pinned Minerva tokenizer")
    tokenizer = dependencies["AutoTokenizer"].from_pretrained(
        MINERVA_7B_INSTRUCT_MODEL_ID,
        revision=MINERVA_7B_INSTRUCT_REVISION,
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _report(progress, "loading selected unquantized Minerva 7B adapter")
    model = dependencies["AutoModelForCausalLM"].from_pretrained(
        MINERVA_7B_INSTRUCT_MODEL_ID,
        revision=MINERVA_7B_INSTRUCT_REVISION,
        cache_dir=cache_dir,
        dtype=torch.float16,
        device_map={"": resolved_device.index or 0},
        low_cpu_mem_usage=True,
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
        model, checkpoint["adapter_state_dict"]
    )
    model.eval()

    # This is the first point at which final-test poem bodies are loaded.
    _report(progress, "tokenizing V6 final-test poems after frozen selection")
    test_examples = [
        tokenize_sonnet_chat_example(
            example=example,
            tokenizer=tokenizer,
            context_length=recipe["context_length"],
        )
        for example in build_sonnet_continuation_examples(
            manifest_path=manifest_path,
            repo_root=repo_root,
            dataset=recipe["dataset"],
            split="test",
        )
    ]
    test_loss = evaluate_sonnet_loss(
        model=model,
        examples=test_examples,
        pad_token_id=tokenizer.pad_token_id,
        device=resolved_device,
    )
    model.config.use_cache = True
    _report(progress, "generating frozen final-test prompts")
    output_dir = _resolve(repo_root, output_dir)
    generate_minerva_variant_for_prompts(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        output_dir=output_dir,
        model_variant=f"minerva_7b_v6_selected_epoch_{epoch:02d}",
        max_new_tokens=FINAL_MAX_NEW_TOKENS,
        seeds=FINAL_SEEDS,
        device=resolved_device,
        adapter_checkpoint_path=checkpoint_path,
        prompt_config_path=prompt_config_path,
        conditioning_prompt_builder=lambda opening: build_sonnet_candidate_prompt(
            tokenizer, opening
        ),
        conditioning_format="minerva_chat_complete_sonnet_v1",
        adapter_epoch=epoch,
        model_id=MINERVA_7B_INSTRUCT_MODEL_ID,
        revision=MINERVA_7B_INSTRUCT_REVISION,
        progress=progress,
    )
    controls = score_task_format_acceptance_directory(output_dir)
    result = {
        "evaluation_version": FINAL_EVALUATION_VERSION,
        "selection_path": str(selection_path),
        "selected_epoch": epoch,
        "selected_checkpoint_path": str(checkpoint_path),
        "selected_checkpoint_sha256": _sha256(checkpoint_path),
        "manifest_sha256": V6_MANIFEST_SHA256,
        "final_test_poem_count": len(test_examples),
        "final_test_loss": test_loss,
        "generation_output_count": len(controls),
        "controlled_sonnet_count": sum(
            row["automatic_control_pass"] for row in controls
        ),
        "final_test_used": True,
    }
    (output_dir / "final_evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _load_dependencies() -> dict[str, Any]:
    try:
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _report(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
