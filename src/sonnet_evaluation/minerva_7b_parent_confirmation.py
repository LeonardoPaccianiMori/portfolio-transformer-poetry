"""Validation-only parent-decoding confirmation for Minerva 7B."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from sonnet_evaluation.minerva_7b_sonnet_candidates import (
    build_sonnet_candidate_prompt,
    validate_candidate_checkpoint,
)
from sonnet_evaluation.minerva_generation import (
    _load_dependencies,
    generate_minerva_variant_for_prompts,
)
from sonnet_evaluation.minerva_judge_gate import sha256_file
from sonnet_evaluation.task_generation import (
    load_task_format_prompts,
    validate_task_format_prompts_against_manifest,
)
from sonnet_training.cuda_compat import (
    max_cuda_memory_allocated,
    max_cuda_memory_reserved,
    prepare_cuda_memory_measurement,
)
from sonnet_training.minerva_7b_qlora import (
    MINERVA_7B_INSTRUCT_MODEL_ID,
    MINERVA_7B_INSTRUCT_REVISION,
)


CONFIRMATION_VERSION = "minerva_7b_parent_decoding_confirmation_v1"
CONFIRMATION_STATUS = "predeclared_before_gpu_generation"
CONFIRMATION_PROMPT_COUNT = 24
CONFIRMATION_OUTPUT_COUNT = 72
CONFIRMATION_SEED = 4099
CONFIRMATION_MAX_NEW_TOKENS = 512
CONFIRMATION_CONTINUATION_LINES = 13
CONFIRMATION_PERIOD_COUNTS = {
    "XIII secolo": 6,
    "XIV secolo": 6,
    "XVI secolo": 8,
    "XVII secolo": 2,
    "XVIII secolo": 2,
}
CONFIRMATION_CONDITIONS = [
    {
        "condition_id": "untouched_default",
        "model_state": "untouched",
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
    },
    {
        "condition_id": "untouched_anti_repeat",
        "model_state": "untouched",
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.92,
        "repetition_penalty": 1.1,
    },
    {
        "condition_id": "stage_b_anti_repeat",
        "model_state": "stage_b",
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.92,
        "repetition_penalty": 1.1,
    },
]
CONFIRMATION_THRESHOLDS = {
    "controlled_form_min": 22,
    "grammar_min": 15,
    "topic_min": 12,
    "collapse_max": 2,
    "high_risk_memorization_max": 0,
}


def load_confirmation_config(path: Path) -> dict[str, Any]:
    """Load and validate the frozen confirmation configuration."""
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_confirmation_config(config)
    return config


def validate_confirmation_config(config: Mapping[str, Any]) -> None:
    """Reject any result-driven change to the approved confirmation."""
    expected = {
        "confirmation_version": CONFIRMATION_VERSION,
        "status": CONFIRMATION_STATUS,
        "model_id": MINERVA_7B_INSTRUCT_MODEL_ID,
        "revision": MINERVA_7B_INSTRUCT_REVISION,
        "manifest_path": "data/metadata/sonnets_expanded_v6_manifest.csv",
        "manifest_sha256": (
            "994c4c374f42ba26f1c352d7ad7c3adec7ec4671507770bd7c485cb6f977a4fa"
        ),
        "prompt_path": (
            "configs/minerva_7b_parent_decoding_confirmation_prompts.json"
        ),
        "prompt_sha256": (
            "98f429aeb04c4491517b3e1c218d21a98596476d163c87d62f3d09d535ea70e5"
        ),
        "prompt_count": CONFIRMATION_PROMPT_COUNT,
        "stage_b_checkpoint_path": (
            "runs/minerva_7b_v6_sonnet_fp16_lora_001/checkpoints/"
            "adapter_epoch_04.pt"
        ),
        "stage_b_checkpoint_sha256": (
            "aff3f2c4d193ce880ec9c7a6df6373f433001662c3ca78d7f915890733cb0df3"
        ),
        "stage_b_selection_path": "configs/minerva_7b_v6_selected_adapter.json",
        "stage_b_selection_sha256": (
            "4c9e7da2b94717a004c1a8c3bfc5f883e0889e73e442450272677dbddccea6da"
        ),
        "acceptance_thresholds": CONFIRMATION_THRESHOLDS,
        "final_test_allowed": False,
        "training_allowed": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"parent confirmation configuration mismatch: {key}")

    expected_exclusions = [
        {
            "path": "configs/minerva_3b_validation_sanity_prompts.json",
            "sha256": (
                "8d451030331c2ccd104e0ff9a4c24253f54fd27b3a956120192ebc28db5b62f7"
            ),
        },
        {
            "path": "configs/minerva_7b_quality_recovery_prompts.json",
            "sha256": (
                "25a6e70babf20a10722ca171fddb087c5052c4d54624017283ada864616d0856"
            ),
        },
        {
            "path": "configs/task_format_acceptance_prompts.json",
            "sha256": (
                "e27c74248870bbe72cce4f87cf6563480e42651e991a3135747ee7f13ced5117"
            ),
        },
    ]
    if config.get("excluded_prompt_sets") != expected_exclusions:
        raise ValueError("parent confirmation excluded prompt lock changed")

    generation = config.get("generation")
    expected_generation = {
        "load_mode": "nf4",
        "seed": CONFIRMATION_SEED,
        "max_new_tokens": CONFIRMATION_MAX_NEW_TOKENS,
        "continuation_line_target": CONFIRMATION_CONTINUATION_LINES,
        "conditioning_format": "minerva_chat_complete_sonnet_v1",
        "conditions": CONFIRMATION_CONDITIONS,
    }
    if not isinstance(generation, Mapping) or dict(generation) != expected_generation:
        raise ValueError("parent confirmation generation recipe changed")


def validate_confirmation_artifacts(
    *,
    config: Mapping[str, Any],
    repo_root: Path,
    require_local_checkpoint: bool = True,
) -> None:
    """Verify all public inputs and, when requested, the Stage B adapter."""
    artifacts = [
        (config, "manifest_path", "manifest_sha256", True),
        (config, "prompt_path", "prompt_sha256", True),
        (config, "stage_b_selection_path", "stage_b_selection_sha256", True),
        (
            config,
            "stage_b_checkpoint_path",
            "stage_b_checkpoint_sha256",
            require_local_checkpoint,
        ),
    ]
    for source, path_key, hash_key, required in artifacts:
        path = _resolve(repo_root, Path(str(source[path_key])))
        if not path.is_file():
            if required:
                raise FileNotFoundError(
                    f"parent confirmation artifact is missing: {path}"
                )
            continue
        if sha256_file(path) != source[hash_key]:
            raise ValueError(f"parent confirmation artifact hash mismatch: {path_key}")

    for excluded in config["excluded_prompt_sets"]:
        path = _resolve(repo_root, Path(str(excluded["path"])))
        if not path.is_file() or sha256_file(path) != excluded["sha256"]:
            raise ValueError("parent confirmation excluded prompt set changed")


def validate_confirmation_prompts(
    *, config: Mapping[str, Any], repo_root: Path
) -> list[dict[str, str]]:
    """Require exact fresh validation openings and the frozen author balance."""
    prompts = load_task_format_prompts(
        _resolve(repo_root, Path(str(config["prompt_path"])))
    )
    if len(prompts) != CONFIRMATION_PROMPT_COUNT:
        raise ValueError("parent confirmation requires exactly 24 prompts")
    validate_task_format_prompts_against_manifest(
        prompts=prompts,
        manifest_path=_resolve(repo_root, Path(str(config["manifest_path"]))),
        repo_root=repo_root,
        dataset="expanded_with_petrarch",
        split="validation",
    )
    author_counts = Counter(prompt["author"] for prompt in prompts)
    if len(author_counts) != 12 or set(author_counts.values()) != {2}:
        raise ValueError("parent confirmation requires two prompts from 12 authors")
    if Counter(prompt["period"] for prompt in prompts) != CONFIRMATION_PERIOD_COUNTS:
        raise ValueError("parent confirmation period balance changed")

    prompt_ids = {prompt["poem_id"] for prompt in prompts}
    for excluded in config["excluded_prompt_sets"]:
        prior = load_task_format_prompts(
            _resolve(repo_root, Path(str(excluded["path"])))
        )
        if prompt_ids & {prompt["poem_id"] for prompt in prior}:
            raise ValueError("parent confirmation overlaps a frozen prompt set")
    return prompts


def generate_parent_decoding_confirmation(
    *,
    repo_root: Path,
    config_path: Path,
    output_root: Path,
    device: torch.device | str,
    cache_dir: Path,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Generate all parent-confirmation conditions with one NF4 base load."""
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Minerva 7B parent confirmation requires CUDA")
    config_path = _resolve(repo_root, config_path)
    config = load_confirmation_config(config_path)
    validate_confirmation_artifacts(config=config, repo_root=repo_root)
    prompts = validate_confirmation_prompts(config=config, repo_root=repo_root)

    checkpoint_path = _resolve(
        repo_root, Path(str(config["stage_b_checkpoint_path"]))
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    validate_candidate_checkpoint(checkpoint, expected_epoch=4)
    selection = json.loads(
        _resolve(repo_root, Path(str(config["stage_b_selection_path"]))).read_text(
            encoding="utf-8"
        )
    )
    if (
        selection.get("selected_epoch") != 4
        or selection.get("selected_checkpoint_sha256")
        != config["stage_b_checkpoint_sha256"]
    ):
        raise ValueError("parent confirmation Stage B selection does not match")

    dependencies = _load_dependencies()
    cache_dir = _resolve(repo_root, cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    _report(progress, "loading pinned Minerva 7B tokenizer")
    tokenizer = dependencies["AutoTokenizer"].from_pretrained(
        config["model_id"],
        revision=config["revision"],
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    build_sonnet_candidate_prompt(tokenizer, prompts[0]["opening_line"])

    _report(progress, "loading untouched Minerva 7B base in 4-bit NF4")
    quantization = dependencies["BitsAndBytesConfig"](
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    prepare_cuda_memory_measurement(resolved_device)
    model = dependencies["AutoModelForCausalLM"].from_pretrained(
        config["model_id"],
        revision=config["revision"],
        cache_dir=cache_dir,
        dtype=torch.float16,
        device_map={"": resolved_device.index or 0},
        quantization_config=quantization,
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
    model.eval()
    model.config.use_cache = True

    output_root = _resolve(repo_root, output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    prompt_path = _resolve(repo_root, Path(str(config["prompt_path"])))
    summaries = []
    adapter_loaded = False
    for index, condition in enumerate(config["generation"]["conditions"], start=1):
        condition_id = condition["condition_id"]
        state = condition["model_state"]
        condition_checkpoint = checkpoint_path if state == "stage_b" else None
        output_dir = output_root / condition_id
        metadata = _load_complete_condition(
            output_dir=output_dir,
            condition=condition,
            prompts=prompts,
            checkpoint_path=condition_checkpoint,
            prompt_config_path=prompt_path,
            model_id=config["model_id"],
            revision=config["revision"],
            conditioning_format=config["generation"]["conditioning_format"],
        )
        if metadata is not None:
            _report(
                progress,
                f"reusing complete condition {condition_id} "
                f"({index}/{len(CONFIRMATION_CONDITIONS)})",
            )
        else:
            _report(
                progress,
                f"generating condition {condition_id} "
                f"({index}/{len(CONFIRMATION_CONDITIONS)})",
            )
            if state == "stage_b" and not adapter_loaded:
                dependencies["set_peft_model_state_dict"](
                    model, checkpoint["adapter_state_dict"]
                )
                adapter_loaded = True
            kwargs = dict(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                output_dir=output_dir,
                model_variant=f"minerva_7b_parent_confirmation_{condition_id}",
                max_new_tokens=CONFIRMATION_MAX_NEW_TOKENS,
                seeds=[CONFIRMATION_SEED],
                device=resolved_device,
                temperature=condition["temperature"],
                top_k=condition["top_k"],
                top_p=condition["top_p"],
                repetition_penalty=condition["repetition_penalty"],
                continuation_line_target=CONFIRMATION_CONTINUATION_LINES,
                adapter_checkpoint_path=condition_checkpoint,
                prompt_config_path=prompt_path,
                conditioning_prompt_builder=lambda opening: (
                    build_sonnet_candidate_prompt(tokenizer, opening)
                ),
                conditioning_format=config["generation"]["conditioning_format"],
                adapter_epoch=4 if state == "stage_b" else None,
                model_id=config["model_id"],
                revision=config["revision"],
                progress=progress,
            )
            if state == "untouched":
                with model.disable_adapter():
                    metadata = generate_minerva_variant_for_prompts(**kwargs)
            else:
                metadata = generate_minerva_variant_for_prompts(**kwargs)

        summaries.append({
            "condition_id": condition_id,
            "model_state": state,
            "decoder": {
                key: condition[key]
                for key in ("temperature", "top_k", "top_p", "repetition_penalty")
            },
            "output_dir": str(output_dir),
            "output_count": len(metadata["generated_files"]),
        })

    summary = {
        "confirmation_version": CONFIRMATION_VERSION,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "model_id": config["model_id"],
        "revision": config["revision"],
        "load_mode": "nf4",
        "prompt_count": len(prompts),
        "seed": CONFIRMATION_SEED,
        "condition_count": len(summaries),
        "output_count": sum(row["output_count"] for row in summaries),
        "conditions": summaries,
        "peak_cuda_allocated_mib": max_cuda_memory_allocated(resolved_device)
        / (1024**2),
        "peak_cuda_reserved_mib": max_cuda_memory_reserved(resolved_device)
        / (1024**2),
        "final_test_used": False,
        "training_used": False,
    }
    if summary["output_count"] != CONFIRMATION_OUTPUT_COUNT:
        raise ValueError("parent confirmation output count is incomplete")
    (output_root / "confirmation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _load_complete_condition(
    *,
    output_dir: Path,
    condition: Mapping[str, Any],
    prompts: Sequence[Mapping[str, str]],
    checkpoint_path: Path | None,
    prompt_config_path: Path,
    model_id: str,
    revision: str,
    conditioning_format: str,
) -> dict[str, Any] | None:
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "model_variant": (
            f"minerva_7b_parent_confirmation_{condition['condition_id']}"
        ),
        "model_id": model_id,
        "revision": revision,
        "adapter_checkpoint_path": (
            str(checkpoint_path) if checkpoint_path is not None else None
        ),
        "prompt_config_path": str(prompt_config_path),
        "conditioning_format": conditioning_format,
        "seeds": [CONFIRMATION_SEED],
        "max_new_tokens": CONFIRMATION_MAX_NEW_TOKENS,
        "temperature": condition["temperature"],
        "top_k": condition["top_k"],
        "top_p": condition["top_p"],
        "repetition_penalty": condition["repetition_penalty"],
        "continuation_line_target": CONFIRMATION_CONTINUATION_LINES,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"existing parent confirmation mismatch: {key}")
    generated = metadata.get("generated_files")
    if not isinstance(generated, list) or len(generated) != len(prompts):
        raise ValueError("existing parent confirmation condition is incomplete")
    if {row.get("source_prompt_id") for row in generated} != {
        prompt["id"] for prompt in prompts
    }:
        raise ValueError("existing parent confirmation prompt set changed")
    for row in generated:
        local_path = output_dir / Path(str(row["path"])).name
        if not local_path.is_file():
            raise FileNotFoundError(
                f"existing parent confirmation output is missing: {local_path}"
            )
    return metadata


def _resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _report(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
