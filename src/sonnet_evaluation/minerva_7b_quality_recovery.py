"""Validation-only lineage and decoding diagnostic for Minerva 7B."""

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


RECOVERY_VERSION = "minerva_7b_quality_recovery_v1"
RECOVERY_STATUS = "predeclared_before_gpu_generation"
RECOVERY_PROMPT_COUNT = 12
RECOVERY_SEED = 2029
RECOVERY_MAX_NEW_TOKENS = 512
RECOVERY_CONTINUATION_LINES = 13
RECOVERY_OUTPUT_COUNT = 84
RECOVERY_PERIOD_COUNTS = {
    "XIII secolo": 3,
    "XIV secolo": 3,
    "XVI secolo": 4,
    "XVII secolo": 1,
    "XVIII secolo": 1,
}
RECOVERY_CONDITIONS = [
    {
        "condition_id": "untouched_control",
        "model_state": "untouched",
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
    },
    {
        "condition_id": "stage_a_control",
        "model_state": "stage_a",
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
    },
    {
        "condition_id": "stage_b_control",
        "model_state": "stage_b",
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
    },
    {
        "condition_id": "stage_b_conservative",
        "model_state": "stage_b",
        "temperature": 0.65,
        "top_k": 40,
        "top_p": 0.92,
        "repetition_penalty": 1.05,
    },
    {
        "condition_id": "stage_b_low_temperature",
        "model_state": "stage_b",
        "temperature": 0.55,
        "top_k": 30,
        "top_p": 0.9,
        "repetition_penalty": 1.05,
    },
    {
        "condition_id": "stage_b_anti_repeat",
        "model_state": "stage_b",
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.92,
        "repetition_penalty": 1.1,
    },
    {
        "condition_id": "stage_b_nucleus",
        "model_state": "stage_b",
        "temperature": 0.7,
        "top_k": None,
        "top_p": 0.9,
        "repetition_penalty": 1.05,
    },
]


def load_recovery_config(path: Path) -> dict[str, Any]:
    """Load and validate the predeclared recovery configuration."""
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_recovery_config(config)
    return config


def validate_recovery_config(config: Mapping[str, Any]) -> None:
    """Reject result-driven changes to the approved diagnostic."""
    expected = {
        "recovery_version": RECOVERY_VERSION,
        "status": RECOVERY_STATUS,
        "model_id": MINERVA_7B_INSTRUCT_MODEL_ID,
        "revision": MINERVA_7B_INSTRUCT_REVISION,
        "manifest_path": "data/metadata/sonnets_expanded_v6_manifest.csv",
        "manifest_sha256": "994c4c374f42ba26f1c352d7ad7c3adec7ec4671507770bd7c485cb6f977a4fa",
        "prompt_path": "configs/minerva_7b_quality_recovery_prompts.json",
        "prompt_sha256": "25a6e70babf20a10722ca171fddb087c5052c4d54624017283ada864616d0856",
        "prompt_count": RECOVERY_PROMPT_COUNT,
        "stage_a_checkpoint_path": (
            "runs/minerva_7b_historical_fp16_lora_001/checkpoints/"
            "adapter_step_004000.pt"
        ),
        "stage_a_checkpoint_sha256": (
            "acfad4d442ac8ea7349dcb1bd379c9b41859027ab45daac54c6b6aa35e0bbc63"
        ),
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
        "final_test_allowed": False,
        "training_allowed": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"quality-recovery configuration mismatch: {key}")
    if config.get("excluded_prompt_sets") != [
        {
            "path": "configs/minerva_3b_validation_sanity_prompts.json",
            "sha256": "8d451030331c2ccd104e0ff9a4c24253f54fd27b3a956120192ebc28db5b62f7",
        },
        {
            "path": "configs/task_format_acceptance_prompts.json",
            "sha256": "e27c74248870bbe72cce4f87cf6563480e42651e991a3135747ee7f13ced5117",
        },
    ]:
        raise ValueError("quality-recovery excluded prompt lock changed")

    generation = config.get("generation")
    if not isinstance(generation, Mapping):
        raise ValueError("quality-recovery generation configuration is missing")
    generation_expected = {
        "load_mode": "nf4",
        "seed": RECOVERY_SEED,
        "max_new_tokens": RECOVERY_MAX_NEW_TOKENS,
        "continuation_line_target": RECOVERY_CONTINUATION_LINES,
        "conditioning_format": "minerva_chat_complete_sonnet_v1",
        "conditions": RECOVERY_CONDITIONS,
    }
    if dict(generation) != generation_expected:
        raise ValueError("quality-recovery generation recipe does not match")

    judge = config.get("judge")
    if not isinstance(judge, Mapping):
        raise ValueError("quality-recovery judge configuration is missing")
    expected_judge_fields = {
        "judge_version": "minerva_7b_instruction_judge_v1",
        "load_mode": "nf4",
        "score_range": [0, 4],
        "max_new_tokens": 96,
        "do_sample": False,
        "human_case_count": 56,
        "human_mapping_path": (
            "outputs/generations/minerva_3b_validation_sanity_v1/"
            "blind_mapping.json"
        ),
        "human_mapping_sha256": (
            "56d9435b92dcfe64ffa09861efd63a89d2579479ceeca3cbc275ee52917f1e40"
        ),
        "human_judgments_path": (
            "outputs/reports/minerva_3b_validation_sanity_blinded_judgments.md"
        ),
        "human_judgments_sha256": (
            "fca814d8a16f5d67375b8f67b24b53d969e3c7ae896eb004f4c68306c7eeda3e"
        ),
        "thresholds": {
            "parse_rate": 0.98,
            "grammar_auroc": 0.75,
            "topic_auroc": 0.7,
            "noncollapse_auroc": 0.75,
            "human_ordinal_pairwise_concordance": 0.65,
        },
        "remote_fp16_authorization": {
            "required_checks": ["parse_rate", "noncollapse_auroc"],
            "minimum_total_passed_checks": 4,
        },
    }
    for key, value in expected_judge_fields.items():
        if judge.get(key) != value:
            raise ValueError(f"quality-recovery judge mismatch: {key}")


def validate_recovery_artifacts(
    *,
    config: Mapping[str, Any],
    repo_root: Path,
    require_local_checkpoints: bool = True,
) -> None:
    """Verify frozen public inputs and, when requested, local adapters."""
    path_hash_fields = [
        ("manifest_path", "manifest_sha256", True),
        ("prompt_path", "prompt_sha256", True),
        ("stage_b_selection_path", "stage_b_selection_sha256", True),
        (
            "stage_a_checkpoint_path",
            "stage_a_checkpoint_sha256",
            require_local_checkpoints,
        ),
        (
            "stage_b_checkpoint_path",
            "stage_b_checkpoint_sha256",
            require_local_checkpoints,
        ),
    ]
    judge = config["judge"]
    path_hash_fields.extend([
        ("human_mapping_path", "human_mapping_sha256", True, judge),
        ("human_judgments_path", "human_judgments_sha256", True, judge),
    ])
    for entry in path_hash_fields:
        path_key, hash_key, required = entry[:3]
        source = entry[3] if len(entry) == 4 else config
        path = _resolve(repo_root, Path(str(source[path_key])))
        if not path.is_file():
            if required:
                raise FileNotFoundError(f"quality-recovery artifact is missing: {path}")
            continue
        if sha256_file(path) != source[hash_key]:
            raise ValueError(f"quality-recovery artifact hash mismatch: {path_key}")

    for excluded in config.get("excluded_prompt_sets", []):
        path = _resolve(repo_root, Path(str(excluded["path"])))
        if not path.is_file() or sha256_file(path) != excluded["sha256"]:
            raise ValueError("quality-recovery excluded prompt set changed")


def validate_recovery_prompts(
    *, config: Mapping[str, Any], repo_root: Path
) -> list[dict[str, str]]:
    """Require exact validation openings and disjoint prior/final prompt sets."""
    prompt_path = _resolve(repo_root, Path(str(config["prompt_path"])))
    manifest_path = _resolve(repo_root, Path(str(config["manifest_path"])))
    prompts = load_task_format_prompts(prompt_path)
    if len(prompts) != RECOVERY_PROMPT_COUNT:
        raise ValueError("quality recovery requires exactly twelve prompts")
    validate_task_format_prompts_against_manifest(
        prompts=prompts,
        manifest_path=manifest_path,
        repo_root=repo_root,
        dataset="expanded_with_petrarch",
        split="validation",
    )
    if len({prompt.get("author") for prompt in prompts}) != RECOVERY_PROMPT_COUNT:
        raise ValueError("quality-recovery prompts must use twelve distinct authors")
    if Counter(prompt.get("period") for prompt in prompts) != RECOVERY_PERIOD_COUNTS:
        raise ValueError("quality-recovery prompt period balance changed")

    recovery_poem_ids = {prompt["poem_id"] for prompt in prompts}
    for excluded in config["excluded_prompt_sets"]:
        excluded_prompts = load_task_format_prompts(
            _resolve(repo_root, Path(str(excluded["path"])))
        )
        if recovery_poem_ids & {
            prompt["poem_id"] for prompt in excluded_prompts
        }:
            raise ValueError("quality-recovery prompts overlap a frozen prompt set")
    return prompts


def validate_stage_a_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    """Require the qualifying historical step-4,000 adapter."""
    expected = {
        "checkpoint_type": "minerva_7b_historical_lora_adapter",
        "model_id": MINERVA_7B_INSTRUCT_MODEL_ID,
        "revision": MINERVA_7B_INSTRUCT_REVISION,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"quality-recovery Stage A mismatch: {key}")
    row = checkpoint.get("row")
    if not isinstance(row, Mapping):
        raise ValueError("quality-recovery Stage A row is missing")
    if row.get("preservation_gate_passed") is not True or row.get("step") != 4000:
        raise ValueError("quality recovery requires qualifying Stage A step 4000")


def generate_quality_recovery(
    *,
    repo_root: Path,
    config_path: Path,
    output_root: Path,
    device: torch.device | str,
    cache_dir: Path,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Generate all frozen lineage and decoding conditions with one base load."""
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Minerva 7B quality recovery requires CUDA")
    config_path = _resolve(repo_root, config_path)
    config = load_recovery_config(config_path)
    validate_recovery_artifacts(config=config, repo_root=repo_root)
    prompts = validate_recovery_prompts(config=config, repo_root=repo_root)

    stage_a_path = _resolve(repo_root, Path(config["stage_a_checkpoint_path"]))
    stage_b_path = _resolve(repo_root, Path(config["stage_b_checkpoint_path"]))
    stage_a = torch.load(stage_a_path, map_location="cpu", weights_only=True)
    stage_b = torch.load(stage_b_path, map_location="cpu", weights_only=True)
    validate_stage_a_checkpoint(stage_a)
    validate_candidate_checkpoint(stage_b, expected_epoch=4)
    _require_compatible_adapter_recipes(stage_a, stage_b)

    selection = json.loads(
        _resolve(repo_root, Path(config["stage_b_selection_path"])).read_text(
            encoding="utf-8"
        )
    )
    if (
        selection.get("selected_epoch") != 4
        or selection.get("selected_checkpoint_sha256")
        != config["stage_b_checkpoint_sha256"]
    ):
        raise ValueError("quality-recovery Stage B selection does not match")

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
    recipe = stage_a["recipe_config"]
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
    condition_summaries = []
    loaded_state: str | None = None
    for condition_index, condition in enumerate(
        config["generation"]["conditions"], start=1
    ):
        condition_id = condition["condition_id"]
        state = condition["model_state"]
        checkpoint_for_state = (
            None
            if state == "untouched"
            else stage_a_path if state == "stage_a" else stage_b_path
        )
        output_dir = output_root / condition_id
        existing = _load_complete_condition(
            output_dir=output_dir,
            condition=condition,
            prompts=prompts,
            checkpoint_path=checkpoint_for_state,
            prompt_config_path=_resolve(repo_root, Path(config["prompt_path"])),
            model_id=config["model_id"],
            revision=config["revision"],
            conditioning_format=config["generation"]["conditioning_format"],
        )
        if existing is not None:
            _report(
                progress,
                f"reusing complete condition {condition_id} "
                f"({condition_index}/{len(RECOVERY_CONDITIONS)})",
            )
            metadata = existing
        else:
            _report(
                progress,
                f"generating condition {condition_id} "
                f"({condition_index}/{len(RECOVERY_CONDITIONS)})",
            )
            if state == "stage_a" and loaded_state != state:
                dependencies["set_peft_model_state_dict"](
                    model, stage_a["adapter_state_dict"]
                )
                loaded_state = state
            elif state == "stage_b" and loaded_state != state:
                dependencies["set_peft_model_state_dict"](
                    model, stage_b["adapter_state_dict"]
                )
                loaded_state = state

            generate_kwargs = dict(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                output_dir=output_dir,
                model_variant=f"minerva_7b_recovery_{condition_id}",
                max_new_tokens=config["generation"]["max_new_tokens"],
                seeds=[config["generation"]["seed"]],
                device=resolved_device,
                temperature=condition["temperature"],
                top_k=condition["top_k"],
                top_p=condition["top_p"],
                repetition_penalty=condition["repetition_penalty"],
                continuation_line_target=config["generation"][
                    "continuation_line_target"
                ],
                adapter_checkpoint_path=checkpoint_for_state,
                prompt_config_path=_resolve(repo_root, Path(config["prompt_path"])),
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
                    metadata = generate_minerva_variant_for_prompts(
                        **generate_kwargs
                    )
            else:
                metadata = generate_minerva_variant_for_prompts(**generate_kwargs)

        condition_summaries.append({
            "condition_id": condition_id,
            "model_state": condition["model_state"],
            "decoder": {
                key: condition[key]
                for key in (
                    "temperature",
                    "top_k",
                    "top_p",
                    "repetition_penalty",
                )
            },
            "output_dir": str(output_dir),
            "output_count": len(metadata["generated_files"]),
        })

    summary = {
        "recovery_version": RECOVERY_VERSION,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "model_id": config["model_id"],
        "revision": config["revision"],
        "load_mode": "nf4",
        "prompt_count": len(prompts),
        "seed": RECOVERY_SEED,
        "condition_count": len(condition_summaries),
        "output_count": sum(row["output_count"] for row in condition_summaries),
        "conditions": condition_summaries,
        "peak_cuda_allocated_mib": max_cuda_memory_allocated(resolved_device)
        / (1024**2),
        "peak_cuda_reserved_mib": max_cuda_memory_reserved(resolved_device)
        / (1024**2),
        "final_test_used": False,
        "training_used": False,
    }
    if summary["output_count"] != RECOVERY_OUTPUT_COUNT:
        raise ValueError("quality-recovery output count is incomplete")
    (output_root / "recovery_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _require_compatible_adapter_recipes(
    stage_a: Mapping[str, Any], stage_b: Mapping[str, Any]
) -> None:
    keys = ("lora_rank", "lora_alpha", "lora_dropout", "target_modules")
    left = stage_a.get("recipe_config")
    right = stage_b.get("recipe_config")
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise ValueError("quality-recovery adapter recipe is missing")
    if any(left.get(key) != right.get(key) for key in keys):
        raise ValueError("Stage A and Stage B adapter recipes are incompatible")


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
        "model_variant": f"minerva_7b_recovery_{condition['condition_id']}",
        "model_id": model_id,
        "revision": revision,
        "adapter_checkpoint_path": (
            str(checkpoint_path) if checkpoint_path is not None else None
        ),
        "prompt_config_path": str(prompt_config_path),
        "conditioning_format": conditioning_format,
        "seeds": [RECOVERY_SEED],
        "max_new_tokens": RECOVERY_MAX_NEW_TOKENS,
        "temperature": condition["temperature"],
        "top_k": condition["top_k"],
        "top_p": condition["top_p"],
        "repetition_penalty": condition["repetition_penalty"],
        "continuation_line_target": RECOVERY_CONTINUATION_LINES,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"existing recovery condition mismatch: {key}")
    generated = metadata.get("generated_files")
    if not isinstance(generated, list) or len(generated) != len(prompts):
        raise ValueError("existing recovery condition is incomplete")
    expected_ids = {prompt["id"] for prompt in prompts}
    if {row.get("source_prompt_id") for row in generated} != expected_ids:
        raise ValueError("existing recovery condition prompt set changed")
    for row in generated:
        local_path = output_dir / Path(str(row["path"])).name
        if not local_path.is_file():
            raise FileNotFoundError(f"existing recovery output is missing: {local_path}")
    return metadata


def _resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _report(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
