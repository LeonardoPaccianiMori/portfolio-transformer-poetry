"""Validation-only diagnosis of Minerva prompting and QLoRA adapter strength."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

from sonnet_evaluation.metrics import score_generation_directory
from sonnet_evaluation.minerva_generation import (
    _load_dependencies,
    _validate_adapter_checkpoint,
    generate_minerva_variant_for_prompts,
)
from sonnet_evaluation.qualitative import (
    fenced_text_block,
    load_generated_reviews,
)
from sonnet_evaluation.task_acceptance import (
    score_task_format_acceptance_directory,
)
from sonnet_training.minerva_qlora import (
    MINERVA_3B_MODEL_ID,
    MINERVA_3B_REVISION,
)


MINERVA_SANITY_AUDIT_VERSION = "minerva_3b_validation_sanity_v1"
MINERVA_SANITY_PROMPT_COUNT = 8
MINERVA_SANITY_SEEDS = (4242,)
MINERVA_SANITY_MAX_NEW_TOKENS = 512
MINERVA_SANITY_ADAPTER_SCALES = (0.25, 0.5, 0.75, 1.0)
MINERVA_SANITY_CONDITION_IDS = (
    "base_raw",
    "base_instructed",
    "best_scale_025",
    "best_scale_050",
    "best_scale_075",
    "best_scale_100",
    "final_scale_100",
)
MINERVA_SANITY_SELECTABLE_CONDITIONS = (
    "best_scale_025",
    "best_scale_050",
    "best_scale_075",
    "best_scale_100",
)
MINERVA_SANITY_REQUIRED_PERIODS = {
    "XIII secolo",
    "XIV secolo",
    "XVI secolo",
    "XVII secolo",
    "XVIII secolo",
}


def build_minerva_sonnet_instruction_prompt(opening_line: str) -> str:
    """Condition Base explicitly while retaining the exact visible prefix."""
    if not opening_line.strip():
        raise ValueError("opening_line must not be empty")
    if "\n" in opening_line or "\r" in opening_line:
        raise ValueError("opening_line must contain exactly one line")
    return (
        "Componi un sonetto in italiano classico di esattamente quattordici "
        "versi. Mantieni un tema coerente, usa una sintassi corretta ed evita "
        "ripetizioni. Continua dal primo verso riportato qui sotto.\n\n"
        f"Sonetto:\n{opening_line}\n"
    )


def validate_minerva_sanity_prompts(
    prompts: Sequence[dict[str, str]],
    final_test_prompts: Sequence[dict[str, str]],
) -> None:
    """Lock the validation prompt count, coverage, and final-test isolation."""
    if len(prompts) != MINERVA_SANITY_PROMPT_COUNT:
        raise ValueError(
            "Minerva sanity audit requires exactly "
            f"{MINERVA_SANITY_PROMPT_COUNT} validation prompts"
        )
    poem_ids = [prompt.get("poem_id", "") for prompt in prompts]
    if (
        any(not poem_id for poem_id in poem_ids)
        or len(set(poem_ids)) != len(poem_ids)
    ):
        raise ValueError("Minerva sanity prompt poem_ids must be non-empty and unique")

    authors = [prompt.get("author", "") for prompt in prompts]
    if any(not author for author in authors) or len(set(authors)) != len(authors):
        raise ValueError("Minerva sanity prompts must cover eight distinct authors")

    periods = {prompt.get("period", "") for prompt in prompts}
    if not MINERVA_SANITY_REQUIRED_PERIODS.issubset(periods):
        raise ValueError("Minerva sanity prompts do not cover all required periods")

    final_test_ids = {prompt.get("poem_id", "") for prompt in final_test_prompts}
    overlap = sorted(set(poem_ids) & final_test_ids)
    if overlap:
        raise ValueError(
            "Minerva sanity prompts overlap the fixed final test: "
            + ", ".join(overlap)
        )


def validate_minerva_sanity_checkpoints(
    best_checkpoint: dict[str, Any],
    final_checkpoint: dict[str, Any],
) -> None:
    """Require the recorded epoch-3 selection and epoch-6 overfitting contrast."""
    _validate_adapter_checkpoint(best_checkpoint)
    _validate_adapter_checkpoint(final_checkpoint, require_selected=False)
    if best_checkpoint.get("epoch") != 3 or best_checkpoint.get("step") != 558:
        raise ValueError("sanity audit best adapter must be epoch 3, step 558")
    if final_checkpoint.get("epoch") != 6 or final_checkpoint.get("step") != 1116:
        raise ValueError("sanity audit final adapter must be epoch 6, step 1116")
    for field in ("recipe_config", "manifest_sha256"):
        if best_checkpoint.get(field) != final_checkpoint.get(field):
            raise ValueError(f"sanity audit checkpoint {field} values do not match")


def minerva_sanity_conditions(
    *,
    best_checkpoint_path: Path,
    final_checkpoint_path: Path,
) -> list[dict[str, Any]]:
    """Return the frozen diagnostic conditions in execution order."""
    conditions = [
        {
            "condition_id": "base_raw",
            "checkpoint_kind": "base",
            "checkpoint_path": None,
            "adapter_epoch": None,
            "adapter_scale": 0.0,
            "conditioning_format": "opening_line_newline",
            "selectable": False,
        },
        {
            "condition_id": "base_instructed",
            "checkpoint_kind": "base",
            "checkpoint_path": None,
            "adapter_epoch": None,
            "adapter_scale": 0.0,
            "conditioning_format": "explicit_italian_sonnet_instruction_v1",
            "selectable": False,
        },
    ]
    conditions.extend(
        {
            "condition_id": f"best_scale_{round(scale * 100):03d}",
            "checkpoint_kind": "best",
            "checkpoint_path": str(best_checkpoint_path),
            "adapter_epoch": 3,
            "adapter_scale": scale,
            "conditioning_format": "opening_line_newline",
            "selectable": True,
        }
        for scale in MINERVA_SANITY_ADAPTER_SCALES
    )
    conditions.append({
        "condition_id": "final_scale_100",
        "checkpoint_kind": "final",
        "checkpoint_path": str(final_checkpoint_path),
        "adapter_epoch": 6,
        "adapter_scale": 1.0,
        "conditioning_format": "opening_line_newline",
        "selectable": False,
    })
    if (
        tuple(condition["condition_id"] for condition in conditions)
        != MINERVA_SANITY_CONDITION_IDS
    ):
        raise AssertionError("Minerva sanity condition construction drifted")
    return conditions


def generate_minerva_sanity_audit(
    *,
    repo_root: Path,
    best_checkpoint_path: Path,
    final_checkpoint_path: Path,
    output_root: Path,
    prompts: Sequence[dict[str, str]],
    prompt_config_path: Path,
    device: torch.device | str,
    cache_dir: Path,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Generate the frozen validation-only conditions with one model load."""
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda":
        raise ValueError("Minerva sanity generation requires a CUDA device")
    if len(prompts) != MINERVA_SANITY_PROMPT_COUNT:
        raise ValueError("Minerva sanity generation received the wrong prompt count")

    best_path = _resolve_path(repo_root, best_checkpoint_path)
    final_path = _resolve_path(repo_root, final_checkpoint_path)
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    final_checkpoint = torch.load(final_path, map_location="cpu", weights_only=True)
    validate_minerva_sanity_checkpoints(best_checkpoint, final_checkpoint)
    conditions = minerva_sanity_conditions(
        best_checkpoint_path=best_path,
        final_checkpoint_path=final_path,
    )
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
    recipe = best_checkpoint["recipe_config"]
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
        best_checkpoint["adapter_state_dict"],
    )
    model.eval()
    model.config.use_cache = True
    output_root.mkdir(parents=True, exist_ok=True)

    condition_results: list[dict[str, Any]] = []
    for condition in conditions[:2]:
        with model.disable_adapter():
            condition_results.append(
                _generate_or_reuse_condition(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    output_root=output_root,
                    condition=condition,
                    device=resolved_device,
                    prompt_config_path=prompt_config_path,
                    conditioning_prompt_builder=(
                        build_minerva_sonnet_instruction_prompt
                        if condition["condition_id"] == "base_instructed"
                        else None
                    ),
                    progress=progress,
                )
            )

    for condition in conditions[2:6]:
        with dependencies["rescale_adapter_scale"](
            model,
            condition["adapter_scale"],
        ):
            condition_results.append(
                _generate_or_reuse_condition(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    output_root=output_root,
                    condition=condition,
                    device=resolved_device,
                    prompt_config_path=prompt_config_path,
                    conditioning_prompt_builder=None,
                    progress=progress,
                )
            )

    _report(progress, "loading epoch-6 final adapter contrast")
    dependencies["set_peft_model_state_dict"](
        model,
        final_checkpoint["adapter_state_dict"],
    )
    condition_results.append(
        _generate_or_reuse_condition(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            output_root=output_root,
            condition=conditions[-1],
            device=resolved_device,
            prompt_config_path=prompt_config_path,
            conditioning_prompt_builder=None,
            progress=progress,
        )
    )

    metadata = {
        "audit_version": MINERVA_SANITY_AUDIT_VERSION,
        "model_id": MINERVA_3B_MODEL_ID,
        "revision": MINERVA_3B_REVISION,
        "split": "validation",
        "prompt_config_path": str(prompt_config_path),
        "prompt_count": len(prompts),
        "seeds": list(MINERVA_SANITY_SEEDS),
        "max_new_tokens": MINERVA_SANITY_MAX_NEW_TOKENS,
        "temperature": 0.8,
        "top_k": 50,
        "best_checkpoint": str(best_path),
        "final_checkpoint": str(final_path),
        "conditions": condition_results,
    }
    (output_root / "audit_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def score_minerva_sanity_audit(output_root: Path) -> list[dict[str, Any]]:
    """Aggregate automatic diagnostics for every completed audit condition."""
    metadata = _load_audit_metadata(output_root)
    rows = []
    for condition in metadata["conditions"]:
        generation_dir = output_root / condition["condition_id"]
        metric_rows = score_generation_directory(generation_dir)
        acceptance_rows = score_task_format_acceptance_directory(generation_dir)
        if len(metric_rows) != MINERVA_SANITY_PROMPT_COUNT:
            raise ValueError("sanity condition has an unexpected output count")
        rows.append({
            **condition,
            "output_count": len(metric_rows),
            "controlled_forms": sum(
                row["automatic_control_pass"] for row in acceptance_rows
            ),
            "token_ceiling_outputs": sum(
                row["stop_reason"] == "max_new_tokens"
                for row in acceptance_rows
            ),
            "mean_characters": statistics.mean(
                row["character_count"] for row in metric_rows
            ),
            "mean_repetition_ratio": statistics.mean(
                row["repetition_ratio"] for row in metric_rows
            ),
        })
    return rows


def build_minerva_sanity_automatic_report(
    output_root: Path,
    rows: Sequence[dict[str, Any]],
) -> str:
    """Render automatic evidence without selecting a condition prematurely."""
    table = [
        "| Condition | Role | Scale | Form | Ceiling | Mean chars | Mean repetition |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        table.append(
            "| {condition} | {role} | {scale:.2f} | {forms}/8 | {ceiling}/8 | "
            "{characters:.1f} | {repetition:.4f} |".format(
                condition=row["condition_id"],
                role="selectable" if row["selectable"] else "diagnostic",
                scale=row["adapter_scale"],
                forms=row["controlled_forms"],
                ceiling=row["token_ceiling_outputs"],
                characters=row["mean_characters"],
                repetition=row["mean_repetition_ratio"],
            )
        )
    return "\n\n".join([
        "# Minerva 3B Validation Sanity Audit: Automatic Evidence",
        f"Generation root: `{output_root}`",
        "## Frozen Scope",
        (
            "Eight V5 validation openings, seed 4242, temperature 0.8, top-k 50, "
            "a 512-token ceiling, and decoder-enforced 14-line stopping. No "
            "final-test prompt or output participates in selection."
        ),
        "## Conditions",
        "\n".join(table),
        "## Selection Rule",
        (
            "Only `best_scale_025`, `best_scale_050`, `best_scale_075`, and "
            "`best_scale_100` are eligible. A condition qualifies only with at least "
            "7/8 controlled forms, at least 5/8 generally grammatical outputs, at "
            "least 5/8 seven-line topic continuations, and no more than 1/8 severe "
            "collapse. Rank qualifiers by grammatical count, then fewer collapses, "
            "then topic count, then lower adapter scale. Human judgments come from "
            "the separately blinded review."
        ),
        (
            "Automatic repetition is diagnostic only and cannot replace the "
            "blinded review."
        ),
    ]) + "\n"


def build_minerva_sanity_blinded_review(
    output_root: Path,
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Hide condition identity while retaining every validation output for review."""
    metadata = _load_audit_metadata(output_root)
    blinded_rows = []
    mapping: dict[str, dict[str, Any]] = {}
    for condition in metadata["conditions"]:
        generation_dir = output_root / condition["condition_id"]
        for review in load_generated_reviews(generation_dir):
            blind_id = hashlib.sha256(
                (
                    f"{MINERVA_SANITY_AUDIT_VERSION}:"
                    f"{condition['condition_id']}:{review['prompt_id']}"
                ).encode("utf-8")
            ).hexdigest()[:12]
            mapping[blind_id] = {
                "condition_id": condition["condition_id"],
                "selectable": condition["selectable"],
                "adapter_scale": condition["adapter_scale"],
                "prompt_id": review["prompt_id"],
                "path": review["path"],
            }
            blinded_rows.append({
                "blind_id": blind_id,
                "prompt_text": review["prompt_text"],
                "seed": review["seed"],
                "generated_text": review["generated_text"],
            })
    blinded_rows.sort(key=lambda row: row["blind_id"])

    sections = [
        "# Minerva 3B Validation Sanity Audit: Blinded Review",
        "Condition identities are deliberately omitted until every judgment is fixed.",
        (
            "For each output, replace each `TODO` with `yes` or `no` and add "
            "a concise note."
        ),
    ]
    for row in blinded_rows:
        sections.append("\n\n".join([
            f"## Output {row['blind_id']}",
            f"- Opening line: `{row['prompt_text']}`",
            f"- Seed: `{row['seed']}`",
            "- Generally grammatical Italian: TODO yes/no",
            (
                "- Topic or argument sustained for at least seven generated "
                "lines: TODO yes/no"
            ),
            "- Severe repetition or generation collapse: TODO yes/no",
            "- Notes: TODO",
            "### Generated Text",
            fenced_text_block(row["generated_text"]),
        ]))
    return "\n\n".join(sections) + "\n", mapping


def write_minerva_sanity_audit_scaffolds(
    *,
    output_root: Path,
    automatic_report_path: Path,
    blinded_review_path: Path,
    blind_mapping_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Write automatic evidence, blinded samples, and the local unblinding key."""
    rows = score_minerva_sanity_audit(output_root)
    automatic_report_path.parent.mkdir(parents=True, exist_ok=True)
    automatic_report_path.write_text(
        build_minerva_sanity_automatic_report(output_root, rows),
        encoding="utf-8",
    )
    review, mapping = build_minerva_sanity_blinded_review(output_root)
    blinded_review_path.parent.mkdir(parents=True, exist_ok=True)
    blinded_review_path.write_text(review, encoding="utf-8")
    blind_mapping_path.parent.mkdir(parents=True, exist_ok=True)
    blind_mapping_path.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return rows, mapping


def _generate_or_reuse_condition(
    *,
    model: Any,
    tokenizer: Any,
    prompts: Sequence[dict[str, str]],
    output_root: Path,
    condition: dict[str, Any],
    device: torch.device,
    prompt_config_path: Path,
    conditioning_prompt_builder: Callable[[str], str] | None,
    progress: Callable[[str], None] | None,
) -> dict[str, Any]:
    output_dir = output_root / condition["condition_id"]
    existing = _reusable_condition_metadata(
        output_dir=output_dir,
        condition=condition,
        prompt_count=len(prompts),
    )
    if existing is not None:
        _report(progress, f"reusing complete condition {condition['condition_id']}")
        return _condition_result(condition, output_dir, existing)

    _report(progress, f"generating condition {condition['condition_id']}")
    metadata = generate_minerva_variant_for_prompts(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        output_dir=output_dir,
        model_variant=condition["condition_id"],
        max_new_tokens=MINERVA_SANITY_MAX_NEW_TOKENS,
        seeds=MINERVA_SANITY_SEEDS,
        device=device,
        adapter_checkpoint_path=(
            Path(condition["checkpoint_path"])
            if condition["checkpoint_path"] is not None
            else None
        ),
        prompt_config_path=prompt_config_path,
        conditioning_prompt_builder=conditioning_prompt_builder,
        conditioning_format=condition["conditioning_format"],
        adapter_scale=condition["adapter_scale"],
        adapter_epoch=condition["adapter_epoch"],
        progress=progress,
    )
    return _condition_result(condition, output_dir, metadata)


def _reusable_condition_metadata(
    *,
    output_dir: Path,
    condition: dict[str, Any],
    prompt_count: int,
) -> dict[str, Any] | None:
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "model_variant": condition["condition_id"],
        "adapter_scale": condition["adapter_scale"],
        "conditioning_format": condition["conditioning_format"],
        "adapter_checkpoint_path": condition["checkpoint_path"],
        "adapter_epoch": condition["adapter_epoch"],
        "max_new_tokens": MINERVA_SANITY_MAX_NEW_TOKENS,
        "seeds": list(MINERVA_SANITY_SEEDS),
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        return None
    generated_files = metadata.get("generated_files")
    if not isinstance(generated_files, list) or len(generated_files) != prompt_count:
        return None
    if not all(Path(row["path"]).is_file() for row in generated_files):
        return None
    return metadata


def _condition_result(
    condition: dict[str, Any],
    output_dir: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        **condition,
        "output_dir": str(output_dir),
        "output_count": len(metadata["generated_files"]),
    }


def _load_audit_metadata(output_root: Path) -> dict[str, Any]:
    metadata = json.loads(
        (output_root / "audit_metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("audit_version") != MINERVA_SANITY_AUDIT_VERSION:
        raise ValueError("generation root is not the fixed Minerva sanity audit")
    condition_ids = tuple(
        condition.get("condition_id") for condition in metadata.get("conditions", [])
    )
    if condition_ids != MINERVA_SANITY_CONDITION_IDS:
        raise ValueError("Minerva sanity audit conditions are missing or reordered")
    return metadata


def _resolve_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _report(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
