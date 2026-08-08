"""Quantized Minerva 7B Instruct validation baseline and review artifacts."""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

from sonnet_evaluation.minerva_generation import (
    _load_dependencies,
    generate_minerva_variant_for_prompts,
)
from sonnet_evaluation.qualitative import fenced_text_block, load_generated_reviews
from sonnet_evaluation.task_acceptance import score_task_format_acceptance_directory
from sonnet_training.minerva_7b_qlora import (
    MINERVA_7B_INSTRUCT_MODEL_ID,
    MINERVA_7B_INSTRUCT_REVISION,
)


MINERVA_7B_BASELINE_VERSION = "minerva_7b_instruct_validation_baseline_v1"
MINERVA_7B_BASELINE_VARIANT = "minerva_7b_instruct_4bit"
MINERVA_7B_BASELINE_SEEDS = (4242,)
MINERVA_7B_BASELINE_MAX_NEW_TOKENS = 512


def build_minerva_7b_instruct_prompt(tokenizer: Any, opening_line: str) -> str:
    """Apply the published chat template and prefill the exact opening line."""
    if not opening_line.strip() or "\n" in opening_line or "\r" in opening_line:
        raise ValueError("opening_line must contain exactly one non-empty line")
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise ValueError("Minerva 7B Instruct tokenizer must define a chat template")
    user_content = (
        "Componi un sonetto in italiano classico di esattamente quattordici "
        "versi. Usa come primo verso esattamente quello indicato, mantieni un "
        "tema coerente e una sintassi grammaticale, ed evita ripetizioni. "
        "Restituisci soltanto il sonetto, senza titolo, spiegazioni o commenti.\n\n"
        f"Primo verso: {opening_line}"
    )
    rendered = apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("Minerva chat template must render a non-empty string")
    return f"{rendered}{opening_line}\n"


def generate_minerva_7b_instruct_baseline(
    *,
    output_root: Path,
    prompts: Sequence[dict[str, str]],
    prompt_config_path: Path,
    device: torch.device | str,
    cache_dir: Path,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Load the exact 7B Instruct revision in NF4 and generate eight outputs."""
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda":
        raise ValueError("Minerva 7B Instruct baseline requires a CUDA device")
    if len(prompts) != 8:
        raise ValueError("Minerva 7B Instruct baseline requires eight prompts")

    dependencies = _load_dependencies()
    cache_dir.mkdir(parents=True, exist_ok=True)
    _report(progress, "loading published Minerva 7B Instruct tokenizer")
    tokenizer = dependencies["AutoTokenizer"].from_pretrained(
        MINERVA_7B_INSTRUCT_MODEL_ID,
        revision=MINERVA_7B_INSTRUCT_REVISION,
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Render once before allocating model memory so template incompatibility fails early.
    build_minerva_7b_instruct_prompt(tokenizer, prompts[0]["opening_line"])

    _report(progress, "loading 7B Instruct weights in 4-bit NF4")
    quantization = dependencies["BitsAndBytesConfig"](
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    device_index = resolved_device.index or 0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)
    model = dependencies["AutoModelForCausalLM"].from_pretrained(
        MINERVA_7B_INSTRUCT_MODEL_ID,
        revision=MINERVA_7B_INSTRUCT_REVISION,
        cache_dir=cache_dir,
        quantization_config=quantization,
        torch_dtype=torch.float16,
        device_map={"": device_index},
    )
    model.eval()
    model.config.use_cache = True

    output_dir = output_root / "instruct"
    _report(progress, "generating frozen validation baseline")
    generation_metadata = generate_minerva_variant_for_prompts(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        output_dir=output_dir,
        model_variant=MINERVA_7B_BASELINE_VARIANT,
        max_new_tokens=MINERVA_7B_BASELINE_MAX_NEW_TOKENS,
        seeds=MINERVA_7B_BASELINE_SEEDS,
        device=resolved_device,
        prompt_config_path=prompt_config_path,
        conditioning_prompt_builder=lambda opening: build_minerva_7b_instruct_prompt(
            tokenizer, opening
        ),
        conditioning_format="published_chat_template_with_opening_prefill_v1",
        model_id=MINERVA_7B_INSTRUCT_MODEL_ID,
        revision=MINERVA_7B_INSTRUCT_REVISION,
        progress=progress,
    )
    baseline_metadata = {
        "baseline_version": MINERVA_7B_BASELINE_VERSION,
        "model_id": MINERVA_7B_INSTRUCT_MODEL_ID,
        "revision": MINERVA_7B_INSTRUCT_REVISION,
        "license": "Apache-2.0",
        "quantization": {
            "load_in_4bit": True,
            "quant_type": "nf4",
            "double_quantization": True,
            "compute_dtype": "float16",
        },
        "prompt_count": len(prompts),
        "output_count": len(generation_metadata["generated_files"]),
        "output_dir": str(output_dir),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device_index)
        / (1024**2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device_index) / (1024**2),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "baseline_metadata.json").write_text(
        json.dumps(baseline_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return baseline_metadata


def write_minerva_7b_instruct_baseline_scaffolds(
    *,
    generation_dir: Path,
    automatic_report_path: Path,
    review_path: Path,
) -> list[dict[str, Any]]:
    """Write automatic controls and the fixed human-quality review scaffold."""
    rows = score_task_format_acceptance_directory(generation_dir)
    if len(rows) != 8:
        raise ValueError("Minerva 7B baseline evaluation requires eight outputs")
    automatic_report_path.parent.mkdir(parents=True, exist_ok=True)
    automatic_report_path.write_text(
        _build_automatic_report(generation_dir, rows),
        encoding="utf-8",
    )
    reviews = load_generated_reviews(generation_dir)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(_build_review(reviews), encoding="utf-8")
    return rows


def _build_automatic_report(
    generation_dir: Path,
    rows: Sequence[dict[str, Any]],
) -> str:
    controlled = sum(row["automatic_control_pass"] for row in rows)
    ceiling = sum(row["stop_reason"] == "max_new_tokens" for row in rows)
    mean_repetition = statistics.mean(row["repetition_ratio"] for row in rows)
    lines = [
        "# Minerva 7B Instruct Validation Baseline: Automatic Evidence",
        "",
        f"Generation directory: `{generation_dir}`",
        "",
        f"- Exact-opening controlled forms: **{controlled}/8**.",
        f"- Outputs reaching the 512-token ceiling: **{ceiling}/8**.",
        f"- Mean repeated character 4-gram ratio: **{mean_repetition:.4f}**.",
        "- Line count is decoder-enforced and is not evidence of metre or rhyme.",
        "- The parent-quality decision requires the separately recorded human judgments.",
        "",
        "| Output | Author | Lines | Exact opening | Controlled form | Stop | Repetition |",
        "| --- | --- | ---: | --- | --- | --- | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['prompt_id']} | {row['author']} | "
            f"{row['non_empty_line_count']} | {_yes_no(row['opening_line_preserved'])} | "
            f"{_yes_no(row['controlled_sonnet_form'])} | {row['stop_reason']} | "
            f"{row['repetition_ratio']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def _build_review(reviews: Sequence[dict[str, Any]]) -> str:
    sections = [
        "# Minerva 7B Instruct Validation Baseline: Quality Review",
        "",
        "Qualification requires at least 5/8 generally grammatical outputs, "
        "at least 5/8 seven-line topic continuations, and at most 1/8 severe "
        "collapse. Replace every `TODO` before making the decision.",
    ]
    for review in reviews:
        sections.extend([
            "",
            f"## {review['prompt_id']}",
            "",
            f"- Opening line: `{review['prompt_text']}`",
            f"- Seed: `{review['seed']}`",
            "- Generally grammatical Italian: TODO yes/no",
            "- Topic or argument sustained for at least seven generated lines: TODO yes/no",
            "- Severe repetition or generation collapse: TODO yes/no",
            "- Notes: TODO",
            "",
            "### Generated Text",
            "",
            fenced_text_block(review["generated_text"]),
        ])
    return "\n".join(sections) + "\n"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _report(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
