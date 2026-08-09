#!/usr/bin/env python3
"""Generate validation controls and run the frozen Minerva 3B judge gate."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.task_format import build_sonnet_continuation_examples
from sonnet_evaluation.minerva_judge_gate import (
    JUDGE_CONTROL_SEED,
    JUDGE_HUMAN_CASE_COUNT,
    JUDGE_PROMPT_COUNT,
    build_judge_cases,
    build_judge_gate_report,
    evaluate_judge_gate,
    load_judge_gate_config,
    score_judge_cases,
    sha256_file,
    validate_judge_gate_artifacts,
)
from sonnet_evaluation.task_generation import (
    generate_task_format_for_prompts,
    load_task_format_prompts,
    validate_task_format_prompts_against_manifest,
)
from sonnet_training.cuda_compat import (
    cuda_device_name,
    max_cuda_memory_allocated,
    max_cuda_memory_reserved,
    prepare_cuda_memory_measurement,
)
from sonnet_training.progress import format_duration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("configs/minerva_3b_judge_gate.json"),
    )
    parser.add_argument(
        "--generated-dir",
        type=Path,
        default=Path(
            "outputs/generations/minerva_3b_judge_gate_from_scratch_validation_v1"
        ),
    )
    parser.add_argument(
        "--result-path",
        type=Path,
        default=Path("data/local/minerva_guided/minerva_3b_judge_gate.json"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/minerva_3b_judge_gate.md"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/local/minerva_qlora/huggingface"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--progress-interval", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.progress_interval <= 0:
        raise ValueError("--progress-interval must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("the frozen FP16 Minerva judge gate requires CUDA")

    config_path = _resolve(args.config_path)
    config = load_judge_gate_config(config_path)
    validate_judge_gate_artifacts(config=config, repo_root=ROOT)
    manifest_path = _resolve(Path(config["manifest_path"]))
    prompt_path = _resolve(Path(config["validation_prompt_path"]))
    prompts = load_task_format_prompts(prompt_path)
    validate_task_format_prompts_against_manifest(
        prompts=prompts,
        manifest_path=manifest_path,
        repo_root=ROOT,
        dataset="expanded_with_petrarch",
        split="validation",
    )
    if len(prompts) != JUDGE_PROMPT_COUNT:
        raise ValueError("judge gate requires eight frozen validation prompts")

    generated_dir = _resolve(args.generated_dir)
    print(
        "judge-gate | start "
        f"device={device} validation_prompts={len(prompts)} "
        f"human_controls={JUDGE_HUMAN_CASE_COUNT} "
        "estimated_runtime=10m-30m_cached",
        flush=True,
    )
    _ensure_from_scratch_controls(
        config=config,
        prompts=prompts,
        prompt_path=prompt_path,
        generated_dir=generated_dir,
        device=device,
    )
    gc.collect()
    torch.cuda.empty_cache()

    validation_examples = build_sonnet_continuation_examples(
        manifest_path=manifest_path,
        repo_root=ROOT,
        dataset="expanded_with_petrarch",
        split="validation",
    )
    cases = build_judge_cases(
        repo_root=ROOT,
        prompts=prompts,
        validation_examples=validation_examples,
        generated_dir=generated_dir,
        human_mapping_path=_resolve(Path(config["human_mapping_path"])),
        human_judgments_path=_resolve(Path(config["human_judgments_path"])),
    )
    print(
        f"judge-gate | built validation-only cases count={len(cases)} "
        "final_test_used=false",
        flush=True,
    )

    dependencies = _load_dependencies()
    cache_dir = _resolve(args.cache_dir)
    print("judge-gate | loading pinned Minerva 3B tokenizer", flush=True)
    tokenizer = dependencies["AutoTokenizer"].from_pretrained(
        config["model_id"],
        revision=config["revision"],
        cache_dir=cache_dir,
    )
    prepare_cuda_memory_measurement(device)
    print("judge-gate | loading untouched Minerva 3B in FP16", flush=True)
    model = dependencies["AutoModelForCausalLM"].from_pretrained(
        config["model_id"],
        revision=config["revision"],
        cache_dir=cache_dir,
        dtype=torch.float16,
        device_map={"": device.index or 0},
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.config.use_cache = False

    score_started_at = perf_counter()
    print(
        f"minerva_3b_judge_gate started | steps=0/{len(cases)} "
        f"| device={device} | progress_interval={args.progress_interval}",
        flush=True,
    )

    def report_progress(index: int, total: int, mean_nll: float) -> None:
        if index != 1 and index % args.progress_interval != 0 and index != total:
            return
        elapsed = perf_counter() - score_started_at
        rate = index / elapsed if elapsed else 0.0
        eta = (total - index) / rate if rate else 0.0
        print(
            "judge-progress | "
            f"case={index}/{total} | progress={100 * index / total:.1f}% "
            f"| mean_nll={mean_nll:.4f} "
            f"| elapsed={format_duration(elapsed)} "
            f"| eta={format_duration(eta)}",
            flush=True,
        )

    scored_cases = score_judge_cases(
        model=model,
        tokenizer=tokenizer,
        cases=cases,
        device=device,
        context_length=int(config["context_length"]),
        progress=report_progress,
    )
    gate = evaluate_judge_gate(
        scored_cases=scored_cases,
        thresholds=config["thresholds"],
    )
    result = {
        "gate_version": config["gate_version"],
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "model_id": config["model_id"],
        "revision": config["revision"],
        "model_precision": config["model_precision"],
        "device": str(device),
        "gpu_name": cuda_device_name(device),
        "peak_cuda_allocated_mib": max_cuda_memory_allocated(device) / (1024**2),
        "peak_cuda_reserved_mib": max_cuda_memory_reserved(device) / (1024**2),
        "final_test_used": False,
        "generated_control_hashes": {
            path.name: sha256_file(path)
            for path in sorted(generated_dir.glob("*.txt"))
        },
        "scored_cases": scored_cases,
        "gate": gate,
    }
    result_path = _resolve(args.result_path)
    report_path = _resolve(args.report_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(build_judge_gate_report(result), encoding="utf-8")
    print(f"judge-gate | wrote result: {result_path}", flush=True)
    print(f"judge-gate | wrote public report: {report_path}", flush=True)
    print(
        "judge-gate | complete "
        f"status={'pass' if gate['gate_passed'] else 'fail'} "
        f"elapsed={format_duration(perf_counter() - score_started_at)}",
        flush=True,
    )


def _ensure_from_scratch_controls(
    *,
    config: dict[str, Any],
    prompts: list[dict[str, str]],
    prompt_path: Path,
    generated_dir: Path,
    device: torch.device,
) -> None:
    metadata_path = generated_dir / "metadata.json"
    if metadata_path.is_file():
        _validate_control_metadata(
            metadata=json.loads(metadata_path.read_text(encoding="utf-8")),
            prompts=prompts,
            generated_dir=generated_dir,
            recipe=config["control_generation"],
        )
        print("judge-gate | reusing validated from-scratch controls", flush=True)
        return
    if generated_dir.exists() and any(generated_dir.iterdir()):
        raise ValueError("judge control directory is partial and has no metadata")

    recipe = config["control_generation"]
    run_dir = _resolve(Path(config["from_scratch_run_dir"]))
    checkpoint_path = _resolve(Path(config["from_scratch_checkpoint_path"]))
    print(
        "judge-gate | generating 8 fixed from-scratch validation controls",
        flush=True,
    )
    metadata = generate_task_format_for_prompts(
        run_dir=run_dir,
        prompts=prompts,
        output_dir=generated_dir,
        max_new_tokens=int(recipe["max_new_tokens"]),
        seeds=[JUDGE_CONTROL_SEED],
        device=device,
        temperature=float(recipe["temperature"]),
        top_k=int(recipe["top_k"]),
        continuation_line_target=int(recipe["continuation_line_target"]),
        checkpoint_path=checkpoint_path,
        model_config_path=run_dir / "config.json",
        prompt_config_path=prompt_path,
        progress=lambda message: print(f"judge-controls | {message}", flush=True),
    )
    _validate_control_metadata(
        metadata=metadata,
        prompts=prompts,
        generated_dir=generated_dir,
        recipe=recipe,
    )


def _validate_control_metadata(
    *,
    metadata: dict[str, Any],
    prompts: list[dict[str, str]],
    generated_dir: Path,
    recipe: dict[str, Any],
) -> None:
    expected = {
        "seeds": [JUDGE_CONTROL_SEED],
        "max_new_tokens": recipe["max_new_tokens"],
        "temperature": recipe["temperature"],
        "top_k": recipe["top_k"],
        "continuation_line_target": recipe["continuation_line_target"],
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"judge control metadata mismatch: {key}")
    generated_files = metadata.get("generated_files")
    if not isinstance(generated_files, list) or len(generated_files) != len(prompts):
        raise ValueError("judge control metadata must contain eight outputs")
    rows_by_source = {row["source_prompt_id"]: row for row in generated_files}
    if set(rows_by_source) != {prompt["id"] for prompt in prompts}:
        raise ValueError("judge control prompt IDs differ from the frozen set")
    for prompt in prompts:
        output_path = generated_dir / f"{prompt['id']}__seed_{JUDGE_CONTROL_SEED}.txt"
        lines = [
            line.strip()
            for line in output_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(lines) != 14 or lines[0] != prompt["opening_line"]:
            raise ValueError(f"judge control is not an exact 14-line output: {output_path}")


def _load_dependencies() -> dict[str, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Minerva judge dependencies are missing") from error
    return {
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
    }


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    main()
