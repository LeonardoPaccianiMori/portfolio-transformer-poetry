#!/usr/bin/env python3
"""Run the frozen local NF4 Minerva 7B instruction-judge preflight."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_evaluation.minerva_7b_instruction_judge import (
    build_instruction_judge_report,
    evaluate_instruction_judge,
    load_instruction_judge_cases,
    score_instruction_judge_cases,
)
from sonnet_evaluation.minerva_7b_quality_recovery import (
    load_recovery_config,
    validate_recovery_artifacts,
)
from sonnet_evaluation.minerva_generation import _load_dependencies
from sonnet_evaluation.minerva_judge_gate import sha256_file
from sonnet_training.cuda_compat import (
    cuda_device_name,
    max_cuda_memory_allocated,
    max_cuda_memory_reserved,
    prepare_cuda_memory_measurement,
)
from sonnet_training.progress import format_duration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("configs/minerva_7b_quality_recovery.json"),
    )
    parser.add_argument(
        "--result-path",
        type=Path,
        default=Path(
            "data/local/minerva_quality_recovery/"
            "minerva_7b_instruction_judge_preflight.json"
        ),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/minerva_7b_instruction_judge_preflight.md"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/local/minerva_qlora/huggingface"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--progress-interval", type=int, default=4)
    args = parser.parse_args()
    if args.progress_interval <= 0:
        raise ValueError("--progress-interval must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("the Minerva 7B judge preflight requires CUDA")

    config_path = _resolve(args.config_path)
    config = load_recovery_config(config_path)
    validate_recovery_artifacts(
        config=config,
        repo_root=ROOT,
        require_local_checkpoints=False,
    )
    judge_config = config["judge"]
    cases = load_instruction_judge_cases(
        repo_root=ROOT,
        mapping_path=_resolve(Path(judge_config["human_mapping_path"])),
        judgments_path=_resolve(Path(judge_config["human_judgments_path"])),
    )
    print(
        "minerva-7b-judge | start "
        f"device={device} cases={len(cases)} progress_interval={args.progress_interval} "
        "estimated_runtime=10m-30m_cached final_test=false training=false",
        flush=True,
    )

    dependencies = _load_dependencies()
    cache_dir = _resolve(args.cache_dir)
    print("minerva-7b-judge | loading pinned tokenizer", flush=True)
    tokenizer = dependencies["AutoTokenizer"].from_pretrained(
        config["model_id"],
        revision=config["revision"],
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("minerva-7b-judge | loading untouched 7B in 4-bit NF4", flush=True)
    quantization = dependencies["BitsAndBytesConfig"](
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    prepare_cuda_memory_measurement(device)
    model = dependencies["AutoModelForCausalLM"].from_pretrained(
        config["model_id"],
        revision=config["revision"],
        cache_dir=cache_dir,
        dtype=torch.float16,
        device_map={"": device.index or 0},
        quantization_config=quantization,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.config.use_cache = True

    started_at = perf_counter()

    def progress(index: int, total: int, parsed: bool) -> None:
        if index != 1 and index % args.progress_interval != 0 and index != total:
            return
        elapsed = perf_counter() - started_at
        rate = index / elapsed if elapsed else 0.0
        eta = (total - index) / rate if rate else 0.0
        print(
            "judge-progress | "
            f"case={index}/{total} | progress={100 * index / total:.1f}% "
            f"| parsed={'yes' if parsed else 'no'} "
            f"| elapsed={format_duration(elapsed)} "
            f"| eta={format_duration(eta)}",
            flush=True,
        )

    scored_cases = score_instruction_judge_cases(
        model=model,
        tokenizer=tokenizer,
        cases=cases,
        device=device,
        max_new_tokens=int(judge_config["max_new_tokens"]),
        progress=progress,
    )
    gate = evaluate_instruction_judge(
        scored_cases=scored_cases,
        thresholds=judge_config["thresholds"],
        remote_policy=judge_config["remote_fp16_authorization"],
    )
    result = {
        "judge_version": judge_config["judge_version"],
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "model_id": config["model_id"],
        "revision": config["revision"],
        "load_mode": "nf4",
        "device": str(device),
        "gpu_name": cuda_device_name(device),
        "peak_cuda_allocated_mib": max_cuda_memory_allocated(device) / (1024**2),
        "peak_cuda_reserved_mib": max_cuda_memory_reserved(device) / (1024**2),
        "final_test_used": False,
        "training_used": False,
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
    report_path.write_text(
        build_instruction_judge_report(result), encoding="utf-8"
    )
    print(f"minerva-7b-judge | wrote local result: {result_path}", flush=True)
    print(f"minerva-7b-judge | wrote public report: {report_path}", flush=True)
    print(
        "minerva-7b-judge | complete "
        f"gate={'pass' if gate['gate_passed'] else 'fail'} "
        f"remote_fp16={'authorized' if gate['remote_fp16_authorized'] else 'not_authorized'} "
        f"elapsed={format_duration(perf_counter() - started_at)}",
        flush=True,
    )


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    main()
