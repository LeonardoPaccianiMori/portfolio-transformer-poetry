"""Benchmark broader-pretraining model candidates."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from sonnet_corpus.bpe import BytePairEncodingTokenizer
from sonnet_model.transformer import CausalTransformerLanguageModel
from sonnet_training.pretraining_run import count_parameters, load_token_tensor
from sonnet_training.steps import estimate_next_token_loss, train_next_token_step
from sonnet_training.transformer_run import resolve_device


@dataclass(frozen=True)
class PretrainingModelCandidate:
    """One architecture and batch-size candidate for broader pretraining."""

    name: str
    embedding_dim: int
    num_layers: int
    num_heads: int
    head_dim: int
    feed_forward_dim: int
    batch_size: int


@dataclass(frozen=True)
class PretrainingBenchmarkConfig:
    """Configuration for benchmarking broader-pretraining candidates."""

    train_tokens_path: Path = Path("data/local/pretraining/encoded/bpe_8000_train.pt")
    validation_tokens_path: Path = Path(
        "data/local/pretraining/encoded/bpe_8000_validation.pt"
    )
    tokenizer_path: Path = Path("data/local/pretraining/tokenizers/bpe_8000.json")
    json_report_path: Path = Path(
        "data/local/pretraining/benchmarks/pretraining_benchmark.json"
    )
    markdown_report_path: Path = Path("reports/pretraining_hardware_benchmark.md")
    context_length: int = 512
    warmup_steps: int = 3
    benchmark_steps: int = 20
    eval_batches: int = 1
    learning_rate: float = 3e-4
    seed: int = 1337
    device: str = "auto"


def default_pretraining_candidates() -> list[PretrainingModelCandidate]:
    """Return the confirmed first benchmark candidate set."""

    return [
        PretrainingModelCandidate(
            name="small",
            embedding_dim=256,
            num_layers=6,
            num_heads=8,
            head_dim=32,
            feed_forward_dim=1024,
            batch_size=8,
        ),
        PretrainingModelCandidate(
            name="medium",
            embedding_dim=384,
            num_layers=8,
            num_heads=8,
            head_dim=48,
            feed_forward_dim=1536,
            batch_size=4,
        ),
        PretrainingModelCandidate(
            name="larger",
            embedding_dim=512,
            num_layers=8,
            num_heads=8,
            head_dim=64,
            feed_forward_dim=2048,
            batch_size=2,
        ),
        PretrainingModelCandidate(
            name="upper",
            embedding_dim=640,
            num_layers=10,
            num_heads=10,
            head_dim=64,
            feed_forward_dim=2560,
            batch_size=1,
        ),
    ]


def benchmark_pretraining_candidates(
    *,
    repo_root: Path,
    config: PretrainingBenchmarkConfig,
    candidates: list[PretrainingModelCandidate] | None = None,
) -> dict[str, Any]:
    """Benchmark candidate architectures and write JSON/Markdown reports."""

    _validate_config(config)
    started_at = _utc_now()
    selected_candidates = candidates or default_pretraining_candidates()
    device = resolve_device(config.device)
    tokenizer = BytePairEncodingTokenizer.load(repo_root / config.tokenizer_path)
    train_tokens = load_token_tensor(repo_root / config.train_tokens_path)
    validation_tokens = load_token_tensor(repo_root / config.validation_tokens_path)

    results = []
    for index, candidate in enumerate(selected_candidates):
        torch.manual_seed(config.seed + index)
        results.append(
            benchmark_one_candidate(
                candidate=candidate,
                config=config,
                tokenizer=tokenizer,
                train_tokens=train_tokens,
                validation_tokens=validation_tokens,
                device=device,
            )
        )

    report = {
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "vocab_size": tokenizer.vocab_size,
        "context_length": config.context_length,
        "warmup_steps": config.warmup_steps,
        "benchmark_steps": config.benchmark_steps,
        "eval_batches": config.eval_batches,
        "learning_rate": config.learning_rate,
        "train_tokens": int(train_tokens.numel()),
        "validation_tokens": int(validation_tokens.numel()),
        "results": results,
    }
    _write_json(repo_root / config.json_report_path, report)
    _write_markdown(repo_root / config.markdown_report_path, report)
    return report


def benchmark_one_candidate(
    *,
    candidate: PretrainingModelCandidate,
    config: PretrainingBenchmarkConfig,
    tokenizer: BytePairEncodingTokenizer,
    train_tokens: torch.Tensor,
    validation_tokens: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    """Benchmark one candidate and return a serializable result row."""

    started_at = _utc_now()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    try:
        model = CausalTransformerLanguageModel(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=candidate.embedding_dim,
            num_layers=candidate.num_layers,
            num_heads=candidate.num_heads,
            head_dim=candidate.head_dim,
            feed_forward_dim=candidate.feed_forward_dim,
            max_context_length=config.context_length,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        parameter_count = count_parameters(model)

        for _ in range(config.warmup_steps):
            train_next_token_step(
                model=model,
                optimizer=optimizer,
                token_ids=train_tokens,
                batch_size=candidate.batch_size,
                context_length=config.context_length,
                device=device,
            )

        _synchronize_if_cuda(device)
        started = time.perf_counter()
        last_train_loss = 0.0
        for _ in range(config.benchmark_steps):
            last_train_loss = train_next_token_step(
                model=model,
                optimizer=optimizer,
                token_ids=train_tokens,
                batch_size=candidate.batch_size,
                context_length=config.context_length,
                device=device,
            )
        _synchronize_if_cuda(device)
        elapsed_seconds = time.perf_counter() - started

        validation_loss = estimate_next_token_loss(
            model=model,
            token_ids=validation_tokens,
            batch_size=candidate.batch_size,
            context_length=config.context_length,
            eval_batches=config.eval_batches,
            device=device,
        )
        tokens_processed = (
            config.benchmark_steps * candidate.batch_size * config.context_length
        )
        return {
            "name": candidate.name,
            "status": "ok",
            "error": "",
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now(),
            "candidate": asdict(candidate),
            "parameter_count": parameter_count,
            "last_train_loss": last_train_loss,
            "validation_loss": validation_loss,
            "elapsed_seconds": elapsed_seconds,
            "seconds_per_step": elapsed_seconds / config.benchmark_steps,
            "tokens_per_second": tokens_processed / elapsed_seconds,
            "peak_cuda_memory_mib": _peak_cuda_memory_mib(device),
        }
    except RuntimeError as exc:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {
            "name": candidate.name,
            "status": "error",
            "error": str(exc),
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now(),
            "candidate": asdict(candidate),
            "parameter_count": None,
            "last_train_loss": None,
            "validation_loss": None,
            "elapsed_seconds": None,
            "seconds_per_step": None,
            "tokens_per_second": None,
            "peak_cuda_memory_mib": _peak_cuda_memory_mib(device),
        }


def _validate_config(config: PretrainingBenchmarkConfig) -> None:
    if config.context_length <= 0:
        raise ValueError("context_length must be greater than 0")
    if config.warmup_steps < 0:
        raise ValueError("warmup_steps must be greater than or equal to 0")
    if config.benchmark_steps <= 0:
        raise ValueError("benchmark_steps must be greater than 0")
    if config.eval_batches <= 0:
        raise ValueError("eval_batches must be greater than 0")


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_cuda_memory_mib(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024 * 1024)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_markdown_report(report), encoding="utf-8")


def build_markdown_report(report: dict[str, Any]) -> str:
    """Build the public Markdown benchmark summary."""

    lines = [
        "# Pretraining Hardware Benchmark",
        "",
        "This report benchmarks candidate broader-pretraining model sizes on the",
        "local hardware using the current BPE-encoded broader Italian corpus.",
        "",
        "## Configuration",
        "",
        f"- Device: `{report['device']}`",
        f"- CUDA available: `{report['cuda_available']}`",
        f"- Vocabulary size: `{report['vocab_size']}`",
        f"- Context length: `{report['context_length']}`",
        f"- Warmup steps: `{report['warmup_steps']}`",
        f"- Timed steps: `{report['benchmark_steps']}`",
        f"- Evaluation batches: `{report['eval_batches']}`",
        f"- Learning rate: `{report['learning_rate']}`",
        "",
        "## Results",
        "",
        "| Candidate | Status | Params | Batch | Seconds/Step | "
        "Tokens/Sec | Peak CUDA MiB | Train Loss | Validation Loss |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for result in report["results"]:
        candidate = result["candidate"]
        lines.append(
            (
                "| {name} | {status} | {params} | {batch} | {seconds} | "
                "{tokens} | {memory} | {train_loss} | {validation_loss} |"
            ).format(
                name=result["name"],
                status=result["status"],
                params=_format_int(result["parameter_count"]),
                batch=candidate["batch_size"],
                seconds=_format_float(result["seconds_per_step"]),
                tokens=_format_float(result["tokens_per_second"]),
                memory=_format_float(result["peak_cuda_memory_mib"]),
                train_loss=_format_float(result["last_train_loss"]),
                validation_loss=_format_float(result["validation_loss"]),
            )
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "Use this report to choose the largest model that fits reliably and still",
        "processes enough tokens per second for a long local pretraining run.",
        "A successful benchmark does not prove final generation quality; it only",
        "measures practical training throughput and memory for candidate sizes.",
        "",
    ])
    return "\n".join(lines)


def _format_int(value: int | None) -> str:
    if value is None:
        return ""
    return f"{value:,}"


def _format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
