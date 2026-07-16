import json
from pathlib import Path

import pytest
import torch

from sonnet_corpus.pretraining_tokenizer import train_weighted_pretoken_bpe_tokenizer
from sonnet_training.pretraining_benchmark import (
    PretrainingBenchmarkConfig,
    PretrainingModelCandidate,
    _cuda_device_index,
    _call_cuda_device_function,
    benchmark_pretraining_candidates,
    build_markdown_report,
    default_pretraining_candidates,
)


def write_tiny_benchmark_artifacts(repo_root: Path) -> None:
    corpus_dir = (
        repo_root
        / "data"
        / "local"
        / "pretraining"
        / "expanded_italian_1200_1800_v1"
    )
    encoded_dir = corpus_dir / "encoded"
    tokenizer_dir = corpus_dir / "tokenizers"
    encoded_dir.mkdir(parents=True)
    tokenizer_dir.mkdir(parents=True)
    text = "amor antico memoria cronica virtute novella lingua storia\n"
    tokenizer = train_weighted_pretoken_bpe_tokenizer(
        training_text=text,
        base_text=text,
        vocab_size=50,
        special_tokens=["<|endoftext|>"],
    )
    tokenizer.save(tokenizer_dir / "bpe_8000.json")
    torch.save(
        torch.tensor(([1, 2, 3, 4, 5, 6] * 40), dtype=torch.long),
        encoded_dir / "bpe_8000_train.pt",
    )
    torch.save(
        torch.tensor(([1, 2, 3, 4, 5, 6] * 40), dtype=torch.long),
        encoded_dir / "bpe_8000_validation.pt",
    )


def tiny_candidate() -> PretrainingModelCandidate:
    return PretrainingModelCandidate(
        name="tiny",
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        head_dim=4,
        feed_forward_dim=16,
        batch_size=4,
    )


def test_default_pretraining_candidates_match_confirmed_names_and_batches():
    candidates = default_pretraining_candidates()

    assert [candidate.name for candidate in candidates] == [
        "small",
        "medium",
        "larger",
        "upper",
        "max",
    ]
    assert [candidate.batch_size for candidate in candidates] == [8, 4, 2, 1, 1]
    assert candidates[0].embedding_dim == 256
    assert candidates[-2].num_layers == 10
    assert candidates[-1].embedding_dim == 768
    assert candidates[-1].num_layers == 12
    assert candidates[-1].num_heads == 12
    assert candidates[-1].head_dim == 64
    assert candidates[-1].feed_forward_dim == 3072


def test_benchmark_pretraining_candidates_writes_reports(tmp_path: Path):
    write_tiny_benchmark_artifacts(tmp_path)
    json_report_path = Path("data/local/pretraining/benchmarks/benchmark.json")
    markdown_report_path = Path("reports/pretraining_hardware_benchmark.md")

    progress_messages = []
    report = benchmark_pretraining_candidates(
        repo_root=tmp_path,
        config=PretrainingBenchmarkConfig(
            json_report_path=json_report_path,
            markdown_report_path=markdown_report_path,
            context_length=8,
            warmup_steps=1,
            benchmark_steps=2,
            eval_batches=1,
            device="cpu",
        ),
        candidates=[tiny_candidate()],
        progress=progress_messages.append,
    )

    result = report["results"][0]
    assert result["name"] == "tiny"
    assert result["status"] == "ok"
    assert result["parameter_count"] > 0
    assert result["tokens_per_second"] > 0
    assert result["seconds_per_step"] > 0
    assert result["peak_cuda_memory_mib"] is None
    assert (tmp_path / json_report_path).is_file()
    assert (tmp_path / markdown_report_path).is_file()

    saved = json.loads((tmp_path / json_report_path).read_text(encoding="utf-8"))
    assert saved["results"][0]["name"] == "tiny"
    markdown = (tmp_path / markdown_report_path).read_text(encoding="utf-8")
    assert "# Pretraining Hardware Benchmark" in markdown
    assert "| tiny | ok |" in markdown
    assert "resolved device: cpu" in progress_messages
    assert "candidate 1/1: tiny" in progress_messages
    assert "tiny: timed steps 2/2" in progress_messages
    assert "complete" in progress_messages


def test_benchmark_pretraining_candidates_rejects_invalid_step_count(tmp_path: Path):
    write_tiny_benchmark_artifacts(tmp_path)

    with pytest.raises(ValueError, match="benchmark_steps"):
        benchmark_pretraining_candidates(
            repo_root=tmp_path,
            config=PretrainingBenchmarkConfig(
                context_length=8,
                benchmark_steps=0,
                device="cpu",
            ),
            candidates=[tiny_candidate()],
        )


def test_build_markdown_report_formats_error_rows():
    markdown = build_markdown_report({
        "device": "cpu",
        "cuda_available": False,
        "vocab_size": 50,
        "context_length": 8,
        "warmup_steps": 1,
        "benchmark_steps": 2,
        "eval_batches": 1,
        "learning_rate": 0.001,
        "results": [
            {
                "name": "bad",
                "status": "error",
                "candidate": {"batch_size": 4},
                "parameter_count": None,
                "seconds_per_step": None,
                "tokens_per_second": None,
                "peak_cuda_memory_mib": None,
                "last_train_loss": None,
                "validation_loss": None,
            }
        ],
    })

    assert "| bad | error |" in markdown


def test_cuda_device_index_handles_explicit_and_implicit_cuda_devices():
    assert _cuda_device_index(torch.device("cuda")) == 0
    assert _cuda_device_index(torch.device("cuda:0")) == 0
    assert _cuda_device_index(torch.device("cuda:1")) == 1


def test_cuda_device_index_rejects_cpu_device():
    with pytest.raises(ValueError, match="CUDA"):
        _cuda_device_index(torch.device("cpu"))


def test_call_cuda_device_function_falls_back_to_no_argument_call(monkeypatch):
    calls = []

    def fake_set_device(device_index):
        calls.append(("set_device", device_index))

    def fake_cuda_function(device_index=None):
        calls.append(("function", device_index))
        if device_index is not None:
            raise RuntimeError("Invalid device argument")
        return 123

    monkeypatch.setattr(torch.cuda, "set_device", fake_set_device)

    result = _call_cuda_device_function(
        fake_cuda_function,
        torch.device("cuda:0"),
    )

    assert result == 123
    assert calls == [
        ("set_device", 0),
        ("function", 0),
        ("function", None),
    ]


def test_call_cuda_device_function_returns_default_if_both_calls_fail(monkeypatch):
    monkeypatch.setattr(torch.cuda, "set_device", lambda device_index: None)

    def always_fails(device_index=None):
        raise RuntimeError("Invalid device argument")

    result = _call_cuda_device_function(
        always_fails,
        torch.device("cuda:0"),
        default="fallback",
    )

    assert result == "fallback"
