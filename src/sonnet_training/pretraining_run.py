"""Run broader Italian from-scratch transformer pretraining."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from sonnet_corpus.bpe import BytePairEncodingTokenizer
from sonnet_model.transformer import CausalTransformerLanguageModel
from sonnet_training.steps import train_next_token_model
from sonnet_training.transformer_run import resolve_device, write_json, write_jsonl


@dataclass(frozen=True)
class PretrainingRunConfig:
    """Configuration for a broader-corpus pretraining run."""

    train_tokens_path: str = "data/local/pretraining/encoded/bpe_8000_train.pt"
    validation_tokens_path: str = (
        "data/local/pretraining/encoded/bpe_8000_validation.pt"
    )
    tokenizer_path: str = "data/local/pretraining/tokenizers/bpe_8000.json"
    batch_size: int = 8
    context_length: int = 512
    train_steps: int = 100
    eval_interval: int = 25
    eval_batches: int = 5
    learning_rate: float = 3e-4
    seed: int = 1337
    prompt: str = "Nel "
    max_new_tokens: int = 300
    device: str = "auto"
    embedding_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    head_dim: int = 32
    feed_forward_dim: int = 1024
    max_context_length: int = 512


def train_pretraining_run(
    repo_root: Path,
    output_dir: Path,
    config: PretrainingRunConfig,
) -> dict[str, Path | list[dict[str, float | int]]]:
    """Train the transformer briefly on broader pretraining token tensors."""

    _validate_config(config)
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)

    train_tokens = load_token_tensor(repo_root / config.train_tokens_path)
    validation_tokens = load_token_tensor(repo_root / config.validation_tokens_path)
    tokenizer = BytePairEncodingTokenizer.load(repo_root / config.tokenizer_path)

    model = CausalTransformerLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        feed_forward_dim=config.feed_forward_dim,
        max_context_length=config.max_context_length,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
    )

    history = train_next_token_model(
        model=model,
        optimizer=optimizer,
        train_token_ids=train_tokens,
        validation_token_ids=validation_tokens,
        batch_size=config.batch_size,
        context_length=config.context_length,
        train_steps=config.train_steps,
        eval_interval=config.eval_interval,
        eval_batches=config.eval_batches,
        device=device,
    )

    generated_text = _generate_sample(
        model=model,
        tokenizer=tokenizer,
        prompt=config.prompt,
        max_new_tokens=config.max_new_tokens,
        device=device,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    log_path = output_dir / "loss_history.jsonl"
    tokenizer_output_path = output_dir / "tokenizer.json"
    sample_path = output_dir / "sample.txt"
    checkpoint_path = output_dir / "model.pt"

    write_json(
        path=config_path,
        payload={
            **asdict(config),
            "resolved_device": str(device),
            "vocab_size": tokenizer.vocab_size,
            "train_tokens": int(train_tokens.numel()),
            "validation_tokens": int(validation_tokens.numel()),
            "parameter_count": count_parameters(model),
        },
    )
    write_jsonl(log_path, history)
    tokenizer.save(tokenizer_output_path)
    sample_path.write_text(generated_text, encoding="utf-8")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "vocab_size": tokenizer.vocab_size,
            "parameter_count": count_parameters(model),
        },
        checkpoint_path,
    )

    return {
        "config_path": config_path,
        "log_path": log_path,
        "tokenizer_path": tokenizer_output_path,
        "sample_path": sample_path,
        "checkpoint_path": checkpoint_path,
        "history": history,
    }


def load_token_tensor(path: Path) -> torch.Tensor:
    """Load one 1D local token tensor for language-model training."""

    if not path.is_file():
        raise FileNotFoundError(f"token tensor file does not exist: {path}")

    tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"token tensor file did not contain a torch.Tensor: {path}")
    if tensor.ndim != 1:
        raise ValueError(f"token tensor must be 1D: {path}")
    if tensor.dtype != torch.long:
        raise ValueError(f"token tensor must use dtype torch.long: {path}")
    return tensor


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable and non-trainable model parameters."""

    return sum(parameter.numel() for parameter in model.parameters())


def _validate_config(config: PretrainingRunConfig) -> None:
    if config.context_length > config.max_context_length:
        raise ValueError(
            "context_length must be less than or equal to max_context_length"
        )
    if config.context_length <= 0:
        raise ValueError("context_length must be greater than 0")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    if config.max_new_tokens < 0:
        raise ValueError("max_new_tokens must be greater than or equal to 0")


def _generate_sample(
    *,
    model: CausalTransformerLanguageModel,
    tokenizer: BytePairEncodingTokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    prompt_ids = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long,
        device=device,
    )
    generated_ids = model.generate(
        input_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
    )
    return tokenizer.decode(generated_ids[0].cpu().tolist())
