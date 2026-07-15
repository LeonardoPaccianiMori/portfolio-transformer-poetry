import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from sonnet_corpus.bpe import BytePairEncodingTokenizer
from sonnet_corpus.dataset_text import load_bpe_encoded_splits, load_encoded_splits
from sonnet_model.normalization import NormalizationType
from sonnet_model.positional_encoding import PositionEncodingType
from sonnet_model.transformer import CausalTransformerLanguageModel, FeedForwardType
from sonnet_training.progress import TrainingProgressReporter
from sonnet_training.steps import train_next_token_model


@dataclass(frozen=True)
class TransformerTrainingConfig:
    dataset: str = "expanded_with_petrarch"
    tokenizer_type: str = "character"
    bpe_tokenizer_path: str = "data/metadata/bpe_tokenizer.json"
    batch_size: int = 32
    context_length: int = 128
    train_steps: int = 200
    eval_interval: int = 50
    eval_batches: int = 10
    learning_rate: float = 3e-4
    seed: int = 1337
    prompt: str = "Amor"
    max_new_tokens: int = 400
    device: str = "auto"
    embedding_dim: int = 32
    num_layers: int = 2
    num_heads: int = 2
    head_dim: int = 16
    feed_forward_dim: int = 128
    max_context_length: int = 128
    normalization_type: NormalizationType = "layer_norm"
    normalization_eps: float = 1e-5
    position_encoding_type: PositionEncodingType = "learned_absolute"
    rope_theta: float = 10_000.0
    feed_forward_type: FeedForwardType = "relu"
    progress_interval: int = 100


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return torch.device(device)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def save_tokenizer(path: Path, char_to_id: dict[str, int]) -> None:
    write_json(
        path=path,
        payload={
            "type": "character",
            "char_to_id": char_to_id,
        },
    )


def load_training_splits_for_tokenizer(
    repo_root: Path,
    manifest_path: Path,
    config: TransformerTrainingConfig,
):
    if config.tokenizer_type == "character":
        return load_encoded_splits(
            manifest_path=manifest_path,
            repo_root=repo_root,
            dataset=config.dataset,
        )

    if config.tokenizer_type == "bpe":
        return load_bpe_encoded_splits(
            manifest_path=manifest_path,
            repo_root=repo_root,
            dataset=config.dataset,
            tokenizer_path=repo_root / config.bpe_tokenizer_path,
        )

    raise ValueError("tokenizer_type must be 'character' or 'bpe'")


def save_run_tokenizer(path: Path, tokenizer) -> None:
    if isinstance(tokenizer, BytePairEncodingTokenizer):
        tokenizer.save(path)
        return

    save_tokenizer(path, tokenizer.char_to_id)


def train_transformer_run(
    repo_root: Path,
    output_dir: Path,
    config: TransformerTrainingConfig,
) -> dict[str, Path | list[dict[str, float | int]]]:
    if config.context_length > config.max_context_length:
        raise ValueError("context_length must be less than or equal to max_context_length")

    torch.manual_seed(config.seed)

    device = resolve_device(config.device)
    manifest_path = repo_root / "data" / "metadata" / "poems_manifest.csv"

    train_tokens, validation_tokens, _, tokenizer = load_training_splits_for_tokenizer(
        repo_root=repo_root,
        manifest_path=manifest_path,
        config=config,
    )

    model = CausalTransformerLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        feed_forward_dim=config.feed_forward_dim,
        max_context_length=config.max_context_length,
        normalization_type=config.normalization_type,
        normalization_eps=config.normalization_eps,
        position_encoding_type=config.position_encoding_type,
        rope_theta=config.rope_theta,
        feed_forward_type=config.feed_forward_type,
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
        progress_interval=config.progress_interval,
    )

    prompt_ids = torch.tensor(
        [tokenizer.encode(config.prompt)],
        dtype=torch.long,
        device=device,
    )
    generated_ids = model.generate(
        input_ids=prompt_ids,
        max_new_tokens=config.max_new_tokens,
    )
    generated_text = tokenizer.decode(generated_ids[0].cpu().tolist())

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    log_path = output_dir / "loss_history.jsonl"
    tokenizer_path = output_dir / "tokenizer.json"
    sample_path = output_dir / "sample.txt"
    checkpoint_path = output_dir / "model.pt"

    write_json(
        path=config_path,
        payload={
            **asdict(config),
            "resolved_device": str(device),
            "tokenizer_type": config.tokenizer_type,
            "bpe_tokenizer_path": (
                config.bpe_tokenizer_path
                if config.tokenizer_type == "bpe"
                else None
            ),
            "vocab_size": tokenizer.vocab_size,
            "train_tokens": int(train_tokens.numel()),
            "validation_tokens": int(validation_tokens.numel()),
        },
    )
    write_jsonl(log_path, history)
    save_run_tokenizer(tokenizer_path, tokenizer)
    sample_path.write_text(generated_text, encoding="utf-8")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "vocab_size": tokenizer.vocab_size,
        },
        checkpoint_path,
    )

    return {
        "config_path": config_path,
        "log_path": log_path,
        "tokenizer_path": tokenizer_path,
        "sample_path": sample_path,
        "checkpoint_path": checkpoint_path,
        "history": history,
    }
