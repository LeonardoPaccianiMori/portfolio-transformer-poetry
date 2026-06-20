import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from sonnet_corpus.dataset_text import load_encoded_splits
from sonnet_model.bigram import BigramLanguageModel
from sonnet_training.steps import train_next_token_model


@dataclass(frozen=True)
class BigramTrainingConfig:
    dataset: str = "expanded_with_petrarch"
    batch_size: int = 64
    context_length: int = 64
    train_steps: int = 200
    eval_interval: int = 50
    eval_batches: int = 10
    learning_rate: float = 1e-2
    seed: int = 1337
    prompt: str = "Amor"
    max_new_tokens: int = 400
    device: str = "auto"


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


def train_bigram_run(
    repo_root: Path,
    output_dir: Path,
    config: BigramTrainingConfig,
) -> dict[str, Path | list[dict[str, float | int]]]:
    torch.manual_seed(config.seed)

    device = resolve_device(config.device)
    manifest_path = repo_root / "data" / "metadata" / "poems_manifest.csv"

    train_tokens, validation_tokens, _, tokenizer = load_encoded_splits(
        manifest_path=manifest_path,
        repo_root=repo_root,
        dataset=config.dataset,
    )

    model = BigramLanguageModel(vocab_size=tokenizer.vocab_size).to(device)
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
            "vocab_size": tokenizer.vocab_size,
            "train_tokens": int(train_tokens.numel()),
            "validation_tokens": int(validation_tokens.numel()),
        },
    )
    write_jsonl(log_path, history)
    save_tokenizer(tokenizer_path, tokenizer.char_to_id)
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
