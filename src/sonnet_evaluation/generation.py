import json
from pathlib import Path
from typing import Any

import torch

from sonnet_corpus.tokenizer import CharTokenizer
from sonnet_model.transformer import CausalTransformerLanguageModel


def read_json(path: Path) -> dict[str, Any] | list[dict[str, str]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_char_tokenizer(tokenizer_path: Path) -> CharTokenizer:
    payload = read_json(tokenizer_path)

    if not isinstance(payload, dict):
        raise ValueError("tokenizer file must contain a JSON object")

    if payload.get("type") != "character":
        raise ValueError("only character tokenizers are supported")

    char_to_id = payload["char_to_id"]

    return CharTokenizer(char_to_id=char_to_id)


def load_transformer_from_checkpoint(
    checkpoint_path: Path,
    config_path: Path,
    device: torch.device | str,
) -> CausalTransformerLanguageModel:
    config = read_json(config_path)

    if not isinstance(config, dict):
        raise ValueError("config file must contain a JSON object")

    model = CausalTransformerLanguageModel(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        head_dim=config["head_dim"],
        feed_forward_dim=config["feed_forward_dim"],
        max_context_length=config["max_context_length"],
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def load_prompts(prompts_path: Path) -> list[dict[str, str]]:
    prompts = read_json(prompts_path)

    if not isinstance(prompts, list):
        raise ValueError("prompts file must contain a JSON list")

    for prompt in prompts:
        if not isinstance(prompt, dict):
            raise ValueError("each prompt must be a JSON object")

        if "id" not in prompt or "text" not in prompt:
            raise ValueError("each prompt must contain id and text")

    return prompts


def generate_text(
    model: CausalTransformerLanguageModel,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device | str,
    seed: int,
) -> str:
    generator = torch.Generator(device=device).manual_seed(seed)
    input_ids = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long,
        device=device,
    )
    generated_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        generator=generator,
    )

    return tokenizer.decode(generated_ids[0].cpu().tolist())


def safe_prompt_filename(prompt_id: str) -> str:
    safe_characters = [
        character
        if character.isalnum() or character in {"-", "_"}
        else "_"
        for character in prompt_id
    ]
    safe_name = "".join(safe_characters).strip("_")

    if safe_name == "":
        raise ValueError("prompt id must contain at least one filename-safe character")

    return f"{safe_name}.txt"


def generate_for_prompts(
    run_dir: Path,
    prompts: list[dict[str, str]],
    output_dir: Path,
    max_new_tokens: int,
    seed: int,
    device: torch.device | str,
) -> dict[str, Any]:
    tokenizer = load_char_tokenizer(run_dir / "tokenizer.json")
    model = load_transformer_from_checkpoint(
        checkpoint_path=run_dir / "model.pt",
        config_path=run_dir / "config.json",
        device=device,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files = []

    for prompt_index, prompt in enumerate(prompts):
        generated_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt["text"],
            max_new_tokens=max_new_tokens,
            device=device,
            seed=seed + prompt_index,
        )
        output_path = output_dir / safe_prompt_filename(prompt["id"])
        output_path.write_text(generated_text, encoding="utf-8")
        generated_files.append({
            "prompt_id": prompt["id"],
            "prompt_text": prompt["text"],
            "path": str(output_path),
            "seed": seed + prompt_index,
        })

    metadata = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "max_new_tokens": max_new_tokens,
        "base_seed": seed,
        "device": str(device),
        "generated_files": generated_files,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return metadata
