import json
from pathlib import Path
from typing import Any

import torch

from sonnet_corpus.bpe import BytePairEncodingTokenizer
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


def load_tokenizer(tokenizer_path: Path) -> CharTokenizer | BytePairEncodingTokenizer:
    payload = read_json(tokenizer_path)

    if not isinstance(payload, dict):
        raise ValueError("tokenizer file must contain a JSON object")

    tokenizer_type = payload.get("type")

    if tokenizer_type == "character":
        return CharTokenizer(char_to_id=payload["char_to_id"])

    if tokenizer_type == "unicode_bpe":
        return BytePairEncodingTokenizer.from_dict(payload)

    raise ValueError(f"unsupported tokenizer type: {tokenizer_type}")


def load_transformer_from_checkpoint(
    checkpoint_path: Path,
    config_path: Path,
    device: torch.device | str,
) -> CausalTransformerLanguageModel:
    config = read_json(config_path)

    if not isinstance(config, dict):
        raise ValueError("config file must contain a JSON object")

    model_config = config.get("model_architecture", config)
    model = CausalTransformerLanguageModel(
        vocab_size=model_config["vocab_size"],
        embedding_dim=model_config["embedding_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        head_dim=model_config["head_dim"],
        feed_forward_dim=model_config["feed_forward_dim"],
        max_context_length=model_config["max_context_length"],
        normalization_type=model_config.get("normalization_type", "layer_norm"),
        normalization_eps=float(model_config.get("normalization_eps", 1e-5)),
        position_encoding_type=model_config.get(
            "position_encoding_type",
            "learned_absolute",
        ),
        rope_theta=float(model_config.get("rope_theta", 10_000.0)),
        feed_forward_type=model_config.get("feed_forward_type", "relu"),
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
    tokenizer: CharTokenizer | BytePairEncodingTokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device | str,
    seed: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    stop_text: str | None = None,
    target_lines: int | None = None,
    suppress_stop_text_until_target_lines: bool = False,
) -> str:
    result = generate_text_result(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        device=device,
        seed=seed,
        temperature=temperature,
        top_k=top_k,
        stop_text=stop_text,
        target_lines=target_lines,
        suppress_stop_text_until_target_lines=suppress_stop_text_until_target_lines,
    )

    return result["text"]


def non_empty_line_count(text: str) -> int:
    return len([
        line
        for line in text.splitlines()
        if line.strip()
    ])


def completed_non_empty_line_count(text: str) -> int:
    if text == "":
        return 0

    if text.endswith(("\n", "\r")):
        completed_text = text
    else:
        completed_text = "\n".join(text.splitlines()[:-1])

    return non_empty_line_count(completed_text)


def line_count_stop_reached(text: str, target_lines: int | None) -> bool:
    if target_lines is None:
        return False

    if target_lines <= 0:
        raise ValueError("target_lines must be greater than 0")

    return completed_non_empty_line_count(text) >= target_lines


def validate_stop_text(
    tokenizer: CharTokenizer | BytePairEncodingTokenizer,
    stop_text: str | None,
) -> None:
    if stop_text is None:
        return

    if stop_text == "":
        raise ValueError("stop_text must not be empty")

    tokenizer.encode(stop_text)


def single_token_id_for_text(
    tokenizer: CharTokenizer | BytePairEncodingTokenizer,
    text: str,
) -> int:
    token_ids = tokenizer.encode(text)

    if len(token_ids) != 1:
        raise ValueError("suppressed stop text must encode to exactly one token")

    return token_ids[0]


def validate_stop_text_suppression(
    tokenizer: CharTokenizer | BytePairEncodingTokenizer,
    stop_text: str | None,
    target_lines: int | None,
    suppress_stop_text_until_target_lines: bool,
) -> int | None:
    if not suppress_stop_text_until_target_lines:
        return None

    if stop_text is None:
        raise ValueError("stop_text is required when suppressing stop text")

    if target_lines is None:
        raise ValueError("target_lines is required when suppressing stop text")

    return single_token_id_for_text(tokenizer, stop_text)


def stop_reason_for_text(
    text: str,
    stop_text: str | None,
    target_lines: int | None,
) -> str | None:
    if stop_text is not None and stop_text in text:
        return "stop_text"

    if line_count_stop_reached(text, target_lines):
        return "target_lines"

    return None


def generate_text_result(
    model: CausalTransformerLanguageModel,
    tokenizer: CharTokenizer | BytePairEncodingTokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device | str,
    seed: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    stop_text: str | None = None,
    target_lines: int | None = None,
    suppress_stop_text_until_target_lines: bool = False,
) -> dict[str, Any]:
    validate_stop_text(tokenizer, stop_text)
    suppressed_stop_token_id = validate_stop_text_suppression(
        tokenizer=tokenizer,
        stop_text=stop_text,
        target_lines=target_lines,
        suppress_stop_text_until_target_lines=suppress_stop_text_until_target_lines,
    )

    generator = torch.Generator(device=device).manual_seed(seed)
    input_ids = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long,
        device=device,
    )

    uses_text_stopping = stop_text is not None or target_lines is not None

    if not uses_text_stopping:
        generated_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            generator=generator,
            temperature=temperature,
            top_k=top_k,
        )

        return {
            "text": tokenizer.decode(generated_ids[0].cpu().tolist()),
            "stop_reason": "max_new_tokens",
            "generated_new_tokens": generated_ids.shape[1] - input_ids.shape[1],
        }

    generated_ids = input_ids
    generated_text = prompt
    stop_reason = stop_reason_for_text(
        text=prompt,
        stop_text=stop_text,
        target_lines=target_lines,
    )

    for _ in range(max_new_tokens):
        if stop_reason is not None:
            break

        forbidden_token_ids = None

        if (
            suppressed_stop_token_id is not None
            and not line_count_stop_reached(generated_text, target_lines)
        ):
            forbidden_token_ids = {suppressed_stop_token_id}

        generated_ids = model.generate(
            input_ids=generated_ids,
            max_new_tokens=1,
            generator=generator,
            temperature=temperature,
            top_k=top_k,
            forbidden_token_ids=forbidden_token_ids,
        )
        generated_text = tokenizer.decode(generated_ids[0].cpu().tolist())
        stop_reason = stop_reason_for_text(
            text=generated_text,
            stop_text=stop_text,
            target_lines=target_lines,
        )

    if stop_reason is None:
        stop_reason = "max_new_tokens"

    return {
        "text": tokenizer.decode(generated_ids[0].cpu().tolist()),
        "stop_reason": stop_reason,
        "generated_new_tokens": generated_ids.shape[1] - input_ids.shape[1],
    }


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
    temperature: float = 1.0,
    top_k: int | None = None,
    stop_text: str | None = None,
    target_lines: int | None = None,
    suppress_stop_text_until_target_lines: bool = False,
    checkpoint_path: Path | None = None,
    model_config_path: Path | None = None,
) -> dict[str, Any]:
    tokenizer = load_tokenizer(run_dir / "tokenizer.json")
    checkpoint_path = checkpoint_path or run_dir / "model.pt"
    model_config_path = model_config_path or run_dir / "config.json"
    model = load_transformer_from_checkpoint(
        checkpoint_path=checkpoint_path,
        config_path=model_config_path,
        device=device,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files = []

    for prompt_index, prompt in enumerate(prompts):
        generation_result = generate_text_result(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt["text"],
            max_new_tokens=max_new_tokens,
            device=device,
            seed=seed + prompt_index,
            temperature=temperature,
            top_k=top_k,
            stop_text=stop_text,
            target_lines=target_lines,
            suppress_stop_text_until_target_lines=suppress_stop_text_until_target_lines,
        )
        generated_text = generation_result["text"]
        output_path = output_dir / safe_prompt_filename(prompt["id"])
        output_path.write_text(generated_text, encoding="utf-8")
        generated_files.append({
            "prompt_id": prompt["id"],
            "prompt_text": prompt["text"],
            "path": str(output_path),
            "seed": seed + prompt_index,
            "stop_reason": generation_result["stop_reason"],
            "generated_new_tokens": generation_result["generated_new_tokens"],
        })

    metadata = {
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "model_config_path": str(model_config_path),
        "output_dir": str(output_dir),
        "max_new_tokens": max_new_tokens,
        "base_seed": seed,
        "device": str(device),
        "temperature": temperature,
        "top_k": top_k,
        "stop_text": stop_text,
        "target_lines": target_lines,
        "suppress_stop_text_until_target_lines": suppress_stop_text_until_target_lines,
        "generated_files": generated_files,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return metadata
