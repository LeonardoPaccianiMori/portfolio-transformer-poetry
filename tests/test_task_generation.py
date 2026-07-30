import json
from pathlib import Path

import pytest
import torch

from sonnet_corpus.bpe import BytePairEncodingTokenizer
from sonnet_corpus.task_format import (
    SONNET_CONTINUATION_TOKEN,
    SONNET_OPENING_TOKEN,
)
from sonnet_evaluation import task_generation
from sonnet_evaluation.task_generation import (
    END_OF_TEXT_TOKEN,
    generate_task_format_continuation,
    generate_task_format_for_prompts,
    load_task_format_prompts,
    validate_task_format_acceptance_configuration,
    validate_task_format_prompts_against_manifest,
)


def make_task_tokenizer() -> BytePairEncodingTokenizer:
    characters = sorted(set("Prima lineasecondaterza\n"))
    tokens = [
        END_OF_TEXT_TOKEN,
        SONNET_OPENING_TOKEN,
        SONNET_CONTINUATION_TOKEN,
        *characters,
    ]
    return BytePairEncodingTokenizer(
        token_to_id={token: index for index, token in enumerate(tokens)},
        merges=[],
        special_tokens=[
            END_OF_TEXT_TOKEN,
            SONNET_OPENING_TOKEN,
            SONNET_CONTINUATION_TOKEN,
        ],
    )


class ScriptedGenerator:
    def __init__(self, token_ids: list[int]):
        self.token_ids = iter(token_ids)
        self.forbidden_token_ids: list[set[int]] = []

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        generator: torch.Generator,
        temperature: float,
        top_k: int | None,
        forbidden_token_ids: set[int],
    ) -> torch.Tensor:
        assert max_new_tokens == 1
        assert temperature == 0.8
        assert top_k == 50
        self.forbidden_token_ids.append(forbidden_token_ids)
        next_token_id = next(self.token_ids)
        return torch.cat(
            [
                input_ids,
                torch.tensor([[next_token_id]], dtype=torch.long),
            ],
            dim=1,
        )


def write_manifest(path: Path) -> None:
    path.write_text(
        "poem_id,clean_text_path,include_in_expanded_with_petrarch,split_expanded_with_petrarch\n"
        "test_poem,data/processed/test.txt,True,test\n"
        "train_poem,data/processed/train.txt,True,train\n",
        encoding="utf-8",
    )


def write_strict_sonnet(path: Path, opening_line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join([opening_line, *[f"line {number}" for number in range(2, 15)]]),
        encoding="utf-8",
    )


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_generate_task_format_continuation_hides_markers_and_stops_at_target():
    tokenizer = make_task_tokenizer()
    continuation = "seconda\nterza\n"
    model = ScriptedGenerator(tokenizer.encode(continuation))

    result = generate_task_format_continuation(
        model=model,  # type: ignore[arg-type]
        tokenizer=tokenizer,
        opening_line="Prima linea",
        max_new_tokens=30,
        device=torch.device("cpu"),
        seed=1337,
        continuation_line_target=2,
    )

    assert result["text"] == "Prima linea\nseconda\nterza\n"
    assert result["stop_reason"] == "target_lines"
    assert result["completed_continuation_lines"] == 2
    assert result["generated_new_tokens"] == len(tokenizer.encode(continuation))
    assert SONNET_OPENING_TOKEN not in result["text"]
    assert SONNET_CONTINUATION_TOKEN not in result["text"]
    assert END_OF_TEXT_TOKEN not in result["text"]
    expected_forbidden_ids = {
        tokenizer.encode(END_OF_TEXT_TOKEN)[0],
        tokenizer.encode(SONNET_OPENING_TOKEN)[0],
        tokenizer.encode(SONNET_CONTINUATION_TOKEN)[0],
    }
    assert model.forbidden_token_ids
    assert all(ids == expected_forbidden_ids for ids in model.forbidden_token_ids)


def test_load_task_format_prompts_rejects_duplicate_poem_ids(tmp_path):
    prompts_path = tmp_path / "prompts.json"
    write_json(
        prompts_path,
        [
            {
                "id": "first",
                "poem_id": "poem",
                "opening_line": "Prima linea",
            },
            {
                "id": "second",
                "poem_id": "poem",
                "opening_line": "Seconda linea",
            },
        ],
    )

    with pytest.raises(ValueError, match="poem_ids"):
        load_task_format_prompts(prompts_path)


def test_acceptance_configuration_rejects_changed_decoding_settings():
    prompts = [
        {
            "id": f"prompt_{index}",
            "poem_id": f"poem_{index}",
            "opening_line": f"Prima linea {index}",
        }
        for index in range(10)
    ]

    with pytest.raises(ValueError, match="temperature"):
        validate_task_format_acceptance_configuration(
            prompts=prompts,
            seeds=[1337, 1338],
            temperature=1.0,
            top_k=50,
            continuation_line_target=13,
        )


def test_validate_task_format_prompts_requires_exact_held_out_opening_line(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(manifest_path)
    write_strict_sonnet(
        tmp_path / "data" / "processed" / "test.txt",
        "Prima linea",
    )
    write_strict_sonnet(
        tmp_path / "data" / "processed" / "train.txt",
        "Linea di addestramento",
    )
    prompts = [{
        "id": "first",
        "poem_id": "test_poem",
        "opening_line": "Prima linea",
    }]

    validate_task_format_prompts_against_manifest(
        prompts=prompts,
        manifest_path=manifest_path,
        repo_root=tmp_path,
        dataset="expanded_with_petrarch",
        split="test",
    )

    prompts[0]["opening_line"] = "Linea sbagliata"
    with pytest.raises(ValueError, match="does not match"):
        validate_task_format_prompts_against_manifest(
            prompts=prompts,
            manifest_path=manifest_path,
            repo_root=tmp_path,
            dataset="expanded_with_petrarch",
            split="test",
        )


def test_generate_task_format_for_prompts_writes_one_file_per_seed(
    tmp_path,
    monkeypatch,
):
    tokenizer = make_task_tokenizer()
    model = ScriptedGenerator(
        tokenizer.encode("seconda\n")
        + tokenizer.encode("terza\n")
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    output_dir = tmp_path / "outputs"
    monkeypatch.setattr(task_generation, "load_tokenizer", lambda _: tokenizer)
    monkeypatch.setattr(
        task_generation,
        "load_transformer_from_checkpoint",
        lambda **_: model,
    )

    metadata = generate_task_format_for_prompts(
        run_dir=run_dir,
        prompts=[{
            "id": "first",
            "poem_id": "poem",
            "author": "Author",
            "opening_line": "Prima linea",
        }],
        output_dir=output_dir,
        max_new_tokens=20,
        seeds=[1337, 1338],
        device=torch.device("cpu"),
        continuation_line_target=1,
    )

    assert (output_dir / "first__seed_1337.txt").read_text(encoding="utf-8") == (
        "Prima linea\nseconda\n"
    )
    assert (output_dir / "first__seed_1338.txt").read_text(encoding="utf-8") == (
        "Prima linea\nterza\n"
    )
    assert len(metadata["generated_files"]) == 2
    assert metadata["generated_files"][0]["source_prompt_id"] == "first"
    assert metadata["continuation_line_target"] == 1
