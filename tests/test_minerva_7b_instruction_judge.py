from types import SimpleNamespace

import pytest
import torch

from sonnet_evaluation.minerva_7b_instruction_judge import (
    build_instruction_judge_prompt,
    evaluate_instruction_judge,
    parse_instruction_judge_response,
    score_instruction_judge_cases,
)


class CharacterJudgeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return f"<chat>{messages[0]['content']}<assistant>"

    def __call__(self, text, *, add_special_tokens, return_tensors):
        assert add_special_tokens is False
        assert return_tensors == "pt"
        return {
            "input_ids": torch.tensor([[ord(char) for char in text]]),
            "attention_mask": torch.ones((1, len(text)), dtype=torch.long),
        }

    def decode(self, token_ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        return "".join(chr(token_id) for token_id in token_ids)


class FixedJudgeModel:
    def __init__(self, response):
        self.response = response
        self.training = True

    def eval(self):
        self.training = False
        return self

    def generate(self, *, input_ids, attention_mask, max_new_tokens, do_sample,
                 pad_token_id, eos_token_id):
        assert attention_mask.shape == input_ids.shape
        assert max_new_tokens == 96
        assert do_sample is False
        assert pad_token_id == 0
        assert eos_token_id == 1
        continuation = torch.tensor(
            [[ord(char) for char in self.response]], dtype=torch.long
        )
        return torch.cat([input_ids, continuation], dim=1)


def _text(line_count=14):
    return "\n".join(f"Verso numero {index}" for index in range(line_count))


def test_instruction_judge_prompt_hides_labels_and_locks_schema():
    prompt = build_instruction_judge_prompt(CharacterJudgeTokenizer(), _text())
    assert "<testo>" in prompt
    assert '"grammatica":0' in prompt
    assert "generally grammatical" not in prompt
    assert "<testo>" in build_instruction_judge_prompt(
        CharacterJudgeTokenizer(), _text(8)
    )


def test_instruction_judge_parser_accepts_exact_json_and_fence():
    expected = {"grammatica": 4, "tema": 3, "stabilita": 2}
    assert parse_instruction_judge_response(
        '{"grammatica":4,"tema":3,"stabilita":2}'
    ) == expected
    assert parse_instruction_judge_response(
        '```json\n{"grammatica":4,"tema":3,"stabilita":2}\n```'
    ) == expected

    with pytest.raises(ValueError, match="exactly"):
        parse_instruction_judge_response(
            '{"grammatica":4,"tema":3,"stabilita":2,"altro":1}'
        )
    with pytest.raises(ValueError, match="integer"):
        parse_instruction_judge_response(
            '{"grammatica":true,"tema":3,"stabilita":2}'
        )


def test_instruction_judge_scoring_records_parse_failures_without_labels_in_prompt():
    cases = [{
        "case_id": "human:abc",
        "blind_id": "abc",
        "grammar": True,
        "topic": True,
        "collapse": False,
        "text": _text(),
    }]
    rows = score_instruction_judge_cases(
        model=FixedJudgeModel('{"grammatica":4,"tema":4,"stabilita":4}'),
        tokenizer=CharacterJudgeTokenizer(),
        cases=cases,
        device="cpu",
        max_new_tokens=96,
    )
    assert rows[0]["parsed"] is True
    assert rows[0]["grammar_score"] == 4
    assert "text" not in rows[0]


def test_instruction_judge_gate_and_remote_policy_require_alignment():
    rows = []
    for index in range(56):
        grammar = index < 9
        topic = index < 54
        collapse = index >= 42
        rows.append({
            "parsed": True,
            "grammar": grammar,
            "topic": topic,
            "collapse": collapse,
            "grammar_score": 4 if grammar else 0,
            "topic_score": 4 if topic else 0,
            "stability_score": 0 if collapse else 4,
        })
    thresholds = {
        "parse_rate": 0.98,
        "grammar_auroc": 0.75,
        "topic_auroc": 0.7,
        "noncollapse_auroc": 0.75,
        "human_ordinal_pairwise_concordance": 0.65,
    }
    remote = {
        "required_checks": ["parse_rate", "noncollapse_auroc"],
        "minimum_total_passed_checks": 4,
    }
    result = evaluate_instruction_judge(
        scored_cases=rows,
        thresholds=thresholds,
        remote_policy=remote,
    )
    assert result["gate_passed"] is True
    assert result["remote_fp16_authorized"] is True

    for row in rows:
        row["stability_score"] = 4 if row["collapse"] else 0
    failed = evaluate_instruction_judge(
        scored_cases=rows,
        thresholds=thresholds,
        remote_policy=remote,
    )
    assert failed["gate_passed"] is False
    assert failed["remote_fp16_authorized"] is False
