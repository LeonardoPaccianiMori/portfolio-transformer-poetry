import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sonnet_evaluation.minerva_judge_gate import (
    JUDGE_HUMAN_CASE_COUNT,
    binary_auroc,
    build_candidate_windows,
    evaluate_judge_gate,
    load_judge_gate_config,
    ordinal_pairwise_concordance,
    parse_blinded_judgments,
    reverse_continuation_word_order,
    score_judge_cases,
    sha256_file,
    validate_judge_gate_config,
)


ROOT = Path(__file__).resolve().parents[1]


class FakeTokenizer:
    bos_token_id = 1

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [[2 + (ord(char) % 47) for char in text]]}


class ConstantLossModel(torch.nn.Module):
    def forward(self, *, input_ids, labels, use_cache):
        assert input_ids.shape == labels.shape
        assert use_cache is False
        assert (labels != -100).any()
        return SimpleNamespace(loss=torch.tensor(2.5, device=input_ids.device))


def test_frozen_judge_gate_config_is_valid():
    config = load_judge_gate_config(ROOT / "configs/minerva_3b_judge_gate.json")
    assert config["final_test_allowed"] is False
    assert config["control_generation"]["seed"] == 2027


def test_judge_gate_config_rejects_threshold_changes():
    config = json.loads(
        (ROOT / "configs/minerva_3b_judge_gate.json").read_text(encoding="utf-8")
    )
    config["thresholds"]["grammar_auroc"] = 0.1
    with pytest.raises(ValueError, match="thresholds"):
        validate_judge_gate_config(config)


def test_reverse_continuation_word_order_preserves_opening_and_lines():
    text = "\n".join(
        ["Exact opening"]
        + [f"word{i} alpha beta gamma" for i in range(1, 14)]
    )
    corrupted = reverse_continuation_word_order(text)
    lines = corrupted.splitlines()
    assert len(lines) == 14
    assert lines[0] == "Exact opening"
    assert lines[1] == "gamma beta alpha word1"


def test_parse_real_blinded_judgments_preserves_fixed_counts():
    judgments = parse_blinded_judgments(
        ROOT / "reports/minerva_3b_validation_sanity_blinded_judgments.md"
    )
    assert len(judgments) == JUDGE_HUMAN_CASE_COUNT
    assert sum(row["grammar"] for row in judgments.values()) == 9
    assert sum(row["topic"] for row in judgments.values()) == 54
    assert sum(row["collapse"] for row in judgments.values()) == 14


def test_candidate_windows_mask_prompt_and_cover_every_continuation_token():
    tokenizer = FakeTokenizer()
    text = "opening line\nfirst continuation line\nsecond continuation line"
    windows = build_candidate_windows(
        tokenizer=tokenizer,
        text=text,
        context_length=16,
    )
    expected_targets = len(
        tokenizer(
            "first continuation line\nsecond continuation line",
            add_special_tokens=False,
        )["input_ids"][0]
    )
    assert sum(target_count for _, _, target_count in windows) == expected_targets
    assert all((labels[:, 0] == -100).all() for _, labels, _ in windows)
    assert all(input_ids.shape == labels.shape for input_ids, labels, _ in windows)


def test_score_judge_cases_uses_negative_mean_continuation_nll():
    rows = score_judge_cases(
        model=ConstantLossModel(),
        tokenizer=FakeTokenizer(),
        cases=[{
            "case_id": "triplet:p:genuine",
            "family": "triplet",
            "variant": "genuine",
            "prompt_id": "p",
            "text": "opening\na sufficiently long continuation line",
        }],
        device="cpu",
        context_length=16,
    )
    assert rows[0]["mean_continuation_nll"] == pytest.approx(2.5)
    assert rows[0]["judge_score"] == pytest.approx(-2.5)
    assert "text" not in rows[0]
    assert rows[0]["target_token_count"] > 0


def test_binary_auroc_and_concordance_give_half_credit_to_ties():
    assert binary_auroc([1.0, 0.0], [True, False]) == 1.0
    assert binary_auroc([0.0, 0.0], [True, False]) == 0.5
    assert ordinal_pairwise_concordance([1.0, 0.0], [2, 1]) == 1.0
    assert ordinal_pairwise_concordance([0.0, 0.0], [2, 1]) == 0.5


def test_evaluate_judge_gate_requires_all_six_checks():
    scored = []
    for index in range(8):
        prompt_id = f"prompt_{index}"
        for variant, score in (("genuine", 3.0), ("generated", 2.0), ("corrupted", 1.0)):
            scored.append({
                "family": "triplet",
                "prompt_id": prompt_id,
                "variant": variant,
                "judge_score": score,
            })
    for index in range(56):
        grammar = index < 9
        topic = index < 54
        collapse = index >= 42
        quality = 2 * int(grammar) + int(topic) + 2 * int(not collapse)
        scored.append({
            "family": "human",
            "grammar": grammar,
            "topic": topic,
            "collapse": collapse,
            "judge_score": float(quality),
        })
    thresholds = {
        "genuine_over_corrupted_accuracy": 0.875,
        "genuine_over_generated_accuracy": 0.75,
        "generated_over_corrupted_accuracy": 0.625,
        "grammar_auroc": 0.7,
        "noncollapse_auroc": 0.65,
        "human_ordinal_pairwise_concordance": 0.65,
    }
    result = evaluate_judge_gate(scored_cases=scored, thresholds=thresholds)
    assert result["gate_passed"] is True
    assert len(result["checks"]) == 6

    for prompt_index in range(4):
        row_start = prompt_index * 3
        scored[row_start]["judge_score"] = 0.0
        scored[row_start + 1]["judge_score"] = 0.0
        scored[row_start + 2]["judge_score"] = 4.0
    failed = evaluate_judge_gate(scored_cases=scored, thresholds=thresholds)
    assert failed["gate_passed"] is False


def test_sha256_file_streams_artifact(tmp_path):
    path = tmp_path / "artifact.txt"
    path.write_text("judge gate\n", encoding="utf-8")
    assert sha256_file(path) == "a0bbe8ca5fd0cf11e665a738a983bba6cd25d736116798cba4a9caca26c00771"
