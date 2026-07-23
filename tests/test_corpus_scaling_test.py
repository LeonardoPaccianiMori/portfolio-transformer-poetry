import csv
import hashlib
import json
from pathlib import Path

import pytest
import torch

from sonnet_corpus.bpe import BytePairEncodingTokenizer, train_bpe_tokenizer
from sonnet_evaluation.corpus_scaling_test import (
    EvaluationArm,
    evaluate_shared_sonnet_test,
    load_shared_test_records,
    score_token_ids,
)
from sonnet_model.transformer import CausalTransformerLanguageModel


def write_manifest(
    path: Path,
    records: list[tuple[str, str]],
) -> None:
    rows = [
        {
            "poem_id": poem_id,
            "clean_text_path": f"data/processed/poems/{poem_id}.txt",
            "include_in_expanded_with_petrarch": "True",
            "split_expanded_with_petrarch": split,
        }
        for poem_id, split in records
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def build_tiny_model(vocab_size: int) -> CausalTransformerLanguageModel:
    return CausalTransformerLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        head_dim=4,
        feed_forward_dim=16,
        max_context_length=8,
    )


def write_fixture(repo_root: Path) -> tuple[EvaluationArm, EvaluationArm]:
    poems = {
        "shared_one": "Amor\n",
        "shared_two": "Donna\n",
        "v4_only": "Core\n",
        "v5_only": "Luce\n",
    }
    poems_dir = repo_root / "data" / "processed" / "poems"
    poems_dir.mkdir(parents=True)
    for poem_id, text in poems.items():
        (poems_dir / f"{poem_id}.txt").write_text(text, encoding="utf-8")

    v4_manifest = repo_root / "data" / "metadata" / "v4.csv"
    v5_manifest = repo_root / "data" / "metadata" / "v5.csv"
    write_manifest(
        v4_manifest,
        [
            ("shared_one", "test"),
            ("shared_two", "test"),
            ("v4_only", "test"),
        ],
    )
    write_manifest(
        v5_manifest,
        [
            ("shared_one", "test"),
            ("shared_two", "test"),
            ("v5_only", "test"),
        ],
    )
    tokenizer = train_bpe_tokenizer(
        texts=list(poems.values()),
        base_texts=list(poems.values()),
        vocab_size=16,
    )
    architecture = {
        "vocab_size": tokenizer.vocab_size,
        "embedding_dim": 8,
        "num_layers": 1,
        "num_heads": 2,
        "head_dim": 4,
        "feed_forward_dim": 16,
        "max_context_length": 8,
    }
    arms = []
    for label, manifest_path, favored_token_id in (
        ("v4", v4_manifest, 0),
        ("v5", v5_manifest, 1),
    ):
        run_dir = repo_root / "runs" / label
        run_dir.mkdir(parents=True)
        tokenizer.save(run_dir / "tokenizer.json")
        model = build_tiny_model(tokenizer.vocab_size)
        with torch.no_grad():
            model.output_projection.bias[favored_token_id] = 1.0
        checkpoint_path = run_dir / "best_validation.pt"
        torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
        manifest_path_text = str(manifest_path.relative_to(repo_root))
        config = {
            "manifest_path": manifest_path_text,
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "vocab_size": tokenizer.vocab_size,
            "best_validation_step": 10,
            "best_validation_loss": 2.0,
        }
        write_json(run_dir / "config.json", config)
        selection_path = run_dir / "selected_checkpoint.json"
        write_json(
            selection_path,
            {
                "selected_checkpoint_path": str(checkpoint_path.relative_to(repo_root)),
                "selected_checkpoint_step": 10,
                "best_validation_loss": 2.0,
                "exact_best_checkpoint_available": True,
                "model_architecture": architecture,
            },
        )
        arms.append(
            EvaluationArm(
                label=label,
                run_dir=run_dir.relative_to(repo_root),
                selection_path=selection_path.relative_to(repo_root),
                manifest_path=manifest_path.relative_to(repo_root),
            )
        )
    return arms[0], arms[1]


def test_shared_evaluation_scores_only_matching_held_out_poems(tmp_path):
    v4_arm, v5_arm = write_fixture(tmp_path)
    per_poem_path = tmp_path / "reports" / "per_poem.json"
    report_path = tmp_path / "reports" / "shared_test.md"
    progress = []

    result = evaluate_shared_sonnet_test(
        repo_root=tmp_path,
        left_arm=v4_arm,
        right_arm=v5_arm,
        dataset="expanded_with_petrarch",
        context_length=8,
        device="cpu",
        per_poem_output_path=per_poem_path,
        report_output_path=report_path,
        progress=progress.append,
    )

    assert result["shared_test_poem_count"] == 2
    assert [row["poem_id"] for row in result["per_poem"]] == [
        "shared_one",
        "shared_two",
    ]
    assert result["arms"]["v4"]["target_token_count"] == result["arms"]["v5"][
        "target_token_count"
    ]
    assert result["arms"]["v4"]["loss"] > 0.0
    assert result["arms"]["v5"]["perplexity"] > 1.0
    assert per_poem_path.is_file()
    assert "Shared Sonnet Corpus Test Evaluation" in report_path.read_text(
        encoding="utf-8"
    )
    assert any(message == "scored v4 poem 2/2" for message in progress)
    assert any(message == "scored v5 poem 2/2" for message in progress)


def test_shared_record_loader_excludes_poems_not_held_out_by_both(tmp_path):
    v4_arm, v5_arm = write_fixture(tmp_path)
    v5_manifest = tmp_path / v5_arm.manifest_path
    rows = list(csv.DictReader(v5_manifest.open(encoding="utf-8", newline="")))
    for row in rows:
        if row["poem_id"] == "shared_two":
            row["split_expanded_with_petrarch"] = "train"
    with v5_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    records = load_shared_test_records(
        repo_root=tmp_path,
        left_manifest_path=v4_arm.manifest_path,
        right_manifest_path=v5_arm.manifest_path,
        dataset="expanded_with_petrarch",
    )

    assert [record.poem_id for record in records] == ["shared_one"]


def test_shared_record_loader_rejects_same_id_with_different_text(tmp_path):
    v4_arm, v5_arm = write_fixture(tmp_path)
    v5_manifest = tmp_path / v5_arm.manifest_path
    v5_rows = list(csv.DictReader(v5_manifest.open(encoding="utf-8", newline="")))
    shared_one_row = next(row for row in v5_rows if row["poem_id"] == "shared_one")
    replacement_path = tmp_path / "data" / "processed" / "poems" / "changed.txt"
    replacement_path.write_text("Mutato\n", encoding="utf-8")
    shared_one_row["clean_text_path"] = "data/processed/poems/changed.txt"
    with v5_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(v5_rows[0]))
        writer.writeheader()
        writer.writerows(v5_rows)

    with pytest.raises(ValueError, match="shared held-out poem text differs"):
        load_shared_test_records(
            repo_root=tmp_path,
            left_manifest_path=v4_arm.manifest_path,
            right_manifest_path=v5_arm.manifest_path,
            dataset="expanded_with_petrarch",
        )


def test_shared_evaluation_rejects_manifest_lineage_mismatch(tmp_path):
    v4_arm, v5_arm = write_fixture(tmp_path)
    wrong_v4_arm = EvaluationArm(
        label=v4_arm.label,
        run_dir=v4_arm.run_dir,
        selection_path=v4_arm.selection_path,
        manifest_path=v5_arm.manifest_path,
    )

    with pytest.raises(ValueError, match="v4 run manifest does not match"):
        evaluate_shared_sonnet_test(
            repo_root=tmp_path,
            left_arm=wrong_v4_arm,
            right_arm=v5_arm,
            dataset="expanded_with_petrarch",
            context_length=8,
            device="cpu",
            per_poem_output_path=tmp_path / "per_poem.json",
            report_output_path=tmp_path / "report.md",
        )


def test_shared_evaluation_rejects_tokenizer_mismatch(tmp_path):
    v4_arm, v5_arm = write_fixture(tmp_path)
    tokenizer_path = tmp_path / v5_arm.run_dir / "tokenizer.json"
    tokenizer = BytePairEncodingTokenizer.load(tokenizer_path)
    token_to_id = dict(tokenizer.token_to_id)
    first_token, second_token = list(token_to_id)[:2]
    token_to_id[first_token], token_to_id[second_token] = (
        token_to_id[second_token],
        token_to_id[first_token],
    )
    BytePairEncodingTokenizer(
        token_to_id=token_to_id,
        merges=tokenizer.merges,
        special_tokens=tokenizer.special_tokens,
    ).save(tokenizer_path)

    with pytest.raises(ValueError, match="different tokenizers"):
        evaluate_shared_sonnet_test(
            repo_root=tmp_path,
            left_arm=v4_arm,
            right_arm=v5_arm,
            dataset="expanded_with_petrarch",
            context_length=8,
            device="cpu",
            per_poem_output_path=tmp_path / "per_poem.json",
            report_output_path=tmp_path / "report.md",
        )


def test_score_token_ids_covers_all_next_tokens_across_context_chunks():
    model = build_tiny_model(vocab_size=8)
    token_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.long)

    negative_log_likelihood, target_token_count = score_token_ids(
        model=model,
        token_ids=token_ids,
        context_length=3,
        device="cpu",
    )

    assert target_token_count == len(token_ids) - 1
    assert negative_log_likelihood > 0.0
