"""Evaluate two selected sonnet models on their exact shared test poems."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from sonnet_corpus.bpe import BytePairEncodingTokenizer
from sonnet_corpus.dataset_text import (
    load_poem_text,
    read_manifest_rows,
    select_manifest_rows,
    validate_manifest_rows,
)
from sonnet_corpus.pretraining_tokenizer import encode_text_by_pretoken
from sonnet_model.transformer import CausalTransformerLanguageModel
from sonnet_evaluation.generation import (
    load_tokenizer,
    load_transformer_from_checkpoint,
)


ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class EvaluationArm:
    """One selected corpus-scaling fine-tuning run and its source manifest."""

    label: str
    run_dir: Path
    selection_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class SharedTestRecord:
    """One poem held out by both corpus versions with identical cleaned text."""

    poem_id: str
    text: str


def evaluate_shared_sonnet_test(
    *,
    repo_root: Path,
    left_arm: EvaluationArm,
    right_arm: EvaluationArm,
    dataset: str,
    context_length: int,
    device: torch.device | str,
    per_poem_output_path: Path,
    report_output_path: Path,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Score two selected models on poems held out by both source manifests."""
    if left_arm.label == right_arm.label:
        raise ValueError("evaluation-arm labels must differ")
    if context_length <= 0:
        raise ValueError("context_length must be greater than 0")

    records = load_shared_test_records(
        repo_root=repo_root,
        left_manifest_path=left_arm.manifest_path,
        right_manifest_path=right_arm.manifest_path,
        dataset=dataset,
    )
    _write_progress(
        progress,
        f"loaded {len(records)} shared held-out poems",
    )
    left_loaded = load_evaluation_arm(
        repo_root=repo_root,
        arm=left_arm,
        device=device,
    )
    right_loaded = load_evaluation_arm(
        repo_root=repo_root,
        arm=right_arm,
        device=device,
    )
    _validate_matching_tokenizers(left_loaded["tokenizer"], right_loaded["tokenizer"])
    _validate_matching_architectures(
        left_loaded["selection"],
        right_loaded["selection"],
    )

    left_scores = score_shared_test_records(
        model=left_loaded["model"],
        tokenizer=left_loaded["tokenizer"],
        records=records,
        context_length=context_length,
        device=device,
        label=left_arm.label,
        progress=progress,
    )
    right_scores = score_shared_test_records(
        model=right_loaded["model"],
        tokenizer=right_loaded["tokenizer"],
        records=records,
        context_length=context_length,
        device=device,
        label=right_arm.label,
        progress=progress,
    )
    result = build_shared_test_result(
        dataset=dataset,
        context_length=context_length,
        records=records,
        left_arm=left_arm,
        right_arm=right_arm,
        left_loaded=left_loaded,
        right_loaded=right_loaded,
        left_scores=left_scores,
        right_scores=right_scores,
    )
    write_shared_test_outputs(
        result=result,
        per_poem_output_path=per_poem_output_path,
        report_output_path=report_output_path,
        per_poem_output_reference=_portable_path(per_poem_output_path, repo_root),
    )
    _write_progress(progress, f"wrote per-poem results: {per_poem_output_path}")
    _write_progress(progress, f"wrote shared-test report: {report_output_path}")
    return result


def load_shared_test_records(
    *,
    repo_root: Path,
    left_manifest_path: Path,
    right_manifest_path: Path,
    dataset: str,
) -> list[SharedTestRecord]:
    """Return only poem IDs held out in both manifests with matching text."""
    left_rows = _test_rows_by_poem_id(
        manifest_path=_resolve_path(repo_root, left_manifest_path),
        dataset=dataset,
    )
    right_rows = _test_rows_by_poem_id(
        manifest_path=_resolve_path(repo_root, right_manifest_path),
        dataset=dataset,
    )
    shared_poem_ids = sorted(set(left_rows) & set(right_rows))
    if not shared_poem_ids:
        raise ValueError("the two manifests have no shared held-out poems")

    records = []
    for poem_id in shared_poem_ids:
        left_text = load_poem_text(left_rows[poem_id], repo_root)
        right_text = load_poem_text(right_rows[poem_id], repo_root)
        if left_text != right_text:
            raise ValueError(
                "shared held-out poem text differs between manifests: " + poem_id
            )
        records.append(SharedTestRecord(poem_id=poem_id, text=left_text))
    return records


def load_evaluation_arm(
    *,
    repo_root: Path,
    arm: EvaluationArm,
    device: torch.device | str,
) -> dict[str, Any]:
    """Load one selected checkpoint after verifying its declared manifest lineage."""
    run_dir = _resolve_path(repo_root, arm.run_dir)
    selection_path = _resolve_path(repo_root, arm.selection_path)
    manifest_path = _resolve_path(repo_root, arm.manifest_path)
    config_path = run_dir / "config.json"
    tokenizer_path = run_dir / "tokenizer.json"
    for path in (config_path, selection_path, tokenizer_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"evaluation input does not exist: {path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    _validate_manifest_lineage(
        config=config,
        expected_manifest_path=manifest_path,
        repo_root=repo_root,
        label=arm.label,
    )
    checkpoint_path = _resolve_path(
        repo_root,
        Path(selection["selected_checkpoint_path"]),
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"selected checkpoint does not exist: {checkpoint_path}"
        )
    if int(selection["selected_checkpoint_step"]) != int(
        config["best_validation_step"]
    ):
        raise ValueError(
            f"{arm.label} selection does not use the recorded best validation step"
        )
    if not bool(selection["exact_best_checkpoint_available"]):
        raise ValueError(
            f"{arm.label} must provide an exact best-validation checkpoint"
        )

    tokenizer = load_tokenizer(tokenizer_path)
    if not isinstance(tokenizer, BytePairEncodingTokenizer):
        raise ValueError(f"{arm.label} must use a Unicode BPE tokenizer")
    if tokenizer.vocab_size != int(config["vocab_size"]):
        raise ValueError(f"{arm.label} tokenizer vocabulary does not match its run")
    model = load_transformer_from_checkpoint(
        checkpoint_path=checkpoint_path,
        config_path=selection_path,
        device=device,
    )
    return {
        "config": config,
        "selection": selection,
        "tokenizer": tokenizer,
        "model": model,
        "checkpoint_path": checkpoint_path,
    }


def score_shared_test_records(
    *,
    model: CausalTransformerLanguageModel,
    tokenizer: BytePairEncodingTokenizer,
    records: list[SharedTestRecord],
    context_length: int,
    device: torch.device | str,
    label: str,
    progress: ProgressCallback | None = None,
) -> list[dict[str, float | int | str]]:
    """Score each poem independently without scoring a document separator."""
    if context_length > model.max_context_length:
        raise ValueError("context_length exceeds the selected model's context limit")

    scores = []
    report_interval = max(1, len(records) // 10)
    model.eval()
    for index, record in enumerate(records, start=1):
        token_ids = torch.tensor(
            encode_text_by_pretoken(record.text, tokenizer),
            dtype=torch.long,
        )
        negative_log_likelihood, target_token_count = score_token_ids(
            model=model,
            token_ids=token_ids,
            context_length=context_length,
            device=device,
        )
        scores.append({
            "poem_id": record.poem_id,
            "target_token_count": target_token_count,
            "negative_log_likelihood": negative_log_likelihood,
            "loss": negative_log_likelihood / target_token_count,
        })
        if index % report_interval == 0 or index == len(records):
            _write_progress(progress, f"scored {label} poem {index}/{len(records)}")
    return scores


@torch.inference_mode()
def score_token_ids(
    *,
    model: CausalTransformerLanguageModel,
    token_ids: torch.Tensor,
    context_length: int,
    device: torch.device | str,
) -> tuple[float, int]:
    """Return summed next-token NLL over one token sequence in context chunks."""
    if token_ids.ndim != 1:
        raise ValueError("token_ids must be a 1D tensor")
    if token_ids.dtype != torch.long:
        raise ValueError("token_ids must have dtype torch.long")
    if len(token_ids) < 2:
        raise ValueError("a scored poem must contain at least two tokens")
    if context_length <= 0:
        raise ValueError("context_length must be greater than 0")

    total_negative_log_likelihood = 0.0
    total_target_tokens = 0
    for start in range(0, len(token_ids) - 1, context_length):
        end = min(start + context_length, len(token_ids) - 1)
        input_ids = token_ids[start:end].unsqueeze(0).to(device)
        target_ids = token_ids[start + 1:end + 1].unsqueeze(0).to(device)
        _, loss = model(input_ids, target_ids)
        if loss is None:
            raise RuntimeError("selected model did not return a scoring loss")
        target_token_count = int(target_ids.numel())
        total_negative_log_likelihood += float(loss) * target_token_count
        total_target_tokens += target_token_count
    return total_negative_log_likelihood, total_target_tokens


def build_shared_test_result(
    *,
    dataset: str,
    context_length: int,
    records: list[SharedTestRecord],
    left_arm: EvaluationArm,
    right_arm: EvaluationArm,
    left_loaded: dict[str, Any],
    right_loaded: dict[str, Any],
    left_scores: list[dict[str, float | int | str]],
    right_scores: list[dict[str, float | int | str]],
) -> dict[str, Any]:
    """Combine matched per-poem scores into weighted arm-level measurements."""
    if [score["poem_id"] for score in left_scores] != [
        score["poem_id"] for score in right_scores
    ]:
        raise ValueError("the two arms did not score the same poems in the same order")

    left_summary = _arm_score_summary(left_arm, left_loaded, left_scores)
    right_summary = _arm_score_summary(right_arm, right_loaded, right_scores)
    per_poem = []
    for left_score, right_score in zip(left_scores, right_scores, strict=True):
        if left_score["target_token_count"] != right_score["target_token_count"]:
            raise ValueError("matching poems produced unequal target-token counts")
        per_poem.append({
            "poem_id": left_score["poem_id"],
            "target_token_count": left_score["target_token_count"],
            f"{left_arm.label}_negative_log_likelihood": (
                left_score["negative_log_likelihood"]
            ),
            f"{left_arm.label}_loss": left_score["loss"],
            f"{right_arm.label}_negative_log_likelihood": (
                right_score["negative_log_likelihood"]
            ),
            f"{right_arm.label}_loss": right_score["loss"],
        })
    return {
        "dataset": dataset,
        "context_length": context_length,
        "shared_test_poem_count": len(records),
        "shared_test_poem_ids_sha256": _poem_id_digest(records),
        "tokenizer_sha256": _tokenizer_digest(left_loaded["tokenizer"]),
        "arms": {
            left_arm.label: left_summary,
            right_arm.label: right_summary,
        },
        "per_poem": per_poem,
    }


def write_shared_test_outputs(
    *,
    result: dict[str, Any],
    per_poem_output_path: Path,
    report_output_path: Path,
    per_poem_output_reference: str,
) -> None:
    """Write machine-readable per-poem evidence and a public Markdown summary."""
    per_poem_output_path.parent.mkdir(parents=True, exist_ok=True)
    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    per_poem_output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_output_path.write_text(
        build_shared_test_markdown(result, per_poem_output_reference),
        encoding="utf-8",
    )


def build_shared_test_markdown(
    result: dict[str, Any],
    per_poem_output_reference: str,
) -> str:
    """Render the matched held-out comparison without mixing validation results."""
    arm_rows = []
    for label, arm in result["arms"].items():
        arm_rows.append(
            f"| {label} | {arm['selected_checkpoint_step']:,} | "
            f"{arm['selected_validation_loss']:.4f} | "
            f"{arm['target_token_count']:,} | {arm['loss']:.4f} | "
            f"{arm['perplexity']:.3f} |"
        )
    labels = list(result["arms"])
    left_arm = result["arms"][labels[0]]
    right_arm = result["arms"][labels[1]]
    loss_difference = right_arm["loss"] - left_arm["loss"]
    if loss_difference > 0:
        interpretation = (
            f"{labels[0]} has the lower shared-test loss by {loss_difference:.4f} "
            "nats per BPE token."
        )
    elif loss_difference < 0:
        interpretation = (
            f"{labels[1]} has the lower shared-test loss by {-loss_difference:.4f} "
            "nats per BPE token."
        )
    else:
        interpretation = "The two arms have equal shared-test loss at report precision."

    return "\n".join([
        "# Shared Sonnet Corpus Test Evaluation",
        "",
        "This comparison scores only poems held out by both corpus versions. "
        "Each poem is scored independently, so no artificial document separator "
        "or cross-poem context contributes to the result.",
        "",
        "## Evaluation Set",
        "",
        f"- Dataset selector: `{result['dataset']}`",
        f"- Shared held-out poems: {result['shared_test_poem_count']:,}",
        f"- Shared poem-ID SHA-256: `{result['shared_test_poem_ids_sha256']}`",
        f"- Context length: {result['context_length']}",
        f"- Shared tokenizer SHA-256: `{result['tokenizer_sha256']}`",
        "",
        "## Results",
        "",
        "| Arm | Selected step | Own validation loss | Shared target tokens | "
        "Shared test loss | Shared test perplexity |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        *arm_rows,
        "",
        "## Interpretation",
        "",
        interpretation,
        "Own validation losses are shown only to document checkpoint selection; "
        "they are not compared because the corpus versions use different "
        "validation sets. The shared-test loss is directly comparable because both "
        "arms score the same poems with the same tokenizer and token positions.",
        "",
        "## Per-Poem Evidence",
        "",
        f"Machine-readable per-poem losses: `{per_poem_output_reference}`.",
        "",
    ])


def _test_rows_by_poem_id(
    *,
    manifest_path: Path,
    dataset: str,
) -> dict[str, dict[str, str]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"sonnet manifest does not exist: {manifest_path}")
    rows = read_manifest_rows(manifest_path)
    validate_manifest_rows(rows, dataset)
    test_rows = select_manifest_rows(rows, dataset=dataset, split="test")
    by_poem_id = {row["poem_id"]: row for row in test_rows}
    if len(by_poem_id) != len(test_rows):
        raise ValueError("manifest contains duplicate selected test poem IDs")
    return by_poem_id


def _validate_manifest_lineage(
    *,
    config: dict[str, Any],
    expected_manifest_path: Path,
    repo_root: Path,
    label: str,
) -> None:
    recorded_path = _resolve_path(repo_root, Path(config["manifest_path"]))
    if recorded_path.resolve() != expected_manifest_path.resolve():
        raise ValueError(f"{label} run manifest does not match the requested manifest")
    actual_digest = _file_sha256(expected_manifest_path)
    if config.get("manifest_sha256") != actual_digest:
        raise ValueError(f"{label} run manifest SHA-256 does not match its manifest")


def _validate_matching_tokenizers(
    left: BytePairEncodingTokenizer,
    right: BytePairEncodingTokenizer,
) -> None:
    if left.to_dict() != right.to_dict():
        raise ValueError("the selected runs use different tokenizers")


def _validate_matching_architectures(
    left_selection: dict[str, Any],
    right_selection: dict[str, Any],
) -> None:
    if left_selection["model_architecture"] != right_selection["model_architecture"]:
        raise ValueError("the selected runs use different model architectures")


def _arm_score_summary(
    arm: EvaluationArm,
    loaded: dict[str, Any],
    scores: list[dict[str, float | int | str]],
) -> dict[str, Any]:
    total_target_tokens = sum(int(score["target_token_count"]) for score in scores)
    total_negative_log_likelihood = sum(
        float(score["negative_log_likelihood"])
        for score in scores
    )
    loss = total_negative_log_likelihood / total_target_tokens
    return {
        "run_dir": str(arm.run_dir),
        "manifest_path": str(arm.manifest_path),
        "manifest_sha256": loaded["config"]["manifest_sha256"],
        "checkpoint_path": str(loaded["checkpoint_path"]),
        "selected_checkpoint_step": int(
            loaded["selection"]["selected_checkpoint_step"]
        ),
        "selected_validation_loss": float(
            loaded["selection"]["best_validation_loss"]
        ),
        "target_token_count": total_target_tokens,
        "negative_log_likelihood": total_negative_log_likelihood,
        "loss": loss,
        "perplexity": math.exp(loss),
    }


def _resolve_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _portable_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tokenizer_digest(tokenizer: BytePairEncodingTokenizer) -> str:
    payload = json.dumps(
        tokenizer.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _poem_id_digest(records: list[SharedTestRecord]) -> str:
    poem_ids = "\n".join(record.poem_id for record in records)
    return hashlib.sha256(poem_ids.encode("utf-8")).hexdigest()


def _write_progress(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)
