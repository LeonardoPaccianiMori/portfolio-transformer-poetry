"""Validation-only gate for using untouched Minerva 3B as a quality judge."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from sonnet_corpus.task_format import SonnetContinuationExample
from sonnet_training.minerva_qlora import (
    MINERVA_3B_MODEL_ID,
    MINERVA_3B_REVISION,
)


JUDGE_GATE_VERSION = "minerva_3b_judge_gate_v1"
JUDGE_SCORE_DEFINITION = "negative_mean_continuation_nll"
JUDGE_CONTEXT_LENGTH = 512
JUDGE_CONTROL_SEED = 2027
JUDGE_PROMPT_COUNT = 8
JUDGE_HUMAN_CASE_COUNT = 56
JUDGE_CORRUPTION = "reverse_word_order_in_each_continuation_line_v1"


def load_judge_gate_config(path: Path) -> dict[str, Any]:
    """Load and validate the immutable judge-gate configuration."""
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_judge_gate_config(config)
    return config


def validate_judge_gate_config(config: Mapping[str, Any]) -> None:
    """Reject changes that would turn the fixed gate into a result-driven test."""
    expected = {
        "gate_version": JUDGE_GATE_VERSION,
        "status": "predeclared_before_gpu_scoring",
        "model_id": MINERVA_3B_MODEL_ID,
        "revision": MINERVA_3B_REVISION,
        "model_precision": "float16",
        "context_length": JUDGE_CONTEXT_LENGTH,
        "score_definition": JUDGE_SCORE_DEFINITION,
        "corruption": JUDGE_CORRUPTION,
        "final_test_allowed": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"judge-gate configuration mismatch: {key}")

    generation = config.get("control_generation")
    if generation != {
        "seed": JUDGE_CONTROL_SEED,
        "temperature": 0.8,
        "top_k": 50,
        "max_new_tokens": 900,
        "continuation_line_target": 13,
    }:
        raise ValueError("judge-gate control generation is not the frozen recipe")

    if config.get("human_quality_weights") != {
        "grammar": 2,
        "topic": 1,
        "noncollapse": 2,
    }:
        raise ValueError("judge-gate human quality weights do not match")

    expected_thresholds = {
        "genuine_over_corrupted_accuracy": 0.875,
        "genuine_over_generated_accuracy": 0.75,
        "generated_over_corrupted_accuracy": 0.625,
        "grammar_auroc": 0.7,
        "noncollapse_auroc": 0.65,
        "human_ordinal_pairwise_concordance": 0.65,
    }
    if config.get("thresholds") != expected_thresholds:
        raise ValueError("judge-gate thresholds do not match the frozen gate")


def validate_judge_gate_artifacts(
    *, config: Mapping[str, Any], repo_root: Path
) -> None:
    """Verify every frozen gate input before generation or model scoring."""
    path_hash_fields = (
        ("manifest_path", "manifest_sha256"),
        ("validation_prompt_path", "validation_prompt_sha256"),
        ("from_scratch_checkpoint_path", "from_scratch_checkpoint_sha256"),
        ("human_mapping_path", "human_mapping_sha256"),
        ("human_judgments_path", "human_judgments_sha256"),
    )
    for path_key, hash_key in path_hash_fields:
        path = _resolve(repo_root, Path(str(config[path_key])))
        if not path.is_file():
            raise FileNotFoundError(f"judge-gate artifact is missing: {path}")
        if sha256_file(path) != config[hash_key]:
            raise ValueError(f"judge-gate artifact hash mismatch: {path_key}")


def reverse_continuation_word_order(text: str) -> str:
    """Corrupt syntax while preserving the opening, words, and line count."""
    lines = _nonempty_lines(text)
    if len(lines) != 14:
        raise ValueError("corruption controls require exactly fourteen lines")
    corrupted = [lines[0]]
    for line in lines[1:]:
        words = line.split()
        if len(words) < 2:
            raise ValueError("corruption controls require multiword continuation lines")
        corrupted.append(" ".join(reversed(words)))
    return "\n".join(corrupted)


def parse_blinded_judgments(path: Path) -> dict[str, dict[str, bool]]:
    """Read the fixed yes/no rubric table from the completed blinded review."""
    judgments: dict[str, dict[str, bool]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        blind_id = cells[0].strip("`")
        labels = cells[1:4]
        if any(label not in {"yes", "no"} for label in labels):
            continue
        if blind_id in judgments:
            raise ValueError(f"duplicate blinded judgment: {blind_id}")
        judgments[blind_id] = {
            "grammar": labels[0] == "yes",
            "topic": labels[1] == "yes",
            "collapse": labels[2] == "yes",
        }
    if len(judgments) != JUDGE_HUMAN_CASE_COUNT:
        raise ValueError(
            "judge gate requires exactly "
            f"{JUDGE_HUMAN_CASE_COUNT} blinded judgments"
        )
    return judgments


def build_judge_cases(
    *,
    repo_root: Path,
    prompts: Sequence[Mapping[str, str]],
    validation_examples: Sequence[SonnetContinuationExample],
    generated_dir: Path,
    human_mapping_path: Path,
    human_judgments_path: Path,
) -> list[dict[str, Any]]:
    """Build triplet and human-labelled cases without loading final-test data."""
    if len(prompts) != JUDGE_PROMPT_COUNT:
        raise ValueError(f"judge gate requires exactly {JUDGE_PROMPT_COUNT} prompts")
    prompt_ids = {str(prompt["id"]) for prompt in prompts}
    if len(prompt_ids) != JUDGE_PROMPT_COUNT:
        raise ValueError("judge-gate prompt IDs must be unique")

    examples_by_id = {example.poem_id: example for example in validation_examples}
    if any(example.split != "validation" for example in validation_examples):
        raise ValueError("judge gate may load only validation examples")

    cases: list[dict[str, Any]] = []
    for prompt in prompts:
        prompt_id = str(prompt["id"])
        poem_id = str(prompt["poem_id"])
        example = examples_by_id.get(poem_id)
        if example is None:
            raise ValueError(f"judge prompt is not in validation examples: {poem_id}")
        if example.opening_line != prompt["opening_line"]:
            raise ValueError(f"judge opening line mismatch: {prompt_id}")
        genuine_text = f"{example.opening_line}\n{example.continuation_text}"
        generated_path = generated_dir / f"{prompt_id}__seed_{JUDGE_CONTROL_SEED}.txt"
        generated_text = generated_path.read_text(encoding="utf-8").strip()
        generated_lines = _nonempty_lines(generated_text)
        if len(generated_lines) != 14 or generated_lines[0] != example.opening_line:
            raise ValueError(f"invalid generated judge control: {generated_path}")

        for variant, text in (
            ("genuine", genuine_text),
            ("generated", generated_text),
            ("corrupted", reverse_continuation_word_order(genuine_text)),
        ):
            cases.append({
                "case_id": f"triplet:{prompt_id}:{variant}",
                "family": "triplet",
                "variant": variant,
                "prompt_id": prompt_id,
                "poem_id": poem_id,
                "text": text,
            })

    judgments = parse_blinded_judgments(human_judgments_path)
    mapping = json.loads(human_mapping_path.read_text(encoding="utf-8"))
    if set(mapping) != set(judgments):
        raise ValueError("blinded mapping and judgment IDs differ")
    for blind_id in sorted(mapping):
        record = mapping[blind_id]
        source_prompt_id = str(record["prompt_id"]).split("__seed_", maxsplit=1)[0]
        if source_prompt_id not in prompt_ids:
            raise ValueError("human control references a non-validation prompt")
        output_path = _resolve_recorded_output_path(repo_root, Path(record["path"]))
        text = output_path.read_text(encoding="utf-8").strip()
        labels = judgments[blind_id]
        cases.append({
            "case_id": f"human:{blind_id}",
            "family": "human",
            "variant": str(record["condition_id"]),
            "prompt_id": source_prompt_id,
            "blind_id": blind_id,
            "grammar": labels["grammar"],
            "topic": labels["topic"],
            "collapse": labels["collapse"],
            "text": text,
        })

    expected_count = JUDGE_PROMPT_COUNT * 3 + JUDGE_HUMAN_CASE_COUNT
    if len(cases) != expected_count:
        raise ValueError(f"judge-gate case count must be {expected_count}")
    return cases


def build_candidate_windows(
    *, tokenizer: Any, text: str, context_length: int = JUDGE_CONTEXT_LENGTH
) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
    """Encode continuation-only loss windows with bounded left context."""
    if context_length < 8:
        raise ValueError("judge context_length must be at least 8")
    lines = _nonempty_lines(text)
    if len(lines) < 2:
        raise ValueError("judge candidate must contain an opening and continuation")
    prompt_ids = _token_ids(tokenizer, lines[0] + "\n")
    continuation_ids = _token_ids(tokenizer, "\n".join(lines[1:]))
    if not prompt_ids or not continuation_ids:
        raise ValueError("judge candidate tokenization must not be empty")
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    prefix = ([int(bos_token_id)] if bos_token_id is not None else []) + prompt_ids
    sequence = prefix + continuation_ids
    target_start = len(prefix)
    target_chunk_size = max(1, context_length // 2)
    windows = []
    while target_start < len(sequence):
        target_end = min(len(sequence), target_start + target_chunk_size)
        input_start = max(0, target_end - context_length)
        if input_start >= target_start:
            raise ValueError("judge scoring window has no causal prefix")
        token_ids = torch.tensor(
            [sequence[input_start:target_end]], dtype=torch.long
        )
        labels = token_ids.clone()
        label_start = target_start - input_start
        labels[:, :label_start] = -100
        target_count = int((labels != -100).sum().item())
        if target_count <= 0:
            raise ValueError("judge scoring window has no target tokens")
        windows.append((token_ids, labels, target_count))
        target_start = target_end
    return windows


def score_judge_cases(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    cases: Sequence[Mapping[str, Any]],
    device: torch.device | str,
    context_length: int = JUDGE_CONTEXT_LENGTH,
    progress: Callable[[int, int, float], None] | None = None,
) -> list[dict[str, Any]]:
    """Score cases with mean continuation NLL from a frozen causal model."""
    resolved_device = torch.device(device)
    model.eval()
    scored = []
    with torch.inference_mode():
        for index, case in enumerate(cases, start=1):
            total_nll = 0.0
            total_targets = 0
            windows = build_candidate_windows(
                tokenizer=tokenizer,
                text=str(case["text"]),
                context_length=context_length,
            )
            for input_ids, labels, target_count in windows:
                output = model(
                    input_ids=input_ids.to(resolved_device),
                    labels=labels.to(resolved_device),
                    use_cache=False,
                )
                loss = float(output.loss.detach().float().item())
                if not math.isfinite(loss):
                    raise ValueError(f"non-finite judge loss: {case['case_id']}")
                total_nll += loss * target_count
                total_targets += target_count
            mean_nll = total_nll / total_targets
            row = {key: value for key, value in case.items() if key != "text"}
            row.update({
                "mean_continuation_nll": mean_nll,
                "judge_score": -mean_nll,
                "target_token_count": total_targets,
                "window_count": len(windows),
                "text_sha256": hashlib.sha256(
                    str(case["text"]).encode("utf-8")
                ).hexdigest(),
            })
            scored.append(row)
            if progress is not None:
                progress(index, len(cases), mean_nll)
    return scored


def binary_auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Compute AUROC as positive-negative pairwise ranking accuracy."""
    if len(scores) != len(labels) or not scores:
        raise ValueError("AUROC scores and labels must have equal nonzero length")
    positives = [score for score, label in zip(scores, labels, strict=True) if label]
    negatives = [score for score, label in zip(scores, labels, strict=True) if not label]
    if not positives or not negatives:
        raise ValueError("AUROC requires positive and negative examples")
    credit = 0.0
    for positive in positives:
        for negative in negatives:
            credit += _ranking_credit(positive, negative)
    return credit / (len(positives) * len(negatives))


def ordinal_pairwise_concordance(
    scores: Sequence[float], qualities: Sequence[int]
) -> float:
    """Measure score agreement across every unequal human-quality pair."""
    if len(scores) != len(qualities) or not scores:
        raise ValueError("concordance inputs must have equal nonzero length")
    credit = 0.0
    pair_count = 0
    for left, right in itertools.combinations(range(len(scores)), 2):
        if qualities[left] == qualities[right]:
            continue
        if qualities[left] > qualities[right]:
            credit += _ranking_credit(scores[left], scores[right])
        else:
            credit += _ranking_credit(scores[right], scores[left])
        pair_count += 1
    if pair_count == 0:
        raise ValueError("concordance requires at least two quality levels")
    return credit / pair_count


def evaluate_judge_gate(
    *, scored_cases: Sequence[Mapping[str, Any]], thresholds: Mapping[str, float]
) -> dict[str, Any]:
    """Apply every frozen ranking and human-agreement threshold."""
    triplets: dict[str, dict[str, float]] = {}
    human_rows = []
    for row in scored_cases:
        if row["family"] == "triplet":
            triplets.setdefault(str(row["prompt_id"]), {})[str(row["variant"])] = (
                float(row["judge_score"])
            )
        elif row["family"] == "human":
            human_rows.append(row)
        else:
            raise ValueError(f"unknown judge case family: {row['family']}")
    if len(triplets) != JUDGE_PROMPT_COUNT:
        raise ValueError("judge gate requires eight complete triplets")
    for variants in triplets.values():
        if set(variants) != {"genuine", "generated", "corrupted"}:
            raise ValueError("judge triplet variants are incomplete")
    if len(human_rows) != JUDGE_HUMAN_CASE_COUNT:
        raise ValueError("judge gate requires 56 human-labelled controls")

    pair_specs = (
        ("genuine_over_corrupted_accuracy", "genuine", "corrupted"),
        ("genuine_over_generated_accuracy", "genuine", "generated"),
        ("generated_over_corrupted_accuracy", "generated", "corrupted"),
    )
    metrics: dict[str, float] = {}
    for metric_name, preferred, rejected in pair_specs:
        metrics[metric_name] = sum(
            _ranking_credit(variants[preferred], variants[rejected])
            for variants in triplets.values()
        ) / len(triplets)

    human_scores = [float(row["judge_score"]) for row in human_rows]
    grammar = [bool(row["grammar"]) for row in human_rows]
    noncollapse = [not bool(row["collapse"]) for row in human_rows]
    qualities = [
        2 * int(bool(row["grammar"]))
        + int(bool(row["topic"]))
        + 2 * int(not bool(row["collapse"]))
        for row in human_rows
    ]
    metrics["grammar_auroc"] = binary_auroc(human_scores, grammar)
    metrics["noncollapse_auroc"] = binary_auroc(human_scores, noncollapse)
    metrics["human_ordinal_pairwise_concordance"] = ordinal_pairwise_concordance(
        human_scores, qualities
    )
    checks = {
        name: {
            "value": value,
            "threshold": float(thresholds[name]),
            "passed": value >= float(thresholds[name]),
        }
        for name, value in metrics.items()
    }
    return {
        "metrics": metrics,
        "checks": checks,
        "gate_passed": all(check["passed"] for check in checks.values()),
        "triplet_count": len(triplets),
        "human_case_count": len(human_rows),
    }


def build_judge_gate_report(result: Mapping[str, Any]) -> str:
    """Render the fixed gate outcome as a concise public Markdown report."""
    metric_labels = {
        "genuine_over_corrupted_accuracy": "Genuine above corrupted",
        "genuine_over_generated_accuracy": "Genuine above generated",
        "generated_over_corrupted_accuracy": "Generated above corrupted",
        "grammar_auroc": "Grammar AUROC",
        "noncollapse_auroc": "Non-collapse AUROC",
        "human_ordinal_pairwise_concordance": "Human ordinal concordance",
    }
    lines = [
        "# Minerva 3B Judge Gate Result",
        "",
        f"- Model: `{result['model_id']}`",
        f"- Revision: `{result['revision']}`",
        "- Weights: untouched FP16; no adapter and no parameter updates.",
        "- Score: negative mean continuation NLL conditioned on the opening line.",
        f"- Validation triplets: {result['gate']['triplet_count']}.",
        f"- Blinded human-labelled controls: {result['gate']['human_case_count']}.",
        "- Final-test material used: no.",
        f"- GPU: {result['gpu_name']} with FP16 CPU layer offload.",
        f"- Peak CUDA reservation: {result['peak_cuda_reserved_mib']:.1f} MiB.",
        "",
        "## Gate Checks",
        "",
        "| Check | Value | Required | Status |",
        "| --- | ---: | ---: | --- |",
    ]
    for name, check in result["gate"]["checks"].items():
        lines.append(
            f"| {metric_labels[name]} | {check['value']:.4f} | "
            f">= {check['threshold']:.4f} | "
            f"{'pass' if check['passed'] else 'fail'} |"
        )
    scored_cases = result["scored_cases"]
    triplet_means = {
        variant: _mean([
            float(row["mean_continuation_nll"])
            for row in scored_cases
            if row["family"] == "triplet" and row["variant"] == variant
        ])
        for variant in ("genuine", "generated", "corrupted")
    }
    grammar_yes = _mean([
        float(row["mean_continuation_nll"])
        for row in scored_cases
        if row["family"] == "human" and row["grammar"]
    ])
    grammar_no = _mean([
        float(row["mean_continuation_nll"])
        for row in scored_cases
        if row["family"] == "human" and not row["grammar"]
    ])
    collapse_yes = _mean([
        float(row["mean_continuation_nll"])
        for row in scored_cases
        if row["family"] == "human" and row["collapse"]
    ])
    collapse_no = _mean([
        float(row["mean_continuation_nll"])
        for row in scored_cases
        if row["family"] == "human" and not row["collapse"]
    ])
    lines.extend([
        "",
        "## Diagnostic Mean NLL",
        "",
        "Lower NLL means Minerva assigns higher likelihood.",
        "",
        "| Group | Mean NLL |",
        "| --- | ---: |",
        f"| Genuine validation sonnets | {triplet_means['genuine']:.4f} |",
        f"| From-scratch generated controls | {triplet_means['generated']:.4f} |",
        f"| Word-order corruptions | {triplet_means['corrupted']:.4f} |",
        f"| Human grammar: yes | {grammar_yes:.4f} |",
        f"| Human grammar: no | {grammar_no:.4f} |",
        f"| Human collapse: yes | {collapse_yes:.4f} |",
        f"| Human collapse: no | {collapse_no:.4f} |",
    ])
    passed = bool(result["gate"]["gate_passed"])
    lines.extend([
        "",
        "## Decision",
        "",
        f"**{'Pass' if passed else 'Fail'}.** "
        + (
            "All predeclared checks pass, so one fixed DPO branch and one "
            "independent fixed GRPO branch may proceed."
            if passed
            else "At least one predeclared check fails, so DPO and GRPO remain "
            "unauthorized under the recorded exit policy."
        ),
        "",
        "Authentic sonnets may overlap Minerva's external pretraining data. The "
        "human-labelled generated controls are therefore mandatory evidence, "
        "not an optional diagnostic.",
        "",
        "The judge score measures model likelihood, not metre, rhyme, historical "
        "authenticity, or complete human preference. Those controls remain "
        "separate in any authorized post-training recipe.",
        "",
    ])
    return "\n".join(lines)


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for a local artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ranking_credit(preferred: float, rejected: float) -> float:
    if preferred > rejected:
        return 1.0
    if preferred == rejected:
        return 0.5
    return 0.0


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    return sum(values) / len(values)


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise ValueError("judge tokenizer returned multiple examples")
        input_ids = input_ids[0]
    if not isinstance(input_ids, list) or any(not isinstance(value, int) for value in input_ids):
        raise ValueError("judge tokenizer must return a list of token IDs")
    return input_ids


def _nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _resolve_recorded_output_path(repo_root: Path, path: Path) -> Path:
    if not path.is_absolute():
        return repo_root / path
    parts = path.parts
    try:
        output_index = parts.index("outputs")
    except ValueError as error:
        raise ValueError(f"recorded human-control path is outside outputs: {path}") from error
    return repo_root.joinpath(*parts[output_index:])
