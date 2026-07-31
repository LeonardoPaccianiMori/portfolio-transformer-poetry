"""Define the fixed PAISA-to-historical final rescue training curriculum."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from sonnet_training.pretraining_run import PretrainingRunConfig
from sonnet_training.transformer_run import write_json


RESCUE_DATASET_VERSION = "paisa_historical_rescue_v1"
RESCUE_ENCODED_REPORT_PATH = Path(
    "reports/paisa_historical_rescue_v1_encoded_report.json"
)
RESCUE_HARDWARE_REPORT_PATH = Path(
    "reports/paisa_historical_rescue_v1_hardware_benchmark.md"
)
RESCUE_TRAINING_PLAN_JSON_PATH = Path(
    "reports/paisa_historical_rescue_v1_training_plan.json"
)
RESCUE_TRAINING_PLAN_MARKDOWN_PATH = Path(
    "reports/paisa_historical_rescue_v1_training_plan.md"
)

TOKENS_PER_OPTIMIZER_UPDATE = 4_096
SELECTED_MICROBATCH_SIZE = 4
SELECTED_GRADIENT_ACCUMULATION_STEPS = 2
SELECTED_TOKENS_PER_SECOND = 7_924.406920354695


@dataclass(frozen=True)
class RescueStageSpecification:
    """Fixed schedule and validation policy for one sequential rescue stage."""

    stage_id: str
    train_split_id: str
    validation_split_id: str
    max_passes: int
    learning_rate: float
    warmup_steps: int
    stable_fraction: float
    min_learning_rate: float
    validation_mode: str
    eval_interval: int
    eval_batches: int
    checkpoint_interval: int


@dataclass(frozen=True)
class RescueStagePlan:
    """One resolved stage with its exact stream paths and bounded update budget."""

    stage_id: str
    train_split_id: str
    validation_split_id: str
    train_tokens_path: str
    validation_tokens_path: str
    train_tokens: int
    validation_tokens: int
    max_passes: int
    planned_target_tokens: int
    train_steps: int
    unused_target_token_budget: int
    learning_rate: float
    warmup_steps: int
    stable_steps: int
    min_learning_rate: float
    validation_mode: str
    eval_interval: int
    eval_batches: int
    checkpoint_interval: int


@dataclass(frozen=True)
class RescueTrainingPlan:
    """The immutable architecture, measured throughput, and two training stages."""

    dataset_version: str
    encoded_report_path: str
    tokenizer_path: str
    vocab_size: int
    tokens_per_optimizer_update: int
    measured_tokens_per_second: float
    estimated_raw_training_hours: float
    architecture: dict[str, int | str | bool]
    stages: tuple[RescueStagePlan, ...]


_STAGE_SPECS = (
    RescueStageSpecification(
        stage_id="modern_italian_pretraining",
        train_split_id="paisa_train",
        validation_split_id="paisa_validation",
        max_passes=3,
        learning_rate=3e-4,
        warmup_steps=1_000,
        stable_fraction=0.8,
        min_learning_rate=3e-5,
        validation_mode="random_batches",
        eval_interval=2_000,
        eval_batches=20,
        checkpoint_interval=2_000,
    ),
    RescueStageSpecification(
        stage_id="historical_italian_annealing",
        train_split_id="historical_train",
        validation_split_id="historical_validation",
        max_passes=12,
        learning_rate=1e-4,
        warmup_steps=500,
        stable_fraction=0.8,
        min_learning_rate=1e-5,
        validation_mode="sequential_windows",
        eval_interval=500,
        eval_batches=1,
        checkpoint_interval=1_000,
    ),
)


def build_rescue_training_plan(repo_root: Path) -> RescueTrainingPlan:
    """Resolve fixed capped stage budgets from the completed encoded-corpus report."""

    report_path = repo_root / RESCUE_ENCODED_REPORT_PATH
    report = _read_complete_encoded_report(report_path)
    tokenizer = _required_mapping(report, "tokenizer")
    tokenizer_path = _required_string(tokenizer, "path")
    vocab_size = _required_positive_int(tokenizer, "vocab_size")
    split_by_id = _splits_by_id(report)

    stages = tuple(
        _build_stage_plan(specification, split_by_id)
        for specification in _STAGE_SPECS
    )
    estimated_raw_training_hours = sum(
        stage.train_steps * TOKENS_PER_OPTIMIZER_UPDATE
        for stage in stages
    ) / SELECTED_TOKENS_PER_SECOND / 3_600
    return RescueTrainingPlan(
        dataset_version=RESCUE_DATASET_VERSION,
        encoded_report_path=str(RESCUE_ENCODED_REPORT_PATH),
        tokenizer_path=tokenizer_path,
        vocab_size=vocab_size,
        tokens_per_optimizer_update=TOKENS_PER_OPTIMIZER_UPDATE,
        measured_tokens_per_second=SELECTED_TOKENS_PER_SECOND,
        estimated_raw_training_hours=estimated_raw_training_hours,
        architecture={
            "embedding_dim": 640,
            "num_layers": 10,
            "num_heads": 10,
            "head_dim": 64,
            "feed_forward_dim": 1_707,
            "feed_forward_type": "swiglu",
            "normalization_type": "layer_norm",
            "position_encoding_type": "learned_absolute",
            "tie_token_embeddings": False,
            "context_length": 512,
            "max_context_length": 512,
            "microbatch_size": SELECTED_MICROBATCH_SIZE,
            "gradient_accumulation_steps": SELECTED_GRADIENT_ACCUMULATION_STEPS,
        },
        stages=stages,
    )


def build_rescue_stage_config(
    *,
    plan: RescueTrainingPlan,
    stage_id: str,
    device: str,
    resume_from_checkpoint: str = "",
    historical_parent_checkpoint_path: str = "",
) -> PretrainingRunConfig:
    """Create the fixed generic-pretraining configuration for one rescue stage."""

    stage = _stage_by_id(plan, stage_id)
    architecture = plan.architecture
    initialization_checkpoint_path = ""
    if stage.stage_id == "historical_italian_annealing":
        initialization_checkpoint_path = historical_parent_checkpoint_path
        if not initialization_checkpoint_path:
            raise ValueError(
                "historical_italian_annealing requires a PAISA best-validation "
                "checkpoint path"
            )

    return PretrainingRunConfig(
        dataset_version=plan.dataset_version,
        train_tokens_path=stage.train_tokens_path,
        validation_tokens_path=stage.validation_tokens_path,
        tokenizer_path=plan.tokenizer_path,
        dataset_report_path=str(RESCUE_ENCODED_REPORT_PATH),
        train_split_id=stage.train_split_id,
        validation_split_id=stage.validation_split_id,
        batch_size=int(architecture["microbatch_size"]),
        gradient_accumulation_steps=int(
            architecture["gradient_accumulation_steps"]
        ),
        context_length=int(architecture["context_length"]),
        train_steps=stage.train_steps,
        eval_interval=stage.eval_interval,
        eval_batches=stage.eval_batches,
        validation_mode=stage.validation_mode,  # type: ignore[arg-type]
        learning_rate=stage.learning_rate,
        learning_rate_schedule="warmup_stable_cosine",
        warmup_steps=stage.warmup_steps,
        stable_steps=stage.stable_steps,
        min_learning_rate=stage.min_learning_rate,
        seed=1337,
        prompt="Nel ",
        max_new_tokens=300,
        device=device,
        embedding_dim=int(architecture["embedding_dim"]),
        num_layers=int(architecture["num_layers"]),
        num_heads=int(architecture["num_heads"]),
        head_dim=int(architecture["head_dim"]),
        feed_forward_dim=int(architecture["feed_forward_dim"]),
        max_context_length=int(architecture["max_context_length"]),
        normalization_type="layer_norm",
        position_encoding_type="learned_absolute",
        feed_forward_type="swiglu",
        tie_token_embeddings=False,
        checkpoint_interval=stage.checkpoint_interval,
        checkpoint_retention="latest_only",
        progress_interval=100,
        resume_from_checkpoint=resume_from_checkpoint,
        initialization_checkpoint_path=initialization_checkpoint_path,
    )


def write_rescue_training_plan(repo_root: Path) -> RescueTrainingPlan:
    """Write the public machine-readable and Markdown records of the locked plan."""

    plan = build_rescue_training_plan(repo_root)
    payload = {
        **asdict(plan),
        "hardware_selection": {
            "benchmark_report_path": str(RESCUE_HARDWARE_REPORT_PATH),
            "candidate_name": "rescue_upper_micro4",
            "microbatch_size": SELECTED_MICROBATCH_SIZE,
            "gradient_accumulation_steps": SELECTED_GRADIENT_ACCUMULATION_STEPS,
            "peak_cuda_memory_mib": 3_092.326171875,
        },
    }
    write_json(repo_root / RESCUE_TRAINING_PLAN_JSON_PATH, payload)
    markdown_path = repo_root / RESCUE_TRAINING_PLAN_MARKDOWN_PATH
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        build_rescue_training_plan_markdown(plan),
        encoding="utf-8",
    )
    return plan


def build_rescue_training_plan_markdown(plan: RescueTrainingPlan) -> str:
    """Build the human-readable public rescue training-plan report."""

    lines = [
        "# PAISA-To-Historical Rescue Training Plan",
        "",
        "This is the single final from-scratch rescue experiment. It uses the",
        "GPU-selected fixed architecture and sequential PAISA then historical",
        "Italian training stages; it is not an architecture or hyperparameter sweep.",
        "",
        "## Hardware Selection",
        "",
        "- Candidate: `rescue_upper_micro4`",
        "- Microbatch size: `4`",
        "- Gradient accumulation: `2`",
        f"- Targets per optimizer update: `{plan.tokens_per_optimizer_update:,}`",
        "- Measured throughput: "
        f"`{plan.measured_tokens_per_second:,.1f} tokens/s`",
        "- Peak CUDA memory: `3,092.3 MiB`",
        "",
        "## Architecture",
        "",
    ]
    for key, value in plan.architecture.items():
        lines.append(f"- {key.replace('_', ' ').capitalize()}: `{value}`")

    lines.extend([
        "",
        "## Stages",
        "",
        "| Stage | Stream | Pass cap | Updates | Target tokens | Validation |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ])
    for stage in plan.stages:
        lines.append(
            "| {stage} | `{stream}` | {passes} | {updates:,} | {tokens:,} | "
            "{validation} every {interval:,} updates |".format(
                stage=stage.stage_id,
                stream=stage.train_split_id,
                passes=stage.max_passes,
                updates=stage.train_steps,
                tokens=stage.planned_target_tokens,
                validation=stage.validation_mode,
                interval=stage.eval_interval,
            )
        )
    lines.extend([
        "",
        "The update budgets use floor division so neither stage exceeds its fixed",
        "pass cap. The small unused remainder is below one optimizer update per",
        "stage. PAISA selection uses 20 sampled validation batches; historical",
        "selection uses all non-overlapping validation windows. The historical",
        "stage loads the selected PAISA weights but starts a fresh AdamW optimizer,",
        "so PAISA optimizer moments are not transferred across domains.",
        "",
        "## Runtime",
        "",
        "The measured-throughput estimate for forward/backward updates only is "
        f"`{plan.estimated_raw_training_hours:.1f} hours`. The operational "
        "estimate is 75-90 hours after validation, atomic checkpoints, and normal "
        "runtime variation.",
        "",
    ])
    return "\n".join(lines)


def _build_stage_plan(
    specification: RescueStageSpecification,
    split_by_id: dict[str, dict[str, object]],
) -> RescueStagePlan:
    train_split = _required_split(split_by_id, specification.train_split_id)
    validation_split = _required_split(
        split_by_id,
        specification.validation_split_id,
    )
    train_tokens = _required_positive_int(train_split, "tokens")
    planned_target_tokens = train_tokens * specification.max_passes
    train_steps, unused_target_token_budget = divmod(
        planned_target_tokens,
        TOKENS_PER_OPTIMIZER_UPDATE,
    )
    stable_steps = int(train_steps * specification.stable_fraction)
    if stable_steps < specification.warmup_steps or stable_steps >= train_steps:
        raise ValueError(f"invalid stable schedule boundary for {specification.stage_id}")
    return RescueStagePlan(
        stage_id=specification.stage_id,
        train_split_id=specification.train_split_id,
        validation_split_id=specification.validation_split_id,
        train_tokens_path=_required_string(train_split, "output_path"),
        validation_tokens_path=_required_string(validation_split, "output_path"),
        train_tokens=train_tokens,
        validation_tokens=_required_positive_int(validation_split, "tokens"),
        max_passes=specification.max_passes,
        planned_target_tokens=planned_target_tokens,
        train_steps=train_steps,
        unused_target_token_budget=unused_target_token_budget,
        learning_rate=specification.learning_rate,
        warmup_steps=specification.warmup_steps,
        stable_steps=stable_steps,
        min_learning_rate=specification.min_learning_rate,
        validation_mode=specification.validation_mode,
        eval_interval=specification.eval_interval,
        eval_batches=specification.eval_batches,
        checkpoint_interval=specification.checkpoint_interval,
    )


def _read_complete_encoded_report(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"encoded rescue report does not exist: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("encoded rescue report must be a JSON object")
    if report.get("status") != "complete":
        raise ValueError("encoded rescue report is not complete")
    return report


def _splits_by_id(report: dict[str, object]) -> dict[str, dict[str, object]]:
    splits = report.get("splits")
    if not isinstance(splits, list):
        raise ValueError("encoded rescue report is missing splits")
    mapped_splits = {
        str(split.get("split_id")): split
        for split in splits
        if isinstance(split, dict)
    }
    if len(mapped_splits) != len(splits):
        raise ValueError("encoded rescue report has an invalid split")
    return mapped_splits


def _stage_by_id(plan: RescueTrainingPlan, stage_id: str) -> RescueStagePlan:
    for stage in plan.stages:
        if stage.stage_id == stage_id:
            return stage
    raise ValueError(f"unsupported rescue stage: {stage_id}")


def _required_split(
    split_by_id: dict[str, dict[str, object]],
    split_id: str,
) -> dict[str, object]:
    split = split_by_id.get(split_id)
    if split is None:
        raise ValueError(f"encoded rescue report is missing split: {split_id}")
    if split.get("status") != "complete":
        raise ValueError(f"encoded rescue split is not complete: {split_id}")
    if split.get("dtype") != "torch.uint16":
        raise ValueError(f"encoded rescue split must use torch.uint16: {split_id}")
    return split


def _required_mapping(
    mapping: dict[str, object],
    field: str,
) -> dict[str, object]:
    value = mapping.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"encoded rescue report is missing mapping: {field}")
    return value


def _required_positive_int(mapping: dict[str, object], field: str) -> int:
    value = mapping.get(field)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"encoded rescue report is missing positive integer: {field}")
    return value


def _required_string(mapping: dict[str, object], field: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"encoded rescue report is missing string: {field}")
    return value
