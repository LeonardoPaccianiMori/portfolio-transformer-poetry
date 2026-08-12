"""Five-update BF16 full-weight calibration for Minerva 7B on an H100 80 GB."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from sonnet_training.cuda_compat import (
    cuda_device_name,
    cuda_device_properties,
    cuda_memory_info,
    max_cuda_memory_allocated,
    max_cuda_memory_reserved,
    prepare_cuda_memory_measurement,
    synchronize_cuda,
)
from sonnet_training.minerva_7b_full_weight_data import (
    FULL_WEIGHT_DATA_VERSION,
    load_full_weight_calibration_windows,
)
from sonnet_training.minerva_7b_model_audit import audit_full_weight_model
from sonnet_training.minerva_7b_qlora import (
    MINERVA_7B_INSTRUCT_MODEL_ID,
    MINERVA_7B_INSTRUCT_REVISION,
    is_cuda_out_of_memory,
)


FULL_WEIGHT_CALIBRATION_VERSION = "minerva_7b_bf16_full_weight_five_update_v1"
MINIMUM_H100_MEMORY_MIB = 75 * 1024
MINIMUM_POST_OPTIMIZER_HEADROOM_MIB = 8 * 1024


@dataclass(frozen=True)
class Minerva7BFullWeightCalibrationConfig:
    """Freeze the approved H100 memory and numerical-stability probe."""

    model_id: str = MINERVA_7B_INSTRUCT_MODEL_ID
    revision: str = MINERVA_7B_INSTRUCT_REVISION
    cache_dir: str = "data/local/minerva_qlora/huggingface"
    data_report_path: str = "data/local/minerva_7b_full_weight/encoded/report.json"
    calibration_windows_path: str = (
        "data/local/minerva_7b_full_weight/encoded/calibration_windows.pt"
    )
    output_path: str = (
        "data/local/minerva_7b_full_weight/full_weight_calibration.json"
    )
    device: str = "cuda:0"
    context_length: int = 512
    microbatch_size: int = 1
    gradient_accumulation_steps: int = 1
    optimizer_updates: int = 5
    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    max_gradient_norm: float = 1.0
    optimizer: str = "PagedAdamW8bit"
    parameter_dtype: str = "bfloat16"
    gradient_checkpointing_use_reentrant: bool = False
    required_gpu_name_fragment: str = "H100"
    minimum_total_memory_mib: int = MINIMUM_H100_MEMORY_MIB
    minimum_post_optimizer_headroom_mib: int = (
        MINIMUM_POST_OPTIMIZER_HEADROOM_MIB
    )
    seed: int = 1337


def validate_full_weight_calibration_config(
    config: Minerva7BFullWeightCalibrationConfig,
) -> None:
    """Reject any mutation that would turn the approved probe into a sweep."""
    if config != Minerva7BFullWeightCalibrationConfig():
        raise ValueError("Minerva 7B full-weight calibration configuration is locked")


def build_full_weight_calibration_report(
    *,
    config: Minerva7BFullWeightCalibrationConfig,
    status: str,
    device: torch.device,
    gpu_name: str,
    total_gpu_memory_mib: float,
    native_bf16_supported: bool,
    model_audit: Mapping[str, Any] | None,
    update_rows: list[dict[str, Any]],
    validation: Mapping[str, Any] | None,
    peak_allocated_mib: float,
    peak_reserved_mib: float,
    minimum_free_after_optimizer_mib: float,
    elapsed_seconds: float | None,
    package_versions: Mapping[str, str],
    error: str | None = None,
) -> dict[str, Any]:
    """Build the machine-readable fit decision from measured evidence."""
    if status not in {"ok", "out_of_memory", "numerical_failure", "hardware_reject"}:
        raise ValueError("unsupported full-weight calibration status")
    reserved_headroom_mib = total_gpu_memory_mib - peak_reserved_mib
    numerical_pass = (
        len(update_rows) == config.optimizer_updates
        and all(
            math.isfinite(float(row["loss"]))
            and math.isfinite(float(row["gradient_norm"]))
            for row in update_rows
        )
    )
    model_pass = bool(
        model_audit is not None
        and model_audit.get("all_weights_trainable") is True
        and model_audit.get("adapter_free") is True
        and model_audit.get("quantization_free") is True
        and model_audit.get("parameter_dtype_counts")
        == {"bfloat16": model_audit.get("total_parameter_count")}
    )
    hardware_pass = (
        config.required_gpu_name_fragment.lower() in gpu_name.lower()
        and total_gpu_memory_mib >= config.minimum_total_memory_mib
        and native_bf16_supported
    )
    memory_pass = (
        minimum_free_after_optimizer_mib
        >= config.minimum_post_optimizer_headroom_mib
        and reserved_headroom_mib >= config.minimum_post_optimizer_headroom_mib
    )
    fit = status == "ok" and numerical_pass and model_pass and hardware_pass and memory_pass
    processed_tokens = len(update_rows) * config.context_length
    return {
        "calibration_version": FULL_WEIGHT_CALIBRATION_VERSION,
        "status": status,
        "full_weight_training_fit_decision": "pass" if fit else "reject",
        "config": asdict(config),
        "device": str(device),
        "gpu_name": gpu_name,
        "total_gpu_memory_mib": total_gpu_memory_mib,
        "native_bf16_supported": native_bf16_supported,
        "hardware_gate_passed": hardware_pass,
        "model_audit": dict(model_audit) if model_audit is not None else None,
        "model_gate_passed": model_pass,
        "update_rows": update_rows,
        "numerical_gate_passed": numerical_pass,
        "validation": dict(validation) if validation is not None else None,
        "peak_allocated_mib": peak_allocated_mib,
        "peak_reserved_mib": peak_reserved_mib,
        "reserved_headroom_mib": reserved_headroom_mib,
        "minimum_free_after_optimizer_mib": minimum_free_after_optimizer_mib,
        "minimum_required_headroom_mib": (
            config.minimum_post_optimizer_headroom_mib
        ),
        "memory_gate_passed": memory_pass,
        "elapsed_seconds": elapsed_seconds,
        "processed_tokens": processed_tokens,
        "tokens_per_second": (
            processed_tokens / elapsed_seconds
            if elapsed_seconds is not None and elapsed_seconds > 0
            else None
        ),
        "optimizer_state_measured_after_first_update": bool(update_rows),
        "retained_model_checkpoint": False,
        "long_training_authorized": False,
        "package_versions": dict(package_versions),
        "error": error,
    }


def calibrate_minerva_7b_full_weight(
    *,
    repo_root: Path,
    config: Minerva7BFullWeightCalibrationConfig = (
        Minerva7BFullWeightCalibrationConfig()
    ),
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run exactly five full-weight updates and retain no trained checkpoint."""
    validate_full_weight_calibration_config(config)
    dependencies = _load_dependencies()
    device = torch.device(config.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Minerva 7B full-weight calibration requires CUDA")
    device_index = device.index if device.index is not None else 0
    properties = cuda_device_properties(device)
    total_gpu_memory_mib = properties.total_memory / (1024**2)
    gpu_name = cuda_device_name(device)
    native_bf16_supported = torch.cuda.is_bf16_supported()
    package_versions = _package_versions(dependencies)
    output_path = _resolve(repo_root, config.output_path)

    if (
        config.required_gpu_name_fragment.lower() not in gpu_name.lower()
        or total_gpu_memory_mib < config.minimum_total_memory_mib
        or not native_bf16_supported
    ):
        report = build_full_weight_calibration_report(
            config=config,
            status="hardware_reject",
            device=device,
            gpu_name=gpu_name,
            total_gpu_memory_mib=total_gpu_memory_mib,
            native_bf16_supported=native_bf16_supported,
            model_audit=None,
            update_rows=[],
            validation=None,
            peak_allocated_mib=0.0,
            peak_reserved_mib=0.0,
            minimum_free_after_optimizer_mib=0.0,
            elapsed_seconds=None,
            package_versions=package_versions,
            error="GPU does not satisfy the locked H100 80 GB BF16 requirement",
        )
        _write_json(output_path, report)
        return report

    data_report_path = _resolve(repo_root, config.data_report_path)
    data_report = _read_json(data_report_path)
    windows_path = _resolve(repo_root, config.calibration_windows_path)
    _validate_data_artifacts(
        data_report=data_report,
        windows_path=windows_path,
        config=config,
    )
    windows = load_full_weight_calibration_windows(windows_path)
    torch.manual_seed(config.seed)
    prepare_cuda_memory_measurement(device)
    model_audit: dict[str, Any] | None = None
    update_rows: list[dict[str, Any]] = []
    validation: dict[str, Any] | None = None
    minimum_free_after_optimizer_mib = 0.0
    started_at = time.monotonic()
    try:
        _report(progress, "stage 1/6: loading unquantized Minerva 7B in BF16")
        model = dependencies["AutoModelForCausalLM"].from_pretrained(
            config.model_id,
            revision=config.revision,
            cache_dir=_resolve(repo_root, config.cache_dir),
            dtype=torch.bfloat16,
            device_map={"": device_index},
            low_cpu_mem_usage=True,
        )
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": config.gradient_checkpointing_use_reentrant
            }
        )
        _report(progress, "stage 2/6: auditing full-weight trainability")
        model_audit = audit_full_weight_model(model)
        if not (
            model_audit["all_weights_trainable"]
            and model_audit["adapter_free"]
            and model_audit["quantization_free"]
            and model_audit["parameter_dtype_counts"]
            == {"bfloat16": model_audit["total_parameter_count"]}
        ):
            raise ValueError("loaded model failed the full-weight BF16 audit")

        parameters = list(model.parameters())
        _report(progress, "stage 3/6: measuring stage-zero validation losses")
        initial_validation = _evaluate_windows(
            model=model,
            windows=windows["validation_windows"],
            sources=windows["validation_sources"],
            device=device,
        )
        _report(progress, "stage 4/6: constructing PagedAdamW8bit")
        optimizer = dependencies["bitsandbytes"].optim.PagedAdamW8bit(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        _report(progress, "stage 5/6: running five full-weight optimizer updates")
        free_after_updates: list[float] = []
        for update_index, source_id in enumerate(windows["training_sources"]):
            update_started_at = time.monotonic()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            input_ids = windows["training_windows"][update_index].to(
                device=device,
                dtype=torch.long,
            ).unsqueeze(0)
            loss = model(input_ids=input_ids, labels=input_ids).loss
            loss_value = float(loss.detach().item())
            if not math.isfinite(loss_value):
                raise FloatingPointError(f"non-finite loss at update {update_index + 1}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters,
                config.max_gradient_norm,
            )
            gradient_norm_value = float(gradient_norm.detach().item())
            if not math.isfinite(gradient_norm_value):
                raise FloatingPointError(
                    f"non-finite gradient norm at update {update_index + 1}"
                )
            optimizer.step()
            synchronize_cuda(device)
            update_seconds = time.monotonic() - update_started_at
            optimizer.zero_grad(set_to_none=True)
            del input_ids, loss
            synchronize_cuda(device)
            free_bytes, _ = cuda_memory_info(device)
            free_memory_mib = free_bytes / (1024**2)
            free_after_updates.append(free_memory_mib)
            row = {
                "update": update_index + 1,
                "source_split": source_id,
                "loss": loss_value,
                "gradient_norm": gradient_norm_value,
                "learning_rate": config.learning_rate,
                "update_seconds": update_seconds,
                "free_memory_after_optimizer_mib": free_memory_mib,
            }
            update_rows.append(row)
            _report(
                progress,
                "update {update}/{total} source={source} loss={loss:.4f} "
                "gradient_norm={gradient:.4f} free={free:.1f}MiB".format(
                    update=update_index + 1,
                    total=config.optimizer_updates,
                    source=source_id,
                    loss=loss_value,
                    gradient=gradient_norm_value,
                    free=free_memory_mib,
                ),
            )
        minimum_free_after_optimizer_mib = min(free_after_updates)

        _report(progress, "stage 6/6: measuring post-update validation losses")
        final_validation = _evaluate_windows(
            model=model,
            windows=windows["validation_windows"],
            sources=windows["validation_sources"],
            device=device,
        )
        validation = {
            "initial": initial_validation,
            "final": final_validation,
        }
        status = "ok"
        error_message = None
    except (torch.OutOfMemoryError, RuntimeError) as error:
        if not is_cuda_out_of_memory(error):
            raise
        status = "out_of_memory"
        error_message = str(error)
        _report(progress, "calibration reached the GPU memory limit")
    except FloatingPointError as error:
        status = "numerical_failure"
        error_message = str(error)
        _report(progress, f"calibration failed numerical checks: {error}")

    elapsed_seconds = time.monotonic() - started_at
    if status != "ok" and not update_rows:
        free_bytes, _ = cuda_memory_info(device)
        minimum_free_after_optimizer_mib = free_bytes / (1024**2)
    report = build_full_weight_calibration_report(
        config=config,
        status=status,
        device=device,
        gpu_name=gpu_name,
        total_gpu_memory_mib=total_gpu_memory_mib,
        native_bf16_supported=native_bf16_supported,
        model_audit=model_audit,
        update_rows=update_rows,
        validation=validation,
        peak_allocated_mib=max_cuda_memory_allocated(device) / (1024**2),
        peak_reserved_mib=max_cuda_memory_reserved(device) / (1024**2),
        minimum_free_after_optimizer_mib=minimum_free_after_optimizer_mib,
        elapsed_seconds=elapsed_seconds,
        package_versions=package_versions,
        error=error_message,
    )
    _write_json(output_path, report)
    return report


def _evaluate_windows(
    *,
    model: torch.nn.Module,
    windows: torch.Tensor,
    sources: list[str],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    rows = []
    with torch.no_grad():
        for index, source in enumerate(sources):
            input_ids = windows[index].to(device=device, dtype=torch.long).unsqueeze(0)
            loss = float(model(input_ids=input_ids, labels=input_ids).loss.item())
            if not math.isfinite(loss):
                raise FloatingPointError(f"non-finite validation loss for {source}")
            rows.append({"source_split": source, "loss": loss})
    return {
        "rows": rows,
        "mean_loss": sum(float(row["loss"]) for row in rows) / len(rows),
    }


def _validate_data_artifacts(
    *,
    data_report: Mapping[str, Any],
    windows_path: Path,
    config: Minerva7BFullWeightCalibrationConfig,
) -> None:
    if data_report.get("status") != "complete":
        raise ValueError("full-weight data report is not complete")
    if data_report.get("data_version") != FULL_WEIGHT_DATA_VERSION:
        raise ValueError("full-weight data version is not the approved version")
    if data_report.get("model_id") != config.model_id:
        raise ValueError("full-weight data model ID does not match calibration")
    if data_report.get("revision") != config.revision:
        raise ValueError("full-weight data revision does not match calibration")
    artifact = data_report.get("calibration_windows")
    if not isinstance(artifact, Mapping):
        raise ValueError("full-weight data report is missing calibration windows")
    if not windows_path.is_file():
        raise FileNotFoundError(f"calibration windows do not exist: {windows_path}")
    if _sha256(windows_path) != artifact.get("sha256"):
        raise ValueError("calibration-window SHA-256 does not match data report")


def _load_dependencies() -> dict[str, Any]:
    try:
        import accelerate
        import bitsandbytes
        import transformers
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise RuntimeError(
            "Minerva full-weight dependencies are missing; install "
            "requirements/minerva_qlora.txt in the project virtual environment"
        ) from error
    return {
        "accelerate": accelerate,
        "bitsandbytes": bitsandbytes,
        "transformers": transformers,
        "AutoModelForCausalLM": AutoModelForCausalLM,
    }


def _package_versions(dependencies: Mapping[str, Any]) -> dict[str, str]:
    return {
        "accelerate": dependencies["accelerate"].__version__,
        "bitsandbytes": dependencies["bitsandbytes"].__version__,
        "torch": torch.__version__,
        "transformers": dependencies["transformers"].__version__,
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _report(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
