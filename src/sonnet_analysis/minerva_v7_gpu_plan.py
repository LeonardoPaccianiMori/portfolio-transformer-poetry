"""Bounded GPU extraction plans and a fail-closed causal-experiment gate."""

from __future__ import annotations

import hashlib
import json
import struct
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from sonnet_analysis.minerva_v7_registry import MODEL_STATES


GPU_PLAN_VERSION = "minerva_7b_v7_gpu_extraction_plan_v1"
CAUSAL_GATE_VERSION = "minerva_7b_v7_causal_gate_v1"
PROBE_VERSION = "minerva_7b_v7_activation_probes_v1"
EXPECTED_DOMAINS = {
    "modern_instruction",
    "historical_general",
    "historical_non_sonnet_poetry",
    "standard_sonnet",
}


def validate_probe_manifest(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    if expected_sha256 is not None and _sha256(path) != expected_sha256:
        raise ValueError("activation-probe manifest hash mismatch")
    probe = json.loads(path.read_text(encoding="utf-8"))
    if probe.get("probe_version") != PROBE_VERSION or probe.get("probe_count") != 48:
        raise ValueError("activation-probe manifest is not the frozen 48-probe contract")
    if probe.get("v7_test_accessed") is not False:
        raise ValueError("activation-probe manifest does not prove V7 test isolation")
    rows = probe.get("probes")
    if not isinstance(rows, list) or len(rows) != 48:
        raise ValueError("activation-probe rows are incomplete")
    identities = [
        (
            str(row.get("probe_id")), str(row.get("source_identity")),
            str(row.get("input_ids_sha256")),
        )
        for row in rows
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("activation-probe full identities are not unique")
    domains = Counter(str(row.get("domain")) for row in rows)
    if set(domains) != EXPECTED_DOMAINS or any(domains[domain] != 12 for domain in EXPECTED_DOMAINS):
        raise ValueError("activation-probe domain balance changed")
    for row in rows:
        source_split = str(row.get("source_split", ""))
        if "test" in source_split.lower():
            raise ValueError("activation-probe contract contains a test split")
        tokens = [int(value) for value in row.get("input_ids", [])]
        mask = row.get("attention_mask")
        positions = [int(value) for value in row.get("selected_positions", [])]
        if not tokens or not isinstance(mask, list) or len(mask) != len(tokens):
            raise ValueError(f"invalid tokens or mask for probe: {row.get('probe_id')}")
        if any(value != 1 for value in mask):
            raise ValueError("frozen probe attention mask changed")
        if positions != sorted(set(positions)) or any(value < 0 or value >= len(tokens) for value in positions):
            raise ValueError("frozen probe positions are invalid")
        digest = hashlib.sha256(b"".join(struct.pack("<I", value) for value in tokens)).hexdigest()
        if digest != row.get("input_ids_sha256"):
            raise ValueError("activation-probe token hash mismatch")
    return probe


def build_gpu_extraction_plan(
    *,
    probe_manifest_path: Path,
    state_audit: Mapping[str, Any],
    output_root: Path,
    model_config: Mapping[str, Any],
    expected_probe_sha256: str | None = None,
) -> dict[str, Any]:
    """Estimate extraction outputs and create one resumable job per complete state."""

    probes = validate_probe_manifest(probe_manifest_path, expected_sha256=expected_probe_sha256)
    states = {row["state_id"]: row for row in state_audit["states"]}
    hidden_size = int(model_config["hidden_size"])
    layers = int(model_config["num_hidden_layers"])
    heads = int(model_config["num_attention_heads"])
    total_tokens = sum(len(row["input_ids"]) for row in probes["probes"])
    total_selected = sum(len(row["selected_positions"]) for row in probes["probes"])
    # Local raw BF16 hidden states at every layer; pooled/selected FP32 is authoritative aggregate.
    raw_hidden_bytes = total_tokens * hidden_size * (layers + 1) * 2
    aggregate_hidden_bytes = total_selected * hidden_size * (layers + 1) * 4
    attention_summary_bytes = len(probes["probes"]) * layers * heads * 2 * 4
    bounded = probes["extraction"]["bounded_raw_attention"]
    raw_attention_bytes = (
        len(EXPECTED_DOMAINS)
        * len(bounded["layer_indices"])
        * heads
        * int(bounded["maximum_tokens"]) ** 2
        * 2
    )
    top_k = int(probes["extraction"]["fixed_logit_summary"]["top_k"])
    logit_summary_bytes = total_selected * (top_k * 8 + 16)
    per_state = raw_hidden_bytes + aggregate_hidden_bytes + attention_summary_bytes + raw_attention_bytes + logit_summary_bytes
    jobs = []
    for state in MODEL_STATES:
        audited = states.get(state.state_id)
        if audited is None or audited.get("status") != "complete":
            continue
        jobs.append(
            {
                "state_id": state.state_id,
                "state_path": audited.get("model_dir") or audited.get("path"),
                "output_dir": str(output_root / state.state_id),
                "completion_marker": str(output_root / state.state_id / "complete.json"),
                "estimated_output_bytes": per_state,
                "resumable_unit": "one_complete_model_state",
            }
        )
    return {
        "plan_version": GPU_PLAN_VERSION,
        "probe_manifest_path": str(probe_manifest_path),
        "probe_manifest_sha256": _sha256(probe_manifest_path),
        "probe_count": 48,
        "model_state_count": 7,
        "ready_state_count": len(jobs),
        "jobs": jobs,
        "estimates": {
            "total_probe_tokens": total_tokens,
            "total_selected_positions": total_selected,
            "raw_hidden_states_bytes_per_state": raw_hidden_bytes,
            "aggregate_hidden_states_bytes_per_state": aggregate_hidden_bytes,
            "attention_summaries_bytes_per_state": attention_summary_bytes,
            "bounded_raw_attention_bytes_per_state": raw_attention_bytes,
            "logit_summaries_bytes_per_state": logit_summary_bytes,
            "total_bytes_per_state": per_state,
            "all_seven_states_bytes": per_state * 7,
        },
        "execution": {
            "model_mode": "eval",
            "gradient_enabled": False,
            "model_dtype": "bfloat16",
            "aggregate_dtype": "float32",
            "progress_unit": "probe within model state",
            "authoritative_generation_dtype": "bfloat16",
            "v7_test_accessed": False,
        },
        "causal_experiments_authorized": False,
        "causal_gate_version": CAUSAL_GATE_VERSION,
    }


def validate_causal_experiment_proposal(proposal: Mapping[str, Any]) -> dict[str, Any]:
    """Validate scientific completeness; never authorize execution by itself."""

    required_text = (
        "descriptive_finding_id", "hypothesis", "intervention", "negative_control",
        "predicted_adaptation_effect", "predicted_preservation_effect", "stopping_rule",
    )
    missing = [key for key in required_text if not str(proposal.get(key, "")).strip()]
    comparisons = proposal.get("model_state_comparisons")
    domains = proposal.get("evaluation_domains")
    metrics = proposal.get("primary_metrics")
    if not isinstance(comparisons, list) or not comparisons:
        missing.append("model_state_comparisons")
    if not isinstance(domains, list) or not EXPECTED_DOMAINS.issubset(set(domains)):
        missing.append("evaluation_domains_including_all_adaptation_and_preservation_controls")
    if not isinstance(metrics, list) or not metrics:
        missing.append("primary_metrics")
    if proposal.get("v7_test_accessed") is not False:
        missing.append("v7_test_accessed=false")
    if missing:
        raise ValueError("causal proposal is incomplete: " + ", ".join(missing))
    return {
        "gate_version": CAUSAL_GATE_VERSION,
        "proposal_complete": True,
        "execution_authorized": False,
        "separate_user_approval_required": True,
        "proposal": dict(proposal),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
