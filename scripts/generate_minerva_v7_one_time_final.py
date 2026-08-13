#!/usr/bin/env python3
"""Run the frozen one-time matched Stage-3-versus-DPO V7 final generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_analysis.minerva_v7_dpo_validation import generate_matched_validation
from sonnet_analysis.minerva_v7_final_evaluation import (
    EXPECTED_SEEDS, FINAL_GENERATION_VERSION, open_final_test_prompts,
    load_frozen_final_protocol,
)
from sonnet_analysis.minerva_v7_runtime import (
    gpu_preflight, load_bf16_model_and_tokenizer, load_verified_state,
)
from sonnet_training.minerva_v7_ai_dpo import TARGET_MODULES, load_dpo_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--state-audit", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hourly-rate", type=float, required=True)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/local/minerva_7b_v7_dpo/final_test/generation"),
    )
    args = parser.parse_args()
    protocol_path = ROOT / args.protocol
    protocol = load_frozen_final_protocol(protocol_path)
    selection_path = ROOT / str(protocol["selection_record_path"])
    if hashlib.sha256(selection_path.read_bytes()).hexdigest() != protocol["selection_record_sha256"]:
        raise ValueError("frozen final selection record hash mismatch")
    state = load_verified_state(ROOT / args.state_audit, "stage_3_selected")
    if state["state_identity_sha256"] != protocol["stage_3_state_identity_sha256"]:
        raise ValueError("frozen Stage-3 identity mismatch")
    adapter_path = ROOT / str(protocol["dpo_adapter_path"])
    if hashlib.sha256(adapter_path.read_bytes()).hexdigest() != protocol["dpo_adapter_sha256"]:
        raise ValueError("frozen DPO adapter hash mismatch")
    print(
        "minerva-v7-final | start job=one_time_final device=1xh100-80gb "
        "total_steps=4976 progress_interval=one_batch test_access=authorized_once",
        flush=True,
    )
    preflight = gpu_preflight(
        output_root=ROOT / args.output_dir,
        required_output_bytes=2 * 1024**3, hourly_rate=args.hourly_rate,
    )
    import torch
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

    config = load_dpo_config(ROOT / "configs/minerva_7b_v7_ai_judged_dpo.json")
    model, tokenizer = load_bf16_model_and_tokenizer(
        state=state, config=config, device=torch.device("cuda:0")
    )
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.05,
        bias="none", target_modules=list(TARGET_MODULES),
    ))
    checkpoint = torch.load(adapter_path, map_location="cpu", weights_only=True)
    if checkpoint.get("parent_state_identity_sha256") != state["state_identity_sha256"]:
        raise ValueError("DPO adapter parent mismatch")
    set_peft_model_state_dict(model, checkpoint["adapter_state_dict"])
    # This call is intentionally the first point at which V7 test text is decoded.
    prompts = open_final_test_prompts(
        protocol=protocol,
        encoded_report_path=ROOT / "reports/minerva_7b_v7_encoded_data_v1.json",
        tokenizer=tokenizer, repo_root=ROOT,
    )
    prompt_manifest = {
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "prompt_count": len(prompts), "prompts": prompts,
        "v7_test_accessed": True, "retuning_after_test_forbidden": True,
    }
    args.output_dir = ROOT / args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir.parent / "test_openings.private.json").write_text(
        json.dumps(prompt_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    started = time.monotonic()
    result = generate_matched_validation(
        model=model, tokenizer=tokenizer, prompts=prompts,
        seeds=EXPECTED_SEEDS, recipe=protocol["recipe"],
        output_dir=args.output_dir, state_identity=state["state_identity_sha256"],
        adapter_identity=protocol["dpo_adapter_sha256"], device="cuda:0",
        batch_size=args.batch_size,
        generation_version=FINAL_GENERATION_VERSION,
        analysis_role="one_time_final_test_no_retuning",
        v7_test_accessed=True,
        progress=lambda message: print(f"minerva-v7-final | {message}", flush=True),
    )
    elapsed = time.monotonic() - started
    completion_path = args.output_dir / "complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion.update({
        "final_protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "analysis_role": "one_time_final_test_no_retuning",
        "v7_test_accessed": True,
        "retuning_after_test_forbidden": True,
    })
    completion_path.write_text(
        json.dumps(completion, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"minerva-v7-final | complete outputs={result['completed_output_count']} "
        f"elapsed={elapsed:.1f}s cost_usd={elapsed / 3600 * args.hourly_rate:.3f} "
        f"gpu={preflight['gpu_name']} v7_test_accessed=True no_retuning=True",
        flush=True,
    )


if __name__ == "__main__":
    main()
