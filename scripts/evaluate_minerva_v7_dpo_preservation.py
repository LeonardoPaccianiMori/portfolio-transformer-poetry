#!/usr/bin/env python3
"""Compare Stage-3 and DPO losses on frozen validation/preservation gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_analysis.minerva_v7_runtime import (
    load_bf16_model_and_tokenizer, load_verified_state,
)
from sonnet_training.minerva_7b_v7_execution import (
    FrozenWindowReader, Int32ShardStore, V7ExecutionConfig, build_execution_context,
)
from sonnet_training.minerva_7b_v7_trainer import (
    build_modern_preservation_reader, evaluate_all_gates,
    training_only_encoded_report, training_only_window_manifest,
)
from sonnet_training.minerva_v7_ai_dpo import TARGET_MODULES, load_dpo_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-audit", type=Path, required=True)
    parser.add_argument(
        "--adapter", type=Path,
        default=Path("artifacts/local/minerva_7b_v7_dpo/training/best_adapter.pt"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/local/minerva_7b_v7_dpo/validation/preservation.json"),
    )
    args = parser.parse_args()
    print(
        "minerva-v7-dpo | start job=preservation_validation device=1xh100-80gb "
        "total_steps=2 progress_interval=one_system",
        flush=True,
    )
    import torch
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

    config = load_dpo_config(ROOT / "configs/minerva_7b_v7_ai_judged_dpo.json")
    state = load_verified_state(ROOT / args.state_audit, "stage_3_selected")
    model, tokenizer = load_bf16_model_and_tokenizer(
        state=state, config=config, device=torch.device("cuda:0")
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.05,
            bias="none", target_modules=list(TARGET_MODULES),
        ),
    )
    adapter_path = ROOT / args.adapter
    checkpoint = torch.load(adapter_path, map_location="cpu", weights_only=True)
    set_peft_model_state_dict(model, checkpoint["adapter_state_dict"])
    execution_config = V7ExecutionConfig(
        repo_root=ROOT,
        execution_path=ROOT / "configs/minerva_7b_v7_execution.json",
        encoded_dir=ROOT / "data/local/minerva_7b_v7/encoded",
        window_index_dir=ROOT / "data/local/minerva_7b_v7/window_indexes",
        modern_encoded_dir=ROOT / "data/local/minerva_7b_full_weight/encoded",
        modern_index_path=ROOT / "data/local/minerva_7b_v7/modern_preservation_validation_v1.jsonl",
    )
    context = build_execution_context(execution_config)
    protocol = context["protocol"]
    prompts = json.loads(
        (ROOT / protocol["lineage"]["preservation_prompts_path"]).read_text()
    )
    modern_store, modern_reader = build_modern_preservation_reader(
        repo_root=ROOT, execution_config=execution_config
    )
    results = {}
    started = time.monotonic()
    try:
        with Int32ShardStore(
            encoded_dir=execution_config.encoded_dir,
            encoded_report=training_only_encoded_report(context["encoded_report"]),
        ) as store:
            reader = FrozenWindowReader(
                index_root=execution_config.window_index_dir,
                encoded_store=store,
                window_manifest=training_only_window_manifest(context["window_manifest"]),
            )
            for system_id in ("stage_3", "dpo"):
                manager = model.disable_adapter if system_id == "stage_3" else nullcontext
                with manager():
                    result = evaluate_all_gates(
                        model=model, reader=reader, modern_reader=modern_reader,
                        tokenizer=tokenizer, prompts=prompts,
                        device=torch.device("cuda:0"),
                    )
                results[system_id] = result
                print(
                    f"minerva-v7-dpo | system={system_id} completed="
                    f"{len(results)}/2 metrics={json.dumps(result['metrics'], sort_keys=True)} "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
    finally:
        modern_store.close()
    deltas = {
        key: results["dpo"]["metrics"][key] - results["stage_3"]["metrics"][key]
        for key in results["stage_3"]["metrics"]
    }
    payload = {
        "experiment_version": "minerva_7b_v7_dpo_preservation_v1",
        "stage_3_state_identity_sha256": state["state_identity_sha256"],
        "dpo_adapter_sha256": hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
        "systems": results, "dpo_minus_stage_3": deltas,
        "elapsed_seconds": time.monotonic() - started,
        "v7_test_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        f"minerva-v7-dpo | complete job=preservation_validation "
        f"elapsed={payload['elapsed_seconds']:.1f}s v7_test_accessed=False",
        flush=True,
    )


if __name__ == "__main__":
    main()
