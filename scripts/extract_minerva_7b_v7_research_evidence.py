#!/usr/bin/env python3
"""Dry-run or manually execute one verified V7 state's bounded GPU extraction."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_analysis.minerva_v7_extraction import extract_state, select_raw_attention_probe_hashes
from sonnet_analysis.minerva_v7_gpu_plan import build_gpu_extraction_plan, validate_probe_manifest
from sonnet_analysis.minerva_v7_runtime import (
    gpu_preflight,
    load_bf16_model_and_tokenizer,
    load_research_config,
    load_verified_state,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-id", required=True)
    parser.add_argument("--state-audit", type=Path, required=True)
    parser.add_argument("--research-config", type=Path, default=Path("configs/minerva_7b_v7_research.json"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/local/minerva_7b_v7_analysis/gpu_extraction"))
    parser.add_argument("--hourly-rate", type=float, required=True)
    parser.add_argument("--qualification-one-probe", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    config = load_research_config(args.research_config)
    state = load_verified_state(args.state_audit, args.state_id)
    probe_path = Path(config["probe_manifest_path"])
    probes = validate_probe_manifest(probe_path, expected_sha256=config["probe_manifest_sha256"])
    model_config = json.loads((Path(state["model_dir"]) / "config.json").read_text(encoding="utf-8"))
    audit = json.loads(args.state_audit.read_text(encoding="utf-8"))
    plan = build_gpu_extraction_plan(
        probe_manifest_path=probe_path,
        state_audit=audit,
        output_root=args.output_root,
        model_config=model_config,
        expected_probe_sha256=config["probe_manifest_sha256"],
    )
    job = next((row for row in plan["jobs"] if row["state_id"] == args.state_id), None)
    if job is None:
        raise ValueError("selected state has no ready extraction job")
    print(
        f"minerva-v7-research | start job=probe_extraction state={args.state_id} "
        f"execute={args.execute} probes={1 if args.qualification_one_probe else 48} "
        f"estimated_output_gib={job['estimated_output_bytes'] / 1024**3:.2f}", flush=True,
    )
    if not args.execute:
        print("minerva-v7-research | dry_run_complete no_model_loaded=True", flush=True)
        return
    preflight = gpu_preflight(
        output_root=args.output_root,
        required_output_bytes=int(job["estimated_output_bytes"]),
        hourly_rate=args.hourly_rate,
    )
    print(
        f"minerva-v7-research | preflight gpu={preflight['gpu_name']} "
        f"memory_mib={preflight['gpu_memory_mib']:.0f} free_disk_gib={preflight['free_disk_bytes']/1024**3:.1f}",
        flush=True,
    )
    import torch

    model, _tokenizer = load_bf16_model_and_tokenizer(
        state=state, config=config, device=torch.device("cuda:0")
    )
    selected_probes = probes["probes"][:1] if args.qualification_one_probe else probes["probes"]
    destination = (
        args.output_root / "qualification" / args.state_id
        if args.qualification_one_probe
        else args.output_root / args.state_id
    )
    completion = extract_state(
        model=model,
        probes=selected_probes,
        destination=destination,
        state_metadata={
            "state_id": args.state_id,
            "state_identity_sha256": state["state_identity_sha256"],
            "research_config": str(args.research_config),
        },
        device="cuda:0",
        block_count=32,
        raw_attention_layers=config["extraction"]["bounded_raw_attention_layers"],
        raw_attention_maximum_tokens=config["extraction"]["bounded_raw_attention_maximum_tokens"],
        raw_attention_probe_hashes=select_raw_attention_probe_hashes(probes["probes"]),
        progress=lambda message: print(f"minerva-v7-research | state={args.state_id} {message}", flush=True),
    )
    elapsed = time.monotonic() - started
    print(
        f"minerva-v7-research | complete state={args.state_id} probes={completion['probe_count']} "
        f"elapsed={elapsed:.1f}s cost_usd={elapsed/3600*args.hourly_rate:.2f}", flush=True,
    )


if __name__ == "__main__":
    main()
