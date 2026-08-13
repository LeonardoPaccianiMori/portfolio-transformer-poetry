#!/usr/bin/env python3
"""Run streamed, frequency-controlled embedding and LM-head comparison on CPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_analysis.minerva_v7_embedding import (
    analyze_embedding_pair, count_selected_token_ids, resolve_token_registry,
    resolve_verified_training_shards, verify_embedding_tokenizer,
)
from sonnet_analysis.minerva_v7_runtime import load_research_config, load_verified_comparison, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-audit", type=Path, required=True)
    parser.add_argument("--left-state-id", required=True)
    parser.add_argument("--right-state-id", required=True)
    parser.add_argument("--encoded-report", type=Path, default=Path("reports/minerva_7b_v7_encoded_data_v1.json"))
    parser.add_argument("--registry", type=Path, default=Path("configs/minerva_7b_v7_embedding_tokens.json"))
    parser.add_argument("--research-config", type=Path, default=Path("configs/minerva_7b_v7_research.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    config = load_research_config(args.research_config)
    if sha256_file(args.registry) != config["embedding_registry_sha256"]:
        raise ValueError("embedding token registry hash mismatch")
    if sha256_file(args.encoded_report) != config["encoded_data_report_sha256"]:
        raise ValueError("encoded data report hash mismatch")
    left, right, comparison_id = load_verified_comparison(
        args.state_audit, args.left_state_id, args.right_state_id
    )
    left_model_dir = Path(str(left["model_dir"]))
    right_model_dir = Path(str(right["model_dir"]))
    print(
        f"minerva-v7-research | start job=embedding_registry comparison={comparison_id}",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(left_model_dir, local_files_only=True)
    right_tokenizer = AutoTokenizer.from_pretrained(right_model_dir, local_files_only=True)
    verify_embedding_tokenizer(tokenizer)
    verify_embedding_tokenizer(right_tokenizer)
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    resolved = resolve_token_registry(tokenizer, registry)
    token_ids = [row["token_id"] for row in resolved["accepted"]]
    print(
        f"minerva-v7-research | registry accepted={len(resolved['accepted'])} "
        f"fragmented={len(resolved['rejected_fragmented_terms'])} counting_frequencies=True",
        flush=True,
    )
    shards = resolve_verified_training_shards(args.encoded_report, repo_root=ROOT)
    frequencies = count_selected_token_ids(shards, token_ids)
    report = analyze_embedding_pair(
        left_model_dir=left_model_dir,
        right_model_dir=right_model_dir,
        resolved_registry=resolved,
        frequencies=frequencies,
    )
    report["comparison_id"] = comparison_id
    report["left_state_identity_sha256"] = left["state_identity_sha256"]
    report["right_state_identity_sha256"] = right["state_identity_sha256"]
    report["frequency_shard_count"] = len(shards)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"minerva-v7-research | complete output={args.output}", flush=True)


if __name__ == "__main__":
    main()
