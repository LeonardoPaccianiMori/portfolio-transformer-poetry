#!/usr/bin/env python3
"""Freeze Minerva V7 activation probes and publish aggregate 8F evidence."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_training.minerva_7b_v7_execution import (
    build_activation_probe_manifest,
    build_public_execution_report,
    load_execution_config,
    render_public_execution_markdown,
)


def main() -> None:
    started = time.monotonic()
    execution_path = ROOT / "configs/minerva_7b_v7_execution.json"
    local_probe_path = ROOT / "data/local/minerva_7b_v7/activation_probes_v1.json"
    json_report_path = ROOT / "reports/minerva_7b_v7_execution_v1.json"
    markdown_report_path = ROOT / "reports/minerva_7b_v7_execution_v1.md"
    print(
        "minerva-v7-8f | start job=freeze-probes-and-execution-evidence "
        "device=cpu total_steps=3 progress_interval=1",
        flush=True,
    )
    execution = load_execution_config(execution_path, ROOT)
    print("minerva-v7-8f | step=1/3 progress=33.3% lineage=verified", flush=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "sapienzanlp/Minerva-7B-instruct-v1.0",
        revision="d1fc0f0e589ae879c5ac763e0e4206a4d14a3f6d",
        cache_dir=ROOT / "data/local/minerva_qlora/huggingface",
        local_files_only=True,
    )
    encoded_report = json.loads(
        (ROOT / execution["lineage"]["encoded_report_path"]).read_text(
            encoding="utf-8"
        )
    )
    probe = build_activation_probe_manifest(
        execution=execution,
        tokenizer=tokenizer,
        encoded_report=encoded_report,
        encoded_dir=ROOT / execution["local_paths"]["encoded_dir"],
        preservation_prompts_path=ROOT
        / "configs/minerva_7b_preservation_prompts.json",
        output_path=local_probe_path,
    )
    print(
        f"minerva-v7-8f | step=2/3 progress=66.7% probes={probe['probe_count']} "
        f"probe_sha256={probe['manifest_sha256']}",
        flush=True,
    )
    report = build_public_execution_report(
        execution_path=execution_path,
        repo_root=ROOT,
        local_probe_manifest_path=local_probe_path,
    )
    json_report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_report_path.write_text(
        render_public_execution_markdown(report), encoding="utf-8"
    )
    print(
        "minerva-v7-8f | step=3/3 progress=100.0% status={status} "
        "elapsed={elapsed:.1f}s output={output}".format(
            status=report["status"],
            elapsed=time.monotonic() - started,
            output=json_report_path,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
