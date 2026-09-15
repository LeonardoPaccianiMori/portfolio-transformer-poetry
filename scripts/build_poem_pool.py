#!/usr/bin/env python3
"""Build the public poem pools for the website widget.

Two pools are produced from local artifacts only, so the widget is fully
static:

- pipeline: scheme-valid, repair-free sonnets from the released plan-then-poem
  pipeline (the distilled poem writer);
- published: a deterministic sample of the 2026-08 published model outputs,
  used to show the form gap that motivated the follow-up.

Every poem is scored with the project checker, and only checker metadata is
published (scheme, line count, accepted and failed hendecasyllables, rhyme
score). No AI-judge score is attached to individual poems.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_evaluation.sonnet_prosody_sealed import score_output

POOL_VERSION = "sonnet_widget_pool_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline-dir",
        type=Path,
        default=ROOT / "artifacts/local/grammar_ft/validation/stage2",
    )
    parser.add_argument(
        "--published-dir",
        type=Path,
        default=ROOT / "artifacts/local/minerva_7b_v7_dpo/final_test/generation",
    )
    parser.add_argument("--published-limit", type=int, default=300)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/local/public_widget",
    )
    return parser.parse_args()


def poem_record(payload: dict, source: str) -> dict:
    text = str(payload.get("text", "")).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    metrics = score_output(text)
    return {
        "id": hashlib.sha256(f"{source}|{text}".encode()).hexdigest()[:16],
        "opening": str(payload.get("opening_line") or (lines[0] if lines else "")),
        "text": text,
        "line_count": len(lines),
        "scheme": metrics["rhyme_scheme"],
        "accepted_lines": int(metrics["hendecasyllable_lines"]),
        "failed_lines": int(metrics["failed_lines"]),
        "rhyme_score": round(float(metrics["rhyme_score"] or 0.0), 4),
        "scheme_valid": bool(metrics["quatrain_ok"] and metrics["tercet_ok"]),
        "source": source,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_poems = []
    for path in sorted(args.pipeline_dir.glob("*.json")):
        if path.name == "complete.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        record = poem_record(payload, "plan_then_poem_distilled")
        if record["scheme_valid"] and record["line_count"] == 14:
            pipeline_poems.append(record)

    published_candidates = []
    for path in sorted(args.published_dir.glob("*.json")):
        if path.name == "complete.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        published_candidates.append(payload)
    step = max(1, len(published_candidates) // args.published_limit)
    published_poems = [
        poem_record(payload, f"published_2026_08::{payload.get('system_id', 'unknown')}")
        for payload in published_candidates[::step][: args.published_limit]
    ]

    pipeline_pool = {
        "schema_version": POOL_VERSION,
        "mode": "pipeline",
        "label": "Plan-then-poem pipeline (2026-09)",
        "note": (
            "Sonnets written by the released plan-then-poem pipeline: a rhyme "
            "plan is chosen first and then the poem is written to it, with no "
            "repair. Only sonnets whose rhyme scheme and fourteen-line "
            "structure pass the project checker are shown."
        ),
        "count": len(pipeline_poems),
        "poems": pipeline_poems,
    }
    published_pool = {
        "schema_version": POOL_VERSION,
        "mode": "published",
        "label": "Published model (2026-08)",
        "note": (
            "Outputs of the previously published model, sampled from its "
            "sealed test generation. In this sample the rhyme scheme does not "
            "hold, which is the gap the 2026-09 follow-up set out to close."
        ),
        "count": len(published_poems),
        "poems": published_poems,
    }

    pipeline_path = args.output_dir / "sonnet-pool-pipeline.json"
    published_path = args.output_dir / "sonnet-pool-published.json"
    for path, pool in ((pipeline_path, pipeline_pool), (published_path, published_pool)):
        path.write_text(json.dumps(pool, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    provenance = {
        "schema_version": POOL_VERSION,
        "generated": datetime.date.today().isoformat(),
        "pipeline_source": str(args.pipeline_dir.relative_to(ROOT)),
        "published_source": str(args.published_dir.relative_to(ROOT)),
        "pipeline_count": len(pipeline_poems),
        "published_count": len(published_poems),
        "sha256": {
            pipeline_path.name: hashlib.sha256(pipeline_path.read_bytes()).hexdigest(),
            published_path.name: hashlib.sha256(published_path.read_bytes()).hexdigest(),
        },
    }
    (args.output_dir / "sonnet-pool-provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"poem-pool | pipeline={len(pipeline_poems)} published={len(published_poems)} "
        f"-> {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
