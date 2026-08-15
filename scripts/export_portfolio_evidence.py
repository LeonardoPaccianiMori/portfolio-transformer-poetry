#!/usr/bin/env python3
"""Export deterministic aggregate portfolio evidence and Plotly JSON files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLAIMS_PATH = ROOT / "reports/public/portfolio_claims_v1.yml"
EVIDENCE_PATH = ROOT / "reports/public/portfolio_evidence_v1.json"


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def load_claim_map() -> dict:
    return json.loads(CLAIMS_PATH.read_text(encoding="utf-8"))


def verify_sources(claim_map: dict) -> None:
    references = list(claim_map["claims"]) + list(claim_map["charts"].values())
    for item in references:
        path = ROOT / item["source_path"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != item["source_sha256"]:
            raise ValueError(f"source hash mismatch: {item['source_path']}")


def layout(title: str, *, yaxis: dict | None = None) -> dict:
    result = {
        "title": {"text": title, "x": 0.02, "xanchor": "left"},
        "autosize": True,
        "margin": {"l": 58, "r": 28, "t": 64, "b": 70},
        "legend": {"orientation": "h", "y": -0.25},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
    }
    if yaxis:
        result["yaxis"] = yaxis
    return result


def chart_payload(name: str, data: dict) -> dict:
    labels = data.get("labels", data.get("domains"))
    config = {"responsive": True, "displaylogo": False}
    if name == "controlled-architecture-tradeoffs":
        traces = [
            {"type": "bar", "name": "Best validation loss", "x": labels, "y": data["best_validation_loss"], "xaxis": "x", "yaxis": "y"},
            {"type": "bar", "name": "Repeated 4-gram ratio", "x": labels, "y": data["repeated_4gram_ratio"], "xaxis": "x2", "yaxis": "y2"},
        ]
        chart_layout = layout("Controlled architecture trade-offs")
        chart_layout.update({"grid": {"rows": 1, "columns": 2, "pattern": "independent"}, "yaxis": {"title": "Validation loss"}, "yaxis2": {"title": "Repeated 4-gram ratio"}})
    elif name == "curriculum-stage-exposure":
        traces = [
            {"type": "bar", "name": "Target tokens", "x": labels, "y": data["target_tokens"]},
            {"type": "scatter", "mode": "lines+markers", "name": "Windows", "x": labels, "y": data["windows"], "yaxis": "y2"},
            {"type": "scatter", "mode": "markers", "name": "Selected updates", "x": labels, "y": data["selected_updates"], "yaxis": "y2"},
        ]
        chart_layout = layout("Three-stage curriculum exposure", yaxis={"title": "Target tokens"})
        chart_layout["yaxis2"] = {"title": "Windows / updates", "overlaying": "y", "side": "right"}
    elif name == "sealed-test-automatic-outcomes":
        traces = [{"type": "bar", "name": "Stage 3", "x": labels, "y": data["stage3_percent"]}, {"type": "bar", "name": "DPO", "x": labels, "y": data["dpo_percent"]}]
        chart_layout = layout("Sealed-test automatic outcomes", yaxis={"title": "Percent", "rangemode": "tozero"})
        chart_layout["barmode"] = "group"
    elif name == "stagewise-target-loss-reduction":
        traces = [{"type": "bar", "name": key.title().replace("Stage", "Stage "), "x": data["domains"], "y": data[key]} for key in ("stage1", "stage2", "stage3")]
        chart_layout = layout("Stagewise target-loss reduction", yaxis={"title": "Loss reduction"})
        chart_layout["barmode"] = "group"
    elif name == "relative-parameter-movement":
        traces = [{"type": "bar", "x": labels, "y": data["relative_l2"], "name": "Relative L2"}]
        chart_layout = layout("Sequential relative parameter movement", yaxis={"title": "Relative L2", "type": "log"})
    elif name == "representation-change-comparison":
        traces = []
        for index, (key, title) in enumerate((("hidden_drift", "Hidden drift"), ("linear_cka", "Linear CKA"), ("top20_overlap", "Top-20 overlap")), start=1):
            traces.append({"type": "bar", "name": title, "x": labels, "y": data[key], "xaxis": "x" if index == 1 else f"x{index}", "yaxis": "y" if index == 1 else f"y{index}"})
        chart_layout = layout("Representation-change comparison")
        chart_layout.update({"grid": {"rows": 1, "columns": 3, "pattern": "independent"}, "yaxis": {"title": "Hidden drift"}, "yaxis2": {"title": "Linear CKA"}, "yaxis3": {"title": "Top-20 overlap"}})
    elif name == "preference-data-funnel":
        traces = [{"type": "funnel", "y": labels, "x": data["counts"], "name": "Records"}]
        chart_layout = layout("Preference-data funnel")
    elif name == "validation-vs-sealed-test-gains":
        lower = [value - low for value, low in zip(data["delta_points"], data["ci_low"])]
        upper = [high - value for value, high in zip(data["delta_points"], data["ci_high"])]
        traces = [{"type": "bar", "x": labels, "y": data["delta_points"], "name": "DPO change", "error_y": {"type": "data", "symmetric": False, "array": upper, "arrayminus": lower}}]
        chart_layout = layout("Validation versus sealed-test gains", yaxis={"title": "Percentage-point change", "zeroline": True})
    elif name == "blind-literary-deltas":
        lower = [value - low for value, low in zip(data["delta"], data["ci_low"])]
        upper = [high - value for value, high in zip(data["delta"], data["ci_high"])]
        traces = [{"type": "bar", "x": labels, "y": data["delta"], "name": "DPO − Stage 3", "error_y": {"type": "data", "symmetric": False, "array": upper, "arrayminus": lower}}]
        chart_layout = layout("Blind literary-score deltas", yaxis={"title": "Mean paired delta", "zeroline": True})
    elif name == "preservation-loss-changes":
        traces = [{"type": "bar", "x": labels, "y": data["dpo_minus_stage3"], "name": "DPO − Stage 3"}]
        chart_layout = layout("Preservation-loss changes", yaxis={"title": "Loss change", "zeroline": True})
    else:
        raise KeyError(name)
    chart_layout["meta"] = {"qualification": data["qualification"], "source_path": data["source_path"], "source_sha256": data["source_sha256"]}
    return {"data": traces, "layout": chart_layout, "config": config}


def export(output_dir: Path, evidence_path: Path) -> None:
    claim_map = load_claim_map()
    verify_sources(claim_map)
    evidence = {"schema_version": claim_map["schema_version"], "claims": claim_map["claims"], "charts": claim_map["charts"]}
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_bytes(canonical_bytes(evidence))
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in sorted(claim_map["charts"].items()):
        (output_dir / f"{name}.json").write_bytes(canonical_bytes(chart_payload(name, data)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/public/plotly")
    parser.add_argument("--evidence-path", type=Path, default=EVIDENCE_PATH)
    args = parser.parse_args()
    export(args.output_dir, args.evidence_path)
    print(f"portfolio-evidence | OK | charts=10 output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
