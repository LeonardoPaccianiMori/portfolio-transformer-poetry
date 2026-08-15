from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/export_portfolio_evidence.py"
EXPECTED = {
    "controlled-architecture-tradeoffs.json", "curriculum-stage-exposure.json",
    "sealed-test-automatic-outcomes.json", "stagewise-target-loss-reduction.json",
    "relative-parameter-movement.json", "representation-change-comparison.json",
    "preference-data-funnel.json", "validation-vs-sealed-test-gains.json",
    "blind-literary-deltas.json", "preservation-loss-changes.json",
}
FORBIDDEN_KEYS = {"poems", "openings", "generations", "preferences", "votes", "annotations", "mappings", "tensors", "corpus_text"}


def load_exporter():
    spec = importlib.util.spec_from_file_location("portfolio_exporter", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_export_is_deterministic_and_aggregate_only(tmp_path):
    exporter = load_exporter()
    first = tmp_path / "first"
    second = tmp_path / "second"
    evidence_1 = tmp_path / "evidence-1.json"
    evidence_2 = tmp_path / "evidence-2.json"
    exporter.export(first, evidence_1)
    exporter.export(second, evidence_2)
    assert evidence_1.read_bytes() == evidence_2.read_bytes()
    assert {path.name for path in first.iterdir()} == EXPECTED
    for name in EXPECTED:
        assert (first / name).read_bytes() == (second / name).read_bytes()
        payload = json.loads((first / name).read_text(encoding="utf-8"))
        assert payload["config"]["responsive"] is True
        assert payload["layout"]["margin"]
    evidence = json.loads(evidence_1.read_text(encoding="utf-8"))
    assert evidence["schema_version"] == 1
    serialized = json.dumps(evidence).lower()
    assert all(f'"{key}"' not in serialized for key in FORBIDDEN_KEYS)


def test_committed_evidence_matches_export(tmp_path):
    exporter = load_exporter()
    generated = tmp_path / "evidence.json"
    exporter.export(tmp_path / "charts", generated)
    assert generated.read_bytes() == (ROOT / "reports/public/portfolio_evidence_v1.json").read_bytes()
