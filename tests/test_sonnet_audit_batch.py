import json
from pathlib import Path

import sonnet_corpus.sonnet_audit_batch as batch_module


def test_default_batch_contains_only_the_approved_core_cohort():
    assert batch_module.CORE_COHORT_AUDIT_IDS == (
        "ws_andreini_rime_1601",
        "ws_colonna_rime_1760",
        "ws_stampa_rime_1913",
        "ws_ariosto_rime_varie_1857",
        "ws_sannazaro_rime_disperse",
    )
    assert "ws_belli_sonetti_romaneschi" not in batch_module.CORE_COHORT_AUDIT_IDS


def test_batch_writes_a_summary_and_continues_after_a_source_error(
    tmp_path: Path, monkeypatch
):
    source_ids = ("first", "second", "third")
    monkeypatch.setattr(batch_module, "CORE_COHORT_AUDIT_IDS", source_ids)

    def fake_probe(**kwargs):
        if kwargs["source_id"] == "second":
            raise ValueError("expected source failure")
        return {
            "activation_status": "audit_then_include",
            "page_count": 3,
            "candidate_count": 3,
            "candidate_status_counts": {"eligible_14_lines": 3},
        }

    monkeypatch.setattr(batch_module, "probe_sonnet_wikisource_source", fake_probe)
    progress = []
    summary_path = tmp_path / "summary.json"

    summary = batch_module.run_remaining_sonnet_source_audits(
        source_manifest_path=tmp_path / "sources.csv",
        active_poems_manifest_path=tmp_path / "active.csv",
        repo_root=tmp_path,
        reports_directory=tmp_path / "reports",
        summary_path=summary_path,
        request_delay=0,
        progress=progress.append,
    )

    assert [result["status"] for result in summary["results"]] == ["ok", "error", "ok"]
    assert summary["results"][1]["error"] == "expected source failure"
    assert summary["results"][0]["report_path"] == "reports/first_probe.json"
    assert "source failed: second: expected source failure" in progress
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary
