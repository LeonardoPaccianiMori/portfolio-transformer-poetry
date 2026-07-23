import csv
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_audit_results_match_the_activated_core_source_manifest_records():
    audit_rows = read_csv(Path("data/metadata/sonnet_expansion_audit_results.csv"))
    manifest_rows = read_csv(Path("data/metadata/sonnet_expansion_sources_manifest.csv"))
    manifest_by_source = {row["source_id"]: row for row in manifest_rows}

    assert [row["source_id"] for row in audit_rows] == [
        "ws_andreini_rime_1601",
        "ws_colonna_rime_1760",
        "ws_stampa_rime_1913",
        "ws_ariosto_rime_varie_1857",
        "ws_sannazaro_rime_disperse",
    ]
    for row in audit_rows:
        manifest = manifest_by_source[row["source_id"]]
        assert manifest["status"] == "activated"
        assert row["activation_decision"] == "activated_in_sonnets_expanded_v5"
        assert int(row["scoped_page_count"]) == int(manifest["known_subpage_count"])
        assert int(row["candidate_count"]) == (
            int(row["eligible_14_line_count"]) + int(row["non_14_line_count"])
        )
        assert int(row["exact_duplicate_active_count"]) == 0


def test_audit_results_record_the_measured_eligible_total_and_projected_corpus_size():
    audit_rows = read_csv(Path("data/metadata/sonnet_expansion_audit_results.csv"))

    assert sum(int(row["eligible_14_line_count"]) for row in audit_rows) == 864
    assert 1011 + sum(int(row["eligible_14_line_count"]) for row in audit_rows) == 1875
