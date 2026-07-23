import csv
from pathlib import Path


SHORTLIST_PATH = Path("data/metadata/sonnet_composition_shortlist.csv")
SOURCE_MANIFEST_PATH = Path("data/metadata/sonnet_expansion_sources_manifest.csv")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_composition_shortlist_records_every_required_gate_field():
    rows = read_csv(SHORTLIST_PATH)
    required_fields = {
        "source_id",
        "author",
        "landing_page_url",
        "language_variety",
        "period",
        "register_genre_content",
        "form_evidence",
        "license_or_reuse_status",
        "attribution_required",
        "indexed_item_count",
        "strict_sonnet_estimate_min",
        "strict_sonnet_estimate_max",
        "estimated_characters_min",
        "estimated_characters_max",
        "estimated_bpe_tokens_min",
        "estimated_bpe_tokens_max",
        "projected_share_if_added_alone_min_pct",
        "projected_share_if_added_alone_max_pct",
        "projected_author_share_in_recommended_bundle_min_pct",
        "projected_author_share_in_recommended_bundle_max_pct",
        "author_concentration_assessment",
        "duplicate_risk",
        "editorial_or_extraction_risk",
        "estimated_full_audit_runtime_minutes",
        "role",
        "gate_decision",
        "recommended_next_action",
        "rationale",
    }

    assert rows
    assert set(rows[0]) >= required_fields
    for row in rows:
        assert row["landing_page_url"].startswith("https://")
        assert row["language_variety"]
        assert row["period"]
        assert row["register_genre_content"]
        assert row["form_evidence"]
        assert row["license_or_reuse_status"]
        assert row["attribution_required"]
        assert row["author_concentration_assessment"]
        assert row["duplicate_risk"]
        assert row["editorial_or_extraction_risk"]
        assert row["rationale"]
        for field in (
            "indexed_item_count",
            "strict_sonnet_estimate_min",
            "strict_sonnet_estimate_max",
            "estimated_characters_min",
            "estimated_characters_max",
            "estimated_bpe_tokens_min",
            "estimated_bpe_tokens_max",
            "estimated_full_audit_runtime_minutes",
        ):
            assert int(row[field]) >= 0
        for field in (
            "projected_share_if_added_alone_min_pct",
            "projected_share_if_added_alone_max_pct",
            "projected_author_share_in_recommended_bundle_min_pct",
            "projected_author_share_in_recommended_bundle_max_pct",
        ):
            assert 0 <= float(row[field]) <= 100
        assert int(row["strict_sonnet_estimate_min"]) <= int(
            row["strict_sonnet_estimate_max"]
        )


def test_passed_core_candidates_are_standard_italian_and_ready_for_full_audit():
    rows = read_csv(SHORTLIST_PATH)
    passed = [row for row in rows if row["gate_decision"] == "passed_core"]

    assert {row["source_id"] for row in passed} == {
        "ws_andreini_rime_1601",
        "ws_ariosto_rime_varie_1857",
        "ws_colonna_rime_1760",
        "ws_sannazaro_rime_disperse",
        "ws_stampa_rime_1913",
    }
    assert all(row["role"] == "core_standard_italian" for row in passed)
    assert all(row["language_variety"] == "standard literary Italian" for row in passed)
    assert all(row["recommended_next_action"] == "include_in_combined_full_page_audit" for row in passed)
    assert all(float(row["projected_share_if_added_alone_max_pct"]) < 25 for row in passed)
    assert sum(int(row["strict_sonnet_estimate_min"]) for row in passed) == 715
    assert sum(int(row["strict_sonnet_estimate_max"]) for row in passed) == 757


def test_source_manifest_cannot_approve_or_audit_a_source_without_a_matching_gate_record():
    shortlist_by_id = {row["source_id"]: row for row in read_csv(SHORTLIST_PATH)}
    source_rows = read_csv(SOURCE_MANIFEST_PATH)
    legacy_activated_sources = {
        "ws_alfieri_rime_1912",
        "ws_foscolo_sonetti",
        "ws_varchi_infermita",
    }
    gate_required_rows = [
        row
        for row in source_rows
        if row["status"] in {"composition_gate_passed", "audit_then_include"}
        or (
            row["status"] == "activated"
            and row["source_id"] not in legacy_activated_sources
        )
    ]

    assert gate_required_rows
    for source in gate_required_rows:
        shortlist = shortlist_by_id[source["source_id"]]
        assert shortlist["gate_decision"] == "passed_core"
        assert shortlist["role"] == source["role"]
        assert shortlist["landing_page_url"] == source["landing_page_url"]
