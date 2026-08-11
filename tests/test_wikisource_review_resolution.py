import hashlib
from pathlib import Path

from sonnet_corpus.wikisource_review_resolution import (
    WikisourceReviewResolutionConfig,
    _Span,
    _partition,
    _resolve_post_segmentation_duplicates,
    _resolve_scan_rights,
    discover_wikisource_sonnet_candidates,
)


def _config(tmp_path: Path) -> WikisourceReviewResolutionConfig:
    values = {field: tmp_path / field for field in WikisourceReviewResolutionConfig.__dataclass_fields__ if field.endswith("_path") or field.endswith("_dir")}
    return WikisourceReviewResolutionConfig(repo_root=tmp_path, **values)


def _sonnet(label: str = "amor") -> str:
    endings = ("ato", "ente", "ente", "ato", "ato", "ente", "ente", "ato", "ivo", "ale", "ivo", "ale", "ivo", "ale")
    return "\n".join(f"{label} verso poetico numero {index} {ending}" for index, ending in enumerate(endings, 1))


def test_structural_sonnet_requires_rhyme_but_metadata_route_is_explicit():
    unrhymed = "\n".join(f"verso poetico {index} parola{index}" for index in range(14))
    assert discover_wikisource_sonnet_candidates(unrhymed, metadata_sonnet=False) == []
    assert len(discover_wikisource_sonnet_candidates(unrhymed, metadata_sonnet=True)) == 1
    inferred = discover_wikisource_sonnet_candidates(_sonnet(), metadata_sonnet=False)
    assert len(inferred) == 1
    assert inferred[0].source_kind == "structural_14_line_in_poetry_root"


def test_partition_is_complete_and_quarantines_only_exact_spans():
    spans = _partition(
        20,
        [_Span(5, 10, "exclude", "excluded", "bounded apparatus")],
        default_role="historical_general",
        include=True,
    )
    assert [(row.start, row.end) for row in spans] == [(0, 5), (5, 10), (10, 20)]
    assert "".join("x" * (row.end - row.start) for row in spans) == "x" * 20


def test_post_segmentation_duplicate_keeps_unique_sonnet_and_drops_remainder():
    sources = {"itws:1": "same\npoem one", "itws:2": "same\npoem two"}
    roots = []
    segments = []
    sonnets = []
    for number in (1, 2):
        root_id = f"itws:{number}"
        source = sources[root_id]
        roots.append({
            "work_root_id": root_id,
            "final_decision": "eligible_inactive_processed_build",
            "retained_broader_character_count": "4",
            "excluded_character_count": str(len(source) - 4),
            "source_character_count": str(len(source)),
            "canonical_reference_ids": "",
        })
        segments.extend([
            {"work_root_id": root_id, "character_start": "0", "character_end": "4", "segment_decision": "include_broader_text", "final_role": "historical_non_sonnet_poetry", "reason": "", "reference_ids": ""},
            {"work_root_id": root_id, "character_start": "5", "character_end": str(len(source)), "segment_decision": "materialize_standard_sonnet_inactive", "final_role": "standard_sonnets", "reason": "", "reference_ids": f"{root_id}:sonnet0001"},
        ])
        sonnets.append({
            "work_root_id": root_id,
            "candidate_decision": "eligible_standard_sonnet_inactive_pending_v7",
        })
    _resolve_post_segmentation_duplicates(roots, segments, sonnets, sources, threshold=0.8)
    loser = next(row for row in roots if row["work_root_id"] == "itws:2")
    assert loser["final_decision"] == "eligible_sonnets_only_inactive"
    assert loser["retained_broader_character_count"] == 0
    assert next(row for row in segments if row["work_root_id"] == "itws:2")["segment_decision"] == "exclude_post_segmentation_duplicate"


def test_post_segmentation_whitespace_only_remainder_is_explicitly_excluded():
    root_id = "itws:3"
    roots = [{
        "work_root_id": root_id,
        "final_decision": "eligible_inactive_processed_build",
        "retained_broader_character_count": "1",
        "excluded_character_count": "0",
        "source_character_count": "1",
        "canonical_reference_ids": "",
    }]
    segments = [{
        "work_root_id": root_id,
        "character_start": "0",
        "character_end": "1",
        "segment_decision": "include_broader_text",
        "final_role": "historical_non_sonnet_poetry",
        "reason": "",
        "reference_ids": "",
    }]

    _resolve_post_segmentation_duplicates(
        roots, segments, [], {root_id: "\n"}, threshold=0.8,
    )

    assert roots[0]["final_decision"] == "exclude_no_unique_material_after_segmentation"
    assert roots[0]["retained_broader_character_count"] == 0
    assert roots[0]["excluded_character_count"] == "1"
    assert segments[0]["segment_decision"] == "exclude_no_unique_material_after_segmentation"


def test_scan_rights_fail_closed_for_post_1930_named_editor(tmp_path):
    config = _config(tmp_path)
    scan = {
        "scan_title": "Modern critical edition.djvu",
        "scan_page_id": "10",
        "scan_url": "https://it.wikisource.org/wiki/Indice:Modern",
    }
    index_pages = {
        "Indice:Modern critical edition.djvu": {
            "revision_id": 20,
            "wikitext": "|Autore=Autore antico\n|Curatore=Curatore moderno\n|Anno=1961\n|Fonte={{Scansione utente}}\n",
        }
    }
    commons = {
        "File:Modern critical edition.djvu": {
            "pageid": 30,
            "imageinfo": [{
                "descriptionurl": "https://commons.wikimedia.org/wiki/File:Modern",
                "extmetadata": {
                    "LicenseShortName": {"value": "Public domain"},
                    "License": {"value": "pd"},
                    "Copyrighted": {"value": "False"},
                },
            }],
        },
        "__retrieved_utc__": {"value": "2026-08-11T00:00:00Z"},
    }
    rows, _by_title = _resolve_scan_rights(config, index_pages=index_pages, commons=commons, scans=[scan])
    assert rows[0]["rights_decision"] == "rights_hold_modern_edition_contributor"
