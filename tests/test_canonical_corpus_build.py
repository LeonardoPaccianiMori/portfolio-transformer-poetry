import hashlib
import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from sonnet_corpus.canonical_corpus_build import (
    _DeltaShardWriter,
    _blank_separated_blocks,
    _clean_verse_line,
    _canonical_delta_text,
    _is_standard_sonnet_layout,
    locate_reference_ranges,
    merge_character_ranges,
    positioned_tokens,
    remove_character_ranges,
)


def test_positioned_tokens_match_overlap_normalization_and_original_bounds():
    text = "L'Amor già È vivo."
    tokens = positioned_tokens(text)
    assert [token.value for token in tokens] == ["l", "amor", "gia", "e", "vivo"]
    assert [text[token.start:token.end] for token in tokens] == ["L", "Amor", "già", "È", "vivo"]


def test_localizer_finds_variant_reference_and_preserves_source_coordinates():
    reference = "uno due tre quattro cinque sei sette otto nove dieci undici dodici"
    source = f"prefazione\n{reference.replace('dieci', 'díeci')}\ncoda"
    ranges = locate_reference_ranges(source, reference)
    assert len(ranges) == 1
    assert ranges[0].coverage >= 0.8
    assert source[ranges[0].character_start:ranges[0].character_end].startswith("uno due")


def test_localizer_returns_every_distinct_occurrence():
    reference = "uno due tre quattro cinque sei sette otto nove dieci undici dodici"
    source = f"{reference}\nintermezzo molto lungo\n{reference}"
    ranges = locate_reference_ranges(source, reference)
    assert len(ranges) == 2
    assert all(source[item.character_start:item.character_end] == reference for item in ranges)


def test_sonnet_localization_expands_to_complete_lines():
    reference = "uno due tre quattro cinque sei sette otto nove dieci undici dodici"
    source = f"titolo\n$ {reference}\nnota"
    [localized] = locate_reference_ranges(source, reference, expand_to_lines=True)
    assert source[localized.character_start:localized.character_end] == f"$ {reference}\n"
    assert localized.line_expanded is True


def test_localizer_fails_closed_below_threshold():
    reference = "uno due tre quattro cinque sei sette otto nove dieci undici dodici tredici quattordici"
    source = "uno due tre quattro cinque sei sette otto testo completamente differente"
    assert locate_reference_ranges(source, reference, threshold=0.8) == []


def test_localizer_accepts_separate_fragments_only_when_aggregate_coverage_passes():
    reference = "uno due tre quattro cinque sei sette otto nove dieci undici dodici tredici quattordici quindici sedici"
    filler = " ".join(f"riempitivo{chr(97 + index)}" for index in range(20))
    source = f"uno due tre quattro cinque sei sette otto\n{filler}\nnove dieci undici dodici tredici quattordici quindici sedici"
    ranges = locate_reference_ranges(source, reference, threshold=0.8)
    assert len(ranges) == 2
    assert {item.localization_method for item in ranges} == {"fragmented_multi_span_shingle_alignment"}


def test_range_merge_and_removal_are_deterministic():
    assert merge_character_ranges([(2, 5), (4, 7), (9, 10)]) == [(2, 7), (9, 10)]
    assert remove_character_ranges("abcdefghijk", [(2, 5), (4, 7), (9, 10)]) == "abhik"


def test_delta_text_has_one_terminal_newline_without_text_normalization():
    assert _canonical_delta_text("Parola, già.\n\n") == "Parola, già.\n"


def test_invalid_range_is_rejected():
    with pytest.raises(ValueError, match="invalid character range"):
        merge_character_ranges([(4, 4)])


def test_blank_block_detector_never_uses_sliding_windows():
    text = "a\nb\n\nc\nd\n"
    assert list(_blank_separated_blocks(text)) == [
        (0, 4, ["a", "b"]),
        (5, 9, ["c", "d"]),
    ]


def test_verse_cleaner_removes_only_known_structural_line_markers():
    assert _clean_verse_line("$VORREI VOLER, $SIGNOR") == "VORREI VOLER, $SIGNOR"
    assert _clean_verse_line("10 e in voi posò l'alma") == "e in voi posò l'alma"
    assert _clean_verse_line("Quando i due lumi drizzai 37") == "Quando i due lumi drizzai"


def test_standard_sonnet_layout_rejects_short_form_and_prose_blocks():
    full = ["Questo verso ha misura sonettistica" for _ in range(14)]
    short = full[:]
    short[4] = "verso breve"
    prose = ["Questo periodo di prosa contiene molte parole e supera nettamente la normale misura del verso tradizionale" for _ in range(14)]
    assert _is_standard_sonnet_layout(full)
    quoted = full[:]
    quoted[0] = "- Questo verso ha misura sonettistica"
    assert _is_standard_sonnet_layout(quoted)
    assert not _is_standard_sonnet_layout(short)
    assert not _is_standard_sonnet_layout(prose)


def test_delta_writer_uses_bounded_deterministic_slices(tmp_path: Path):
    writer = _DeltaShardWriter(tmp_path / "role", "portable/role", 12)
    first = writer.add("one", "abc")
    second = writer.add("two", "def")
    third = writer.add("three", "12345678")
    reports = writer.close()
    assert first.path == second.path
    assert third.path != first.path
    assert (tmp_path / "role/delta-0001.txt").read_text() == "abc\ndef"
    assert reports[0]["sha256"] == hashlib.sha256(b"abc\ndef").hexdigest()


def test_checkpoint_7b_public_artifacts_reconcile_and_remain_inactive():
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "reports/canonical_italian_corpora_v1.json").read_text(encoding="utf-8"))
    assert report["record_universe_count"] == 4_646
    assert report["protected_v6_sonnet_count"] == 387
    assert report["segment_review_count"] == 264
    assert report["routing_row_count"] == 737
    assert report["new_standard_sonnet_count"] == 28
    assert report["token_count_status"] == "not_measured_pending_checkpoint_8_minerva_tokenization"
    assert report["modified_broader_protected_recheck_count"] == 71
    assert report["protected_pair_recheck_count"] == 71 * 387
    assert report["verification"] == {
        "all_264_reviews_accounted": True,
        "conditioned_material_included": False,
        "gpu_work_started": False,
        "mixture_weights_assigned": False,
        "protected_v6_count_preserved": True,
        "v7_created": False,
    }

    output = root / "data/processed/canonical_italian_corpora_v1"
    storage = list(csv.DictReader((output / "storage_manifest.csv").open(encoding="utf-8")))
    assert {row["storage_kind"] for row in storage} == {
        "existing_committed_slice", "checkpoint_7b_delta_slice",
    }
    assert not any("data/local/" in row["storage_path"] for row in storage)
    assert (output / "nineteenth_century_bridge").is_dir()

    routes = list(csv.DictReader((root / "data/metadata/cross_archive_sonnet_routing_v1.csv").open(encoding="utf-8")))
    decisions = Counter(row["routing_decision"] for row in routes)
    assert decisions["exclude_duplicate_broader_representation_reference_canonical_sonnet"] == 75
    assert decisions["retain_new_verified_standard_sonnet_inactive"] == 28
