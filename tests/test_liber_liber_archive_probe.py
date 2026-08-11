import csv
import hashlib
import io
import json
import zipfile
from dataclasses import replace
from pathlib import Path

from sonnet_corpus.liber_liber_archive_probe import (
    LiberLiberArchiveProbeConfig,
    _apply_reviews,
    run_liber_liber_archive_probe,
)


class FakeResponse:
    def __init__(self, *, text="", content=b"", url="https://example.test/"):
        self.text = text
        self.content = content
        self.url = url

    def raise_for_status(self):
        return None


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.headers = {}
        self.calls = []

    def get(self, url, *, timeout):
        self.calls.append((url, timeout))
        return self.responses.pop(0)


def _write_csv(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _zip_text(text: str) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("opera.txt", text.encode("utf-8"))
    return stream.getvalue()


def _fixture(tmp_path: Path, *, low_quality=False):
    metadata = tmp_path / "data/metadata"
    processed = tmp_path / "data/processed"
    local = tmp_path / "data/local"
    reference_text = (
        "Titolo\n" + "Questo è il testo italiano della storia con parole comuni e chiare. " * 220
    )
    if low_quality:
        reference_text = "Titolo\n1234 XYZ\n"
    reference_path = processed / "bibit/part.txt"
    reference_path.parent.mkdir(parents=True)
    reference_path.write_text(reference_text, encoding="utf-8")
    protected_path = processed / "protected.txt"
    protected_path.write_text(
        "Questo è il testo italiano della storia con parole comuni e chiare.\n",
        encoding="utf-8",
    )

    inventory_fields = (
        "record_id", "wordpress_page_id", "title", "author", "landing_page_url",
        "period_bucket", "preliminary_role", "composition_decision", "license_url",
        "download_page_urls",
    )
    _write_csv(metadata / "inventory.csv", inventory_fields, [
        {
            "record_id": "ll:10", "wordpress_page_id": "10", "title": "Titolo",
            "author": "Autore", "landing_page_url": "https://example.test/work",
            "period_bucket": "origins_through_1800", "preliminary_role": "historical_general",
            "composition_decision": "eligible_fulltext_probe_inactive",
            "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
            "download_page_urls": (
                "https://example.test/download?type=opera_url_odt;"
                "https://example.test/download?type=opera_url_txt"
            ),
        },
        {
            "record_id": "ll:20", "wordpress_page_id": "20", "title": "Dialetto",
            "author": "Autore", "landing_page_url": "https://example.test/dialect",
            "period_bucket": "nineteenth_century",
            "preliminary_role": "conditioned_language_variants",
            "composition_decision": "conditioned_language_candidate_inactive",
            "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
            "download_page_urls": "https://example.test/download?type=opera_url_txt",
        },
    ])

    record_fields = (
        "object_id", "artifact_status", "shard_path", "byte_start", "byte_end",
    )
    _write_csv(processed / "bibit_records.csv", record_fields, [{
        "object_id": "one", "artifact_status": "text_materialized",
        "shard_path": reference_path.relative_to(tmp_path).as_posix(),
        "byte_start": "0", "byte_end": str(len(reference_text.encode("utf-8"))),
    }])
    range_fields = (
        "candidate_id", "artifact_status", "shard_path", "byte_start", "byte_end",
    )
    for name in (
        "bibit_sonnets.csv", "gutenberg_records.csv", "gutenberg_sonnets.csv",
        "wikisource_records.csv", "wikisource_sonnets.csv",
    ):
        _write_csv(processed / name, range_fields, [])
    probe_fields = ("ebook_id",)
    _write_csv(metadata / "gutenberg_previous.csv", probe_fields, [])
    _write_csv(metadata / "gutenberg_pass1b.csv", probe_fields, [])
    _write_csv(metadata / "broader.csv", ("source_id", "expected_clean_text_path"), [])
    _write_csv(metadata / "protected.csv", (
        "poem_id", "split_expanded_with_petrarch", "clean_text_path",
    ), [{
        "poem_id": "protected", "split_expanded_with_petrarch": "validation",
        "clean_text_path": protected_path.relative_to(tmp_path).as_posix(),
    }])

    config = LiberLiberArchiveProbeConfig(
        repo_root=tmp_path,
        inventory_path=metadata / "inventory.csv",
        cache_dir=local / "probe",
        output_csv_path=metadata / "probe.csv",
        review_csv_path=metadata / "review.csv",
        json_report_path=tmp_path / "reports/probe.json",
        markdown_report_path=tmp_path / "reports/probe.md",
        bibit_record_manifest_path=processed / "bibit_records.csv",
        bibit_sonnet_manifest_path=processed / "bibit_sonnets.csv",
        gutenberg_previous_probe_path=metadata / "gutenberg_previous.csv",
        gutenberg_previous_cache_dir=local / "gutenberg_previous",
        gutenberg_pass_1b_probe_path=metadata / "gutenberg_pass1b.csv",
        gutenberg_pass_1b_cache_dir=local / "gutenberg_pass1b",
        gutenberg_resolved_record_manifest_path=processed / "gutenberg_records.csv",
        gutenberg_resolved_sonnet_manifest_path=processed / "gutenberg_sonnets.csv",
        wikisource_resolved_record_manifest_path=processed / "wikisource_records.csv",
        wikisource_resolved_sonnet_manifest_path=processed / "wikisource_sonnets.csv",
        broader_sources_manifest_path=metadata / "broader.csv",
        protected_v6_sonnet_manifest_path=metadata / "protected.csv",
        expected_candidate_count=1,
        expected_conditioned_count=1,
        request_delay_seconds=0,
        min_cleaned_characters=1 if not low_quality else 1000,
        require_review_resolutions=not low_quality,
    )
    archive = _zip_text(reference_text)
    session = FakeSession([
        FakeResponse(
            text='<a href="https://media.test/opera.zip">TXT</a>',
            url="https://example.test/download?type=opera_url_txt",
        ),
        FakeResponse(content=archive, url="https://media.test/opera.zip"),
    ])
    return config, session


def _hashes(config):
    return [
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            config.output_csv_path, config.review_csv_path,
            config.json_report_path, config.markdown_report_path,
        )
    ]


def test_probe_prefers_txt_excludes_conditioned_and_reuses_cache_deterministically(tmp_path):
    config, session = _fixture(tmp_path)

    first = run_liber_liber_archive_probe(config, session=session, sleep=lambda _value: None)
    second = run_liber_liber_archive_probe(config, session=FakeSession([]))
    second_hashes = _hashes(config)
    third = run_liber_liber_archive_probe(config, session=FakeSession([]))

    assert first["archive_format_counts"] == {"txt_zip": 1}
    assert first["conditioned_excluded_record_ids"] == ["ll:20"]
    assert first["cache_status_counts"] == {"downloaded": 1}
    assert second["cache_status_counts"] == {"hit": 1}
    assert third == second
    assert _hashes(config) == second_hashes
    assert session.calls[0][0].endswith("type=opera_url_txt")
    assert first["cross_corpus_duplicate_pairs"][0]["source_kind"] == "bibit"
    assert first["cross_corpus_covered_candidate_count"] == 1
    assert first["protected_v6_overlap_record_count"] == 1
    assert first["probe_error_count"] == 0
    assert first["policy"]["text_activated"] is False


def test_probe_writes_bounded_unresolved_quality_review(tmp_path):
    config, session = _fixture(tmp_path, low_quality=True)

    report = run_liber_liber_archive_probe(config, session=session, sleep=lambda _value: None)
    review = list(csv.DictReader(config.review_csv_path.open(encoding="utf-8")))

    assert report["manual_review_anomaly_count"] == 1
    assert report["manual_review_unresolved_count"] == 1
    assert review[0]["record_id"] == "ll:10"
    assert "too_short" in review[0]["quality_review_flags"]


def test_probe_requires_review_for_documented_source_edition_changes(tmp_path):
    config, session = _fixture(tmp_path)
    config = replace(config, require_review_resolutions=False)
    session.responses[-1] = FakeResponse(content=_zip_text(
        "Note di edizione:\n"
        "L'ortografia è stata uniformata secondo criteri moderni.\n"
        + "Questo è il testo italiano della storia con parole comuni e chiare. " * 220
    ))

    report = run_liber_liber_archive_probe(
        config, session=session, sleep=lambda _value: None,
    )
    review = list(csv.DictReader(config.review_csv_path.open(encoding="utf-8")))

    assert report["manual_review_anomaly_count"] == 1
    assert review[0]["quality_review_flags"] == "edition_notes_present"


def test_apply_reviews_drops_blank_stale_anomaly_rows(tmp_path):
    config, _session = _fixture(tmp_path, low_quality=True)
    _write_csv(config.review_csv_path, (
        "record_id", "title", "quality_review_flags", "language_variety_flags",
        "manual_review_resolution", "manual_review_rationale",
    ), [{
        "record_id": "ll:10", "title": "Titolo", "quality_review_flags": "too_short",
        "language_variety_flags": "", "manual_review_resolution": "",
        "manual_review_rationale": "",
    }])

    assert _apply_reviews(config, []) == []
    assert list(csv.DictReader(config.review_csv_path.open(encoding="utf-8"))) == []


def test_apply_reviews_rejects_manually_reviewed_stale_anomaly_rows(tmp_path):
    config, _session = _fixture(tmp_path, low_quality=True)
    _write_csv(config.review_csv_path, (
        "record_id", "title", "quality_review_flags", "language_variety_flags",
        "manual_review_resolution", "manual_review_rationale",
    ), [{
        "record_id": "ll:10", "title": "Titolo", "quality_review_flags": "too_short",
        "language_variety_flags": "", "manual_review_resolution": "accept_primary_text",
        "manual_review_rationale": "Reviewed before the cleaner changed.",
    }])

    try:
        _apply_reviews(config, [])
    except ValueError as error:
        assert "manually reviewed non-anomaly IDs: ll:10" in str(error)
    else:
        raise AssertionError("expected stale manual review evidence to fail closed")
