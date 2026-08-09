import csv
from pathlib import Path

from sonnet_corpus.gutenberg import FetchedGutenbergText
from sonnet_corpus.gutenberg_fulltext_gate import (
    GutenbergFullTextGateConfig,
    run_gutenberg_fulltext_gate,
    select_gutenberg_fulltext_gate_sample,
)


def _row(
    ebook_id: str,
    role: str,
    *,
    status: str = "audit_then_deduplicate",
    overlap: str = "",
):
    return {
        "ebook_id": ebook_id,
        "title": f"Titolo {ebook_id}",
        "authors": "Autore",
        "preliminary_role": role,
        "period_bucket": "origins_through_1800",
        "inventory_status": status,
        "plain_text_url": f"https://example.test/{ebook_id}.txt",
        "possible_existing_work_matches": overlap,
        "intra_gutenberg_duplicate_ids": "",
    }


def test_select_gutenberg_fulltext_gate_sample_is_stratified_and_includes_signals():
    rows = [
        *[_row(str(index), "historical_general_candidate") for index in range(1, 6)],
        _row("10", "historical_non_sonnet_poetry_candidate", overlap="bibit:one"),
        _row("11", "sonnet_specialization_candidate"),
        _row("12", "date_and_role_review", status="review_work_publication_date"),
    ]

    first = select_gutenberg_fulltext_gate_sample(
        rows,
        sample_seed=9,
        role_quotas={
            "historical_general_candidate": 2,
            "historical_non_sonnet_poetry_candidate": 0,
            "sonnet_specialization_candidate": 0,
        },
    )
    second = select_gutenberg_fulltext_gate_sample(
        rows,
        sample_seed=9,
        role_quotas={
            "historical_general_candidate": 2,
            "historical_non_sonnet_poetry_candidate": 0,
            "sonnet_specialization_candidate": 0,
        },
    )

    assert [row["ebook_id"] for row in first] == [row["ebook_id"] for row in second]
    assert sum(row["preliminary_role"] == "historical_general_candidate" for row in first) == 2
    assert any(row["ebook_id"] == "10" for row in first)
    assert any(row["ebook_id"] == "11" for row in first)
    assert not any(row["ebook_id"] == "12" for row in first)


def _write_inventory(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_run_gutenberg_fulltext_gate_measures_and_reuses_cache(tmp_path):
    inventory = tmp_path / "inventory.csv"
    rows = [
        _row("1", "historical_general_candidate"),
        _row("2", "sonnet_specialization_candidate"),
    ]
    _write_inventory(inventory, rows)
    config = GutenbergFullTextGateConfig(
        repo_root=tmp_path,
        inventory_csv_path=inventory,
        cache_dir=tmp_path / "local/cache",
        sample_csv_path=tmp_path / "sample.csv",
        json_report_path=tmp_path / "report.json",
        markdown_report_path=tmp_path / "report.md",
        request_delay_seconds=0,
        min_cleaned_characters=100,
    )
    calls = []

    def fetch(ebook_id, **kwargs):
        calls.append(ebook_id)
        body = ("Il testo italiano che non si perde e che parla della vita. " * 20).strip()
        return FetchedGutenbergText(
            ebook_id=ebook_id,
            url=f"https://example.test/{ebook_id}.txt",
            text=(
                "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
                + body
                + "\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
            ),
        )

    report = run_gutenberg_fulltext_gate(config, fetch_text=fetch)
    rerun = run_gutenberg_fulltext_gate(
        config,
        fetch_text=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("cache should prevent fetch")
        ),
    )

    assert sorted(calls) == ["1", "2"]
    assert report["eligible_probe_candidate_count"] == 2
    assert report["sample_count"] == 2
    assert report["sample_status_counts"] == {"sample_quality_pass": 2}
    assert rerun["projected_total_cleaned_characters"] == report[
        "projected_total_cleaned_characters"
    ]
    assert config.sample_csv_path.is_file()
    assert config.markdown_report_path.is_file()


def test_run_gutenberg_fulltext_gate_measures_bibit_content_overlap(tmp_path):
    inventory = tmp_path / "inventory.csv"
    row = _row(
        "1",
        "historical_general_candidate",
        overlap="bibit:bibit000001",
    )
    _write_inventory(inventory, [row])
    body = ("Il testo italiano narra la storia della città e della sua gente. " * 40).strip()
    shard_path = tmp_path / "data/processed/bibit/part-0001.txt"
    shard_path.parent.mkdir(parents=True)
    shard_path.write_text(body + "\n", encoding="utf-8")
    bibit_manifest = tmp_path / "bibit_manifest.csv"
    _write_inventory(
        bibit_manifest,
        [
            {
                "object_id": "bibit000001",
                "shard_path": str(shard_path.relative_to(tmp_path)),
                "byte_start": "0",
                "byte_end": str(shard_path.stat().st_size),
            }
        ],
    )
    config = GutenbergFullTextGateConfig(
        repo_root=tmp_path,
        inventory_csv_path=inventory,
        cache_dir=tmp_path / "local/cache",
        sample_csv_path=tmp_path / "sample.csv",
        json_report_path=tmp_path / "report.json",
        markdown_report_path=tmp_path / "report.md",
        bibit_record_manifest_path=bibit_manifest,
        request_delay_seconds=0,
        min_cleaned_characters=100,
    )

    def fetch(ebook_id, **kwargs):
        return FetchedGutenbergText(
            ebook_id=ebook_id,
            url="https://example.test/1.txt",
            text=(
                "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
                + body
                + "\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
            ),
        )

    report = run_gutenberg_fulltext_gate(config, fetch_text=fetch)

    assert report["sample_metadata_overlap_count"] == 1
    assert report["sample_content_duplicate_signal_count"] == 1
    assert report["sample_reference_overlaps"] == [
        {
            "ebook_id": "1",
            "title": "Titolo 1",
            "possible_existing_work_matches": "bibit:bibit000001",
            "reference_overlap_metrics": (
                "bibit:bibit000001|containment=1.000000|jaccard=1.000000|exact=true"
            ),
            "max_reference_8gram_containment": 1.0,
            "content_duplicate_signal": True,
        }
    ]
    assert "| 1 | `bibit:bibit000001` | 1.0000 | yes |" in (
        config.markdown_report_path.read_text(encoding="utf-8")
    )
    with config.sample_csv_path.open(encoding="utf-8", newline="") as handle:
        result = next(csv.DictReader(handle))
    assert result["content_duplicate_signal"] == "True"
    assert float(result["max_reference_8gram_containment"]) == 1.0
