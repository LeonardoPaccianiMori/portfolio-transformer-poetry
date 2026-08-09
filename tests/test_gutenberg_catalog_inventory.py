import csv
import json
from pathlib import Path

import pytest

from sonnet_corpus.gutenberg_catalog_inventory import (
    GutenbergCatalogInventoryConfig,
    classify_gutenberg_book,
    fetch_italian_gutenberg_catalog,
    inventory_italian_gutenberg_catalog,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.headers = {}
        self.urls = []

    def get(self, url, timeout):
        self.urls.append(url)
        return FakeResponse(self.payloads.pop(0))


def _book(
    ebook_id: int,
    title: str,
    *,
    author: str = "Autore, Test",
    birth: int | None = 1700,
    death: int | None = 1770,
    subjects: list[str] | None = None,
):
    return {
        "id": ebook_id,
        "title": title,
        "authors": [
            {"name": author, "birth_year": birth, "death_year": death}
        ],
        "subjects": subjects or [],
        "bookshelves": [],
        "languages": ["it"],
        "copyright": False,
        "media_type": "Text",
        "formats": {
            "text/plain; charset=utf-8": f"https://www.gutenberg.org/ebooks/{ebook_id}.txt.utf-8"
        },
        "download_count": 10,
    }


def test_fetch_italian_gutenberg_catalog_follows_all_pages():
    session = FakeSession(
        [
            {
                "count": 2,
                "next": "https://gutendex.com/books/?languages=it&page=2",
                "results": [_book(1, "Libro primo")],
            },
            {"count": 2, "next": None, "results": [_book(2, "Libro secondo")]},
        ]
    )
    messages = []

    catalog = fetch_italian_gutenberg_catalog(
        session=session,
        request_delay_seconds=0,
        progress=messages.append,
    )

    assert catalog["record_count"] == 2
    assert catalog["page_count"] == 2
    assert [book["id"] for book in catalog["books"]] == [1, 2]
    assert "page 2/2" in " ".join(messages)


def test_fetch_italian_gutenberg_catalog_rejects_count_change():
    session = FakeSession(
        [
            {
                "count": 2,
                "next": "https://gutendex.com/books/?languages=it&page=2",
                "results": [_book(1, "Libro primo")],
            },
            {"count": 3, "next": None, "results": [_book(2, "Libro secondo")]},
        ]
    )

    with pytest.raises(ValueError, match="count changed"):
        fetch_italian_gutenberg_catalog(
            session=session,
            request_delay_seconds=0,
        )


def test_classify_gutenberg_book_routes_form_period_and_review_evidence():
    sonnet = classify_gutenberg_book(
        _book(1, "Sonetti", subjects=["Italian poetry -- 18th century"])
    )
    poem = classify_gutenberg_book(
        _book(2, "Poema eroico", subjects=["Epic poetry, Italian"])
    )
    bridge = classify_gutenberg_book(
        _book(3, "Memorie", birth=1810, death=1880)
    )
    dialect = classify_gutenberg_book(
        _book(4, "Poesie in dialetto romanesco")
    )
    translation = classify_gutenberg_book(
        _book(5, "Opera", subjects=["Translations into Italian"])
    )
    historical_sicilian = classify_gutenberg_book(
        _book(6, "La guerra del Vespro Siciliano")
    )
    late_author_poetry = classify_gutenberg_book(
        _book(
            7,
            "Poesie giovanili",
            birth=1870,
            death=1930,
            subjects=["Italian poetry"],
        )
    )
    unlabeled_dialect_author = classify_gutenberg_book(
        _book(
            8,
            "Sonetti",
            author="Pascarella, Cesare",
            birth=1858,
            death=1940,
            subjects=["Sonnets, Italian"],
        )
    )

    assert sonnet["preliminary_role"] == "sonnet_specialization_candidate"
    assert poem["preliminary_role"] == "historical_non_sonnet_poetry_candidate"
    assert bridge["preliminary_role"] == "nineteenth_century_bridge_candidate"
    assert dialect["preliminary_role"] == "excluded_language_variety_metadata"
    assert translation["inventory_status"] == "review_translation_edition_date"
    assert historical_sicilian["preliminary_role"] == "historical_general_candidate"
    assert late_author_poetry["preliminary_role"] == "historical_non_sonnet_poetry_candidate"
    assert late_author_poetry["inventory_status"] == "review_work_publication_date"
    assert unlabeled_dialect_author["preliminary_role"] == "language_variety_review_required"
    assert (
        unlabeled_dialect_author["inventory_status"]
        == "review_language_variety_before_download"
    )


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_inventory_italian_gutenberg_catalog_cross_references_existing_data(tmp_path):
    bibit = tmp_path / "bibit.csv"
    _write_csv(
        bibit,
        [{"object_id": "bibit000001", "title": "Opera comune", "authors": "Test, Autore"}],
    )
    broader = tmp_path / "broader.csv"
    _write_csv(
        broader,
        [
            {
                "source_id": "pg_existing",
                "title": "Titolo registrato",
                "author": "Altro Autore",
                "ebook_id": "2",
            }
        ],
    )
    sonnets = tmp_path / "sonnets.csv"
    _write_csv(
        sonnets,
        [{"poem_id": "poem_1", "title_or_first_line": "Verso", "author": "Poeta"}],
    )
    config = GutenbergCatalogInventoryConfig(
        repo_root=tmp_path,
        snapshot_path=tmp_path / "snapshot.json",
        inventory_csv_path=tmp_path / "inventory.csv",
        json_report_path=tmp_path / "report.json",
        markdown_report_path=tmp_path / "report.md",
        bibit_record_manifest_path=bibit,
        broader_sources_manifest_path=broader,
        sonnet_manifest_path=sonnets,
        request_delay_seconds=0,
    )
    session = FakeSession(
        [
            {
                "count": 2,
                "next": None,
                "results": [
                    _book(1, "Opera comune"),
                    _book(2, "Titolo registrato", author="Altro Autore"),
                ],
            }
        ]
    )

    report = inventory_italian_gutenberg_catalog(config, session=session)

    assert report["record_count"] == 2
    assert report["records_with_existing_project_source_id"] == 1
    assert report["records_with_possible_existing_work_match"] == 2
    assert config.snapshot_path.is_file()
    assert config.inventory_csv_path.is_file()
    assert config.markdown_report_path.is_file()
    assert json.loads(config.json_report_path.read_text())["policy"]["metadata_only"] is True
    with config.inventory_csv_path.open(encoding="utf-8", newline="") as handle:
        rows = {row["ebook_id"]: row for row in csv.DictReader(handle)}
    assert rows["1"]["possible_existing_work_matches"] == "bibit:bibit000001"
    assert rows["2"]["existing_project_source_ids"] == "broader:pg_existing"
