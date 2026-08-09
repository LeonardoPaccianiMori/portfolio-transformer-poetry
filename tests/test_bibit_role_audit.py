import csv
from pathlib import Path

from sonnet_corpus.bibit_composition_audit import (
    ROLE_HISTORICAL_GENERAL,
    ROLE_HISTORICAL_POETRY,
    ROLE_SONNET_ONLY,
)
from sonnet_corpus.bibit_role_audit import (
    BibItRoleAuditConfig,
    audit_bibit_tei_roles,
    normalize_exact_text,
    normalize_loose_text,
    requires_language_variety_review,
)


def _lines(prefix: str, count: int = 14) -> list[str]:
    return [f"{prefix} verso numero {index}" for index in range(1, count + 1)]


def _tei(
    object_id: str,
    title: str,
    *,
    prose: str = "",
    explicit_sonnet: list[str] | None = None,
    structural_verse: list[str] | None = None,
    other_verse: list[str] | None = None,
) -> bytes:
    def lg(lines: list[str], verse_type: str) -> str:
        return f'<lg type="{verse_type}">' + "".join(f"<l>{line}</l>" for line in lines) + "</lg>"

    body = f"<p>{prose}</p>" if prose else ""
    if explicit_sonnet is not None:
        body += lg(explicit_sonnet, "sonetto")
    if structural_verse is not None:
        body += lg(structural_verse, "poesia")
    if other_verse is not None:
        body += lg(other_verse, "canzone")
    return f"""<?xml version="1.0"?>
    <TEI.2>
      <teiHeader>
        <fileDesc>
          <titleStmt><title>{title}</title><author>Autore Test</author></titleStmt>
          <publicationStmt>
            <publisher>Biblioteca Italiana</publisher><idno>{object_id}</idno>
            <availability><p>Uso scientifico consentito.</p></availability>
          </publicationStmt>
          <sourceDesc><bibl><title>Edizione test</title><editor>Curatore</editor><idno>TEST</idno></bibl></sourceDesc>
        </fileDesc>
        <profileDesc><langUsage><language>Italiano</language></langUsage></profileDesc>
      </teiHeader>
      <text><body>{body}</body></text>
    </TEI.2>""".encode()


def _write_decisions(path: Path) -> None:
    rows = [
        ("bibit000001", "Trattato", ROLE_HISTORICAL_GENERAL),
        ("bibit000002", "Poema", ROLE_HISTORICAL_POETRY),
        ("bibit000003", "Rime", ROLE_SONNET_ONLY),
        ("bibit000004", "Lo cunto de li cunti", ROLE_HISTORICAL_GENERAL),
        ("bibit000005", "Alternata", "excluded"),
    ]
    fields = [
        "object_id",
        "canonical_status",
        "role",
        "title",
        "authors",
        "periods",
        "genres",
        "landing_page_url",
        "xml_url",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for object_id, title, role in rows:
            writer.writerow(
                {
                    "object_id": object_id,
                    "canonical_status": "alternate" if object_id == "bibit000005" else "selected",
                    "role": role,
                    "title": title,
                    "authors": "Autore Test",
                    "periods": "500",
                    "genres": "Poesia" if role != ROLE_HISTORICAL_GENERAL else "Trattati",
                    "landing_page_url": f"http://example.test/{object_id}",
                    "xml_url": f"http://example.test/xml/{object_id}",
                }
            )


def _write_sonnet_manifest(root: Path, held_out_lines: list[str]) -> Path:
    poem_path = root / "held_out.txt"
    poem_path.write_text("\n".join(held_out_lines) + "\n", encoding="utf-8")
    manifest = root / "sonnets.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "poem_id",
                "include_in_training",
                "split_expanded_with_petrarch",
                "clean_text_path",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "poem_id": "held_out_test",
                "include_in_training": "True",
                "split_expanded_with_petrarch": "test",
                "clean_text_path": poem_path.name,
            }
        )
    return manifest


def _config(root: Path, decisions: Path, manifest: Path) -> BibItRoleAuditConfig:
    return BibItRoleAuditConfig(
        repo_root=root,
        decision_csv_path=decisions,
        sonnet_manifest_path=manifest,
        tei_cache_dir=root / "local/tei",
        checkpoint_path=root / "local/checkpoint.json",
        record_csv_path=root / "records.csv",
        sonnet_csv_path=root / "sonnets_audit.csv",
        json_report_path=root / "report.json",
        markdown_report_path=root / "report.md",
        request_delay_seconds=0,
        max_retries=1,
        progress_interval=1,
        checkpoint_interval=1,
        min_training_characters=1,
    )


def test_bibit_role_audit_routes_all_three_corpora_and_blocks_held_out(tmp_path):
    decisions = tmp_path / "decisions.csv"
    _write_decisions(decisions)
    held_out = _lines("Held out")
    manifest = _write_sonnet_manifest(tmp_path, held_out)
    unique = _lines("Nuovo")
    documents = {
        "bibit000001": _tei(
            "bibit000001",
            "Trattato",
            prose="Testo generale sufficientemente lungo e senza contaminazioni.",
            explicit_sonnet=held_out,
        ),
        "bibit000002": _tei(
            "bibit000002",
            "Poema",
            structural_verse=_lines("Strutturale"),
            other_verse=["Primo verso lungo", "Secondo verso lungo"],
        ),
        "bibit000003": _tei(
            "bibit000003",
            "Rime",
            explicit_sonnet=unique,
            other_verse=[" ".join(["[nota]"] * 20), "Verso di canzone due"],
        ),
        "bibit000004": _tei(
            "bibit000004",
            "Lo cunto de li cunti",
            prose="Testo in una varietà che richiede revisione separata.",
        ),
    }
    fetched = []

    def fetch(object_id, **kwargs):
        fetched.append(object_id)
        return documents[object_id]

    messages = []
    config = _config(tmp_path, decisions, manifest)
    report = audit_bibit_tei_roles(
        config,
        fetch_tei=fetch,
        sleep=lambda _: None,
        progress=messages.append,
    )

    assert report["record_count"] == 4
    assert report["sonnet_candidate_count"] == 3
    assert report["explicit_sonnet_candidate_count"] == 2
    assert report["structural_sonnet_candidate_count"] == 1
    assert report["held_out_identity_conflict_count"] == 1
    assert report["sonnet_status_counts"]["eligible_explicit_nonduplicate"] == 1
    assert report["sonnet_status_counts"]["review_structural_form"] == 1
    assert report["sonnet_status_counts"]["excluded_held_out_identity_conflict"] == 1
    assert report["record_flag_counts"]["review_language_variety"] == 1
    assert report["record_flag_counts"]["review_editorial_brackets"] == 1
    assert set(fetched) == set(documents)
    assert all((config.tei_cache_dir / f"{object_id}.xml").is_file() for object_id in documents)
    assert config.record_csv_path.is_file()
    assert config.sonnet_csv_path.is_file()
    assert config.json_report_path.is_file()
    assert config.markdown_report_path.is_file()
    assert config.checkpoint_path.is_file()
    assert "record 4/4" in " ".join(messages)

    with config.record_csv_path.open(encoding="utf-8", newline="") as handle:
        records = {row["object_id"]: row for row in csv.DictReader(handle)}
    assert int(records["bibit000001"]["explicit_sonnet_count"]) == 1
    assert records["bibit000001"]["held_out_text_hits"] == ""
    assert records["bibit000003"]["route"] == ROLE_HISTORICAL_POETRY
    assert int(records["bibit000003"]["routed_training_characters"]) > 0


def test_bibit_role_audit_reuses_cached_tei_without_network(tmp_path):
    decisions = tmp_path / "decisions.csv"
    _write_decisions(decisions)
    manifest = _write_sonnet_manifest(tmp_path, _lines("Held out"))
    config = _config(tmp_path, decisions, manifest)
    config.tei_cache_dir.mkdir(parents=True)
    for object_id, title in (
        ("bibit000001", "Trattato"),
        ("bibit000002", "Poema"),
        ("bibit000003", "Rime"),
        ("bibit000004", "Lo cunto de li cunti"),
    ):
        (config.tei_cache_dir / f"{object_id}.xml").write_bytes(
            _tei(object_id, title, prose="Testo locale riutilizzato dalla cache.")
        )

    def unexpected_fetch(*args, **kwargs):
        raise AssertionError("network fetch should not run for cached TEI")

    report = audit_bibit_tei_roles(config, fetch_tei=unexpected_fetch)

    assert report["record_count"] == 4
    with config.record_csv_path.open(encoding="utf-8", newline="") as handle:
        assert {row["cache_status"] for row in csv.DictReader(handle)} == {"hit"}


def test_bibit_duplicate_normalizers_distinguish_exact_from_edition_level_text():
    first = "Perché l'amore è forte.\nSecondo verso."
    second = "Perche l amore e forte! Secondo verso."

    assert normalize_exact_text(first) != normalize_exact_text(second)
    assert normalize_loose_text(first) == normalize_loose_text(second)


def test_language_variety_review_avoids_geographic_false_positives():
    assert requires_language_variety_review(
        "Lo cunto de li cunti",
        ("Italiano",),
        ("Narrativa",),
    )
    assert not requires_language_variety_review(
        "Saggio storico sulla rivoluzione napoletana del 1799",
        ("Italiano", "Latino", "Francese"),
        ("Testi storici",),
    )
