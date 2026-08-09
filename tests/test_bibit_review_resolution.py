import csv
from pathlib import Path

from sonnet_corpus.bibit_review_resolution import (
    BibItReviewResolutionConfig,
    clean_bibit_editorial_brackets,
    coarse_sonnet_rhyme_evidence,
    resolve_bibit_review_queues,
)


def test_clean_bibit_editorial_brackets_preserves_supplied_text_only():
    text = "[1-3] vid[i] il testo [...] secondo [F] testimone."

    assert clean_bibit_editorial_brackets(text) == " vidi il testo ... secondo  testimone."


def test_coarse_sonnet_rhyme_evidence_recognizes_repeated_octave_and_sestet():
    text = "\n".join(
        [
            "Prima voce d'amore",
            "Seconda via smarrita",
            "Terza forma di vita",
            "Quarta torna al core",
            "Quinta parla d'amore",
            "Sesta resta smarrita",
            "Settima cerca vita",
            "Ottava chiude il core",
            "Nona guarda il cielo",
            "Decima passa il mare",
            "Undicesima porta luce",
            "Dodicesima alza il velo",
            "Tredicesima torna al mare",
            "Quattordicesima mi conduce",
        ]
    )

    signature, evidence = coarse_sonnet_rhyme_evidence(text)

    assert signature == "ABBAABBACDECDE"
    assert evidence is True


def test_resolve_bibit_review_queues_closes_record_and_structural_form_rows(tmp_path):
    record_audit = tmp_path / "records.csv"
    _write_csv(
        record_audit,
        [
            {
                "object_id": "bibit000001",
                "title": "Rime",
                "authors": "Autore Test",
                "assigned_role": "sonnet_only",
                "route": "historical_non_sonnet_poetry",
                "audit_status": "review_required",
                "audit_flags": "review_editorial_brackets",
                "routed_training_characters": "1200",
                "error": "",
                "duplicate_of_object_id": "",
                "landing_page_url": "https://example.test/bibit000001",
                "tei_sha256": "abc",
            }
        ],
    )
    sonnet_audit = tmp_path / "sonnets.csv"
    _write_csv(
        sonnet_audit,
        [
            {
                "candidate_id": "bibit000001:structural_0001",
                "object_id": "bibit000001",
                "title": "Rime",
                "authors": "Autore Test",
                "periods": "500",
                "source_kind": "structural_14_line",
                "tei_type": "poesia",
                "heading_path": "I",
                "line_count": "14",
                "stanza_pattern": "4+4+3+3",
                "status": "review_structural_form",
                "held_out_duplicate_poem_ids": "",
                "character_count": "400",
                "text_sha256": "def",
                "normalized_sha256": "ghi",
                "first_line": "Verso 1",
                "last_line": "Verso 14",
                "landing_page_url": "https://example.test/bibit000001",
            }
        ],
    )
    tei_cache = tmp_path / "tei"
    tei_cache.mkdir()
    stanzas = []
    next_line = 1
    for stanza_size in (4, 4, 3, 3):
        lines = "".join(
            f"<l>Verso {index} termina amore</l>"
            for index in range(next_line, next_line + stanza_size)
        )
        stanzas.append(f"<lg>{lines}</lg>")
        next_line += stanza_size
    (tei_cache / "bibit000001.xml").write_text(
        "<TEI.2><teiHeader><fileDesc><titleStmt><title>Rime</title>"
        "<author>Autore Test</author></titleStmt></fileDesc></teiHeader>"
        f"<text><body><div1 type='poesia'><head>I</head>{''.join(stanzas)}"
        "</div1></body></text></TEI.2>",
        encoding="utf-8",
    )
    config = BibItReviewResolutionConfig(
        repo_root=tmp_path,
        record_audit_csv_path=record_audit,
        sonnet_audit_csv_path=sonnet_audit,
        tei_cache_dir=tei_cache,
        record_decision_csv_path=tmp_path / "record_decisions.csv",
        sonnet_decision_csv_path=tmp_path / "sonnet_decisions.csv",
        json_report_path=tmp_path / "report.json",
        markdown_report_path=tmp_path / "report.md",
        progress_interval=1,
    )

    report = resolve_bibit_review_queues(config)

    assert report["unresolved_record_count"] == 0
    assert report["unresolved_sonnet_count"] == 0
    assert report["record_decision_counts"] == {
        "activate_core_with_bracket_cleanup": 1
    }
    assert report["sonnet_decision_counts"] == {
        "activate_inferred_standard_sonnet": 1
    }
    with config.sonnet_decision_csv_path.open(encoding="utf-8", newline="") as handle:
        decision = next(csv.DictReader(handle))
    assert decision["final_role"] == "sonnet_core_inferred_14_line"
    assert decision["cleaning_policy"] == "strip_editorial_square_delimiters_and_labels"


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
