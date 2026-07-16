import json
from pathlib import Path

import pytest

from sonnet_corpus.sidrac_editorial_audit import audit_sidrac_editorial_content


def test_audit_sidrac_editorial_content_reports_boundaries_and_notes(tmp_path: Path):
    source_path = tmp_path / "sidrac.txt"
    report_path = tmp_path / "audit.json"
    source_path.write_text(
        "Editor title page\n"
        "Questo è lo libro lo quale si chiama Sidracco\n"
        "Primary prose (15) continues.\n"
        "(15) Editorial variant note.\n"
        "_Conpiuto di scrivere a' dì XIIII di febraio, 1382\n"
        "INDICE\n"
        "Modern transcription note\n",
        encoding="utf-8",
    )

    report = audit_sidrac_editorial_content(
        source_path=source_path,
        report_path=report_path,
    )

    assert report.primary_text_start_line == 2
    assert report.primary_text_end_line == 5
    assert report.front_matter_character_count == len("Editor title page\n")
    assert report.candidate_inline_note_marker_count == 2
    assert report.candidate_note_line_count == 1
    assert report.candidate_note_line_examples == [
        {"line_number": 4, "text": "(15) Editorial variant note."}
    ]
    assert "Questo è lo libro" in report.primary_text_start_context
    assert "Conpiuto di scrivere" in report.primary_text_end_context

    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["end_matter_character_count"] == len("INDICE\nModern transcription note\n")


def test_audit_sidrac_editorial_content_rejects_missing_boundary_marker(tmp_path: Path):
    source_path = tmp_path / "sidrac.txt"
    source_path.write_text("no primary text marker\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Sidrac marker was not found"):
        audit_sidrac_editorial_content(
            source_path=source_path,
            report_path=tmp_path / "audit.json",
        )
