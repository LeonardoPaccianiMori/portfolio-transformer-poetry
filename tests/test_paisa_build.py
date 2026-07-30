import gzip
import hashlib
import io
import json

import pytest

from sonnet_corpus.paisa_build import (
    PAISA_DOCUMENT_SEPARATOR,
    PaisaBuildConfig,
    build_paisa_corpus,
    canonicalize_paisa_document_text,
    iter_paisa_documents,
    split_for_paisa_fingerprint,
)


RAW_PAISA = """## PAISÀ example header
<text id="one" url="https://example.test/one">
Primo testo, con  spazi.\n
</text>
<text id="two" url="https://example.test/two">
Secondo testo distinto.
</text>
<text id="duplicate" url="https://example.test/duplicate">
Primo testo, con spazi.
</text>
<text id="empty" url="https://example.test/empty">

</text>
<text id="three" url="https://example.test/three">
Terzo testo distinto.
</text>
<text id="four" url="https://example.test/four">
Quarto testo distinto.
</text>
<text id="five" url="https://example.test/five">
Quinto testo distinto.
</text>
<text id="six" url="https://example.test/six">
Sesto testo distinto.
</text>
<text id="seven" url="https://example.test/seven">
Settimo testo distinto.
</text>
<text id="eight" url="https://example.test/eight">
Ottavo testo distinto.
</text>
</text>
"""


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload
        self.url = "https://example.test/paisa.raw.utf8.gz"
        self.headers = {"Content-Length": str(len(payload))}
        self.status_code = 200

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        for start in range(0, len(self.payload), chunk_size):
            yield self.payload[start : start + chunk_size]


class FakeSession:
    def __init__(self, payload):
        self.payload = payload

    def get(self, url, *, stream, timeout):
        assert url == "https://example.test/paisa.raw.utf8.gz"
        assert stream is True
        assert timeout == 120
        return FakeResponse(self.payload)


def _gzip_bytes(text):
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb") as handle:
        handle.write(text.encode("utf-8"))
    return output.getvalue()


def _config(tmp_path):
    return PaisaBuildConfig(
        release_url="https://example.test/paisa.raw.utf8.gz",
        processed_dir=tmp_path / "local/paisa_modern_italian_v1",
        report_path=tmp_path / "reports/build.json",
        temp_dir=tmp_path / "interim/paisa_build",
        validation_fraction=0.5,
        split_salt="test-salt",
        download_chunk_bytes=11,
        download_progress_bytes=17,
        document_progress_interval=2,
    )


def test_iter_paisa_documents_reads_preamble_and_required_provenance_fields(tmp_path):
    archive_path = tmp_path / "paisa.gz"
    archive_path.write_bytes(_gzip_bytes(RAW_PAISA.removesuffix("</text>\n")))

    documents = list(iter_paisa_documents(archive_path))

    assert documents[0].document_id == "one"
    assert documents[0].url == "https://example.test/one"
    assert documents[0].raw_text == "Primo testo, con  spazi.\n\n"
    assert len(documents) == 10


def test_iter_paisa_documents_preserves_literal_closing_markup_inside_text(tmp_path):
    archive_path = tmp_path / "paisa.gz"
    archive_path.write_bytes(
        _gzip_bytes(
            """<text id="one" url="https://example.test/one">
Esempio HTML: </text>
<textarea>scrivi qui</textarea>
</text>
<text id="two" url="https://example.test/two">
Secondo documento.
</text>
"""
        )
    )

    documents = list(iter_paisa_documents(archive_path))

    assert [document.document_id for document in documents] == ["one", "two"]
    assert "</text>\n<textarea>scrivi qui</textarea>" in documents[0].raw_text


def test_build_paisa_corpus_writes_local_text_and_public_aggregate_report(tmp_path):
    payload = _gzip_bytes(RAW_PAISA.removesuffix("</text>\n"))
    config = _config(tmp_path)
    progress = []

    report = build_paisa_corpus(
        config,
        session=FakeSession(payload),
        progress=progress.append,
    )

    assert report["source"]["release"]["sha256"] == hashlib.sha256(payload).hexdigest()
    counts = report["document_counts"]
    assert counts["parsed"] == 10
    assert counts["retained"] == 8
    assert counts["excluded_empty"] == 1
    assert counts["excluded_exact_duplicate"] == 1
    assert counts["train"] + counts["validation"] == 8
    assert counts["train"] > 0
    assert counts["validation"] > 0
    assert report["temporary_raw_and_interim_deleted_after_success"] is True
    assert not config.temp_dir.exists()

    train_text = (config.processed_dir / "train.txt").read_text(encoding="utf-8")
    validation_text = (config.processed_dir / "validation.txt").read_text(encoding="utf-8")
    assert train_text.count(PAISA_DOCUMENT_SEPARATOR) == counts["train"]
    assert validation_text.count(PAISA_DOCUMENT_SEPARATOR) == counts["validation"]
    assert (train_text + validation_text).count("Primo testo, con spazi.") == 1

    inventory = [
        json.loads(line)
        for line in (config.processed_dir / "document_attribution.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    duplicate = next(item for item in inventory if item["document_id"] == "duplicate")
    assert duplicate["status"] == "excluded_exact_duplicate"
    assert duplicate["duplicate_of_document_id"] == "one"
    assert duplicate["url"] == "https://example.test/duplicate"

    public_report = config.report_path.read_text(encoding="utf-8")
    assert "https://example.test/duplicate" not in public_report
    assert "document_attribution.jsonl" in public_report
    assert "data/interim/" not in public_report
    assert report["local_artifacts"]["train_text_path"] == str(
        config.processed_dir / "train.txt"
    )
    assert any(message.startswith("downloaded=") for message in progress)
    assert "inventory documents=10" in progress
    assert "writing documents=8" in progress


def test_fingerprint_split_is_deterministic_and_keeps_exact_duplicates_together():
    text = canonicalize_paisa_document_text("Stesso\t testo.\n")
    fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()

    first_split = split_for_paisa_fingerprint(
        fingerprint,
        validation_fraction=0.01,
        split_salt="paisa_modern_italian_v1",
    )
    second_split = split_for_paisa_fingerprint(
        fingerprint,
        validation_fraction=0.01,
        split_salt="paisa_modern_italian_v1",
    )

    assert first_split == second_split
    assert first_split in {"train", "validation"}


def test_iter_paisa_documents_rejects_an_unclosed_document(tmp_path):
    archive_path = tmp_path / "paisa.gz"
    archive_path.write_bytes(
        _gzip_bytes('<text id="one" url="https://example.test/one">\nnon chiuso\n')
    )

    with pytest.raises(ValueError, match="ended before closing"):
        list(iter_paisa_documents(archive_path))


def test_build_paisa_corpus_rejects_a_landing_page_before_parsing(tmp_path):
    class HtmlResponse(FakeResponse):
        def __init__(self):
            super().__init__(b"<html>landing page</html>")
            self.headers["Content-Type"] = "text/html"

    class HtmlSession:
        def get(self, url, *, stream, timeout):
            return HtmlResponse()

    with pytest.raises(ValueError, match="returned HTML"):
        build_paisa_corpus(_config(tmp_path), session=HtmlSession())


def test_build_paisa_corpus_resumes_a_retained_partial_download(tmp_path):
    payload = _gzip_bytes(RAW_PAISA.removesuffix("</text>\n"))
    config = _config(tmp_path)
    partial_size = len(payload) // 2
    partial_path = config.temp_dir / "raw/paisa.raw.utf8.gz.part"
    partial_path.parent.mkdir(parents=True)
    partial_path.write_bytes(payload[:partial_size])

    class ResumeResponse(FakeResponse):
        def __init__(self):
            super().__init__(payload[partial_size:])
            self.status_code = 206
            self.headers = {
                "Content-Length": str(len(payload) - partial_size),
                "Content-Range": f"bytes {partial_size}-{len(payload) - 1}/{len(payload)}",
            }

    class ResumeSession:
        def get(self, url, *, stream, timeout, headers):
            assert headers == {"Range": f"bytes={partial_size}-"}
            return ResumeResponse()

    report = build_paisa_corpus(config, session=ResumeSession())

    assert report["source"]["release"]["downloaded_bytes"] == len(payload)
    assert report["source"]["release"]["download_attempts"] == 1
    assert report["source"]["release"]["sha256"] == hashlib.sha256(payload).hexdigest()


def test_build_paisa_corpus_reuses_a_valid_archive_after_a_parser_interruption(tmp_path):
    payload = _gzip_bytes(RAW_PAISA.removesuffix("</text>\n"))
    config = _config(tmp_path)
    archive_path = config.temp_dir / "raw/paisa.raw.utf8.gz"
    archive_path.parent.mkdir(parents=True)
    archive_path.write_bytes(payload)

    class UnexpectedDownloadSession:
        def get(self, *args, **kwargs):
            raise AssertionError("a complete archive should be reused without another request")

    report = build_paisa_corpus(config, session=UnexpectedDownloadSession())

    assert report["source"]["release"]["acquisition_mode"] == "reused_complete_local_archive"
    assert report["source"]["release"]["download_attempts"] == 0
