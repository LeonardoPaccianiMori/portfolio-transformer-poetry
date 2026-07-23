import json

from sonnet_corpus.paisa_probe import probe_paisa_metadata


class FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        return None


class FakeSession:
    def __init__(self, text):
        self.text = text

    def get(self, url, timeout):
        assert url == "https://example.test/paisa"
        assert timeout == 30
        return FakeResponse(self.text)


HTML = """
<html><body>
  <p>The corpus contains approximately 380,000 documents coming from about 1,000
  different websites, for a total of about 250 million words.</p>
  <p>The compiled Paisà corpus is licensed under a Creative Commons
  Attribution-Noncommercial-ShareAlike license. It is partly used under
  Attribution-ShareAlike and partly used under Attribution-Noncommercial-ShareAlike.</p>
  <p>Documents are marked in the corpus by an XML "text" tag with "id" and "url" attributes.</p>
  <a href="https://example.test/download">download page</a>
  <p>For citing the corpus: Lyding et al. (2014) [link]</p>
</body></html>
"""


def test_probe_paisa_metadata_writes_license_and_provenance_facts(tmp_path):
    report_path = tmp_path / "paisa.json"

    report = probe_paisa_metadata(
        report_path=report_path,
        source_url="https://example.test/paisa",
        session=FakeSession(HTML),
    )

    result = report["result"]
    assert report["scope"] == "metadata_only_no_corpus_download"
    assert report["activation_status"] == "auxiliary_experiment_not_activated"
    assert result["status"] == "ok"
    assert result["document_count"] == 380_000
    assert result["website_count"] == 1_000
    assert result["reported_word_count"] == 250_000_000
    assert result["corpus_license"] == "CC BY-NC-SA"
    assert result["source_license_families"] == ["CC BY-SA", "CC BY-NC-SA"]
    assert result["document_provenance_fields"] == ["id", "url"]
    assert result["download_page_url"] == "https://example.test/download"
    assert json.loads(report_path.read_text(encoding="utf-8"))["result"]["status"] == "ok"


def test_probe_paisa_metadata_records_an_error_without_downloading_data(tmp_path):
    report_path = tmp_path / "paisa.json"

    report = probe_paisa_metadata(
        report_path=report_path,
        source_url="https://example.test/paisa",
        session=FakeSession("<html><body>incomplete</body></html>"),
    )

    assert report["result"]["status"] == "error"
    assert "corpus license" in report["result"]["error"]
