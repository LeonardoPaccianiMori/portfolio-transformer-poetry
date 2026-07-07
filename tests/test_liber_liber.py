import io
import zipfile

from sonnet_corpus.liber_liber import (
    discover_archive_url,
    discover_download_candidates,
    extract_odt_text,
    extract_txt_zip_text,
    fetch_liber_liber_text,
    strip_liber_liber_boilerplate,
)


class FakeResponse:
    def __init__(self, *, text="", content=b"", status_code=200, url=""):
        self.text = text
        self.content = content
        self.status_code = status_code
        self.url = url

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, responses):
        self.responses = responses
        self.headers = {}
        self.urls = []

    def get(self, url, timeout):
        self.urls.append(url)
        return self.responses.pop(0)


def make_zip(member_name: str, content: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(member_name, content)
    return buffer.getvalue()


def test_discover_download_candidates_prefers_txt_zip_before_odt():
    html = """
    <a href="/opere/download/?op=42&amp;type=opera_url_odt">ODT</a>
    <a href="/opere/download/?op=42&amp;type=opera_url_pdf">PDF</a>
    <a href="/opere/download/?op=42&amp;type=opera_url_txt">TXT</a>
    """

    candidates = discover_download_candidates(
        html,
        base_url="https://liberliber.it/work/",
    )

    assert [candidate.archive_format for candidate in candidates] == [
        "txt_zip",
        "odt",
    ]


def test_discover_archive_url_reads_hidden_download_value():
    html = """
    <input type="hidden"
           value="https://www.liberliber.eu/mediateca/work/text.zip"
           id="myUrl" />
    """

    url = discover_archive_url(
        html,
        base_url="https://liberliber.it/opere/download/",
        archive_format="txt_zip",
    )

    assert url == "https://www.liberliber.eu/mediateca/work/text.zip"


def test_extract_txt_zip_text_decodes_cp1252_literary_text():
    archive = make_zip("opera.txt", "più virtù".encode("cp1252"))

    text = extract_txt_zip_text(archive)

    assert text == "più virtù"


def test_extract_odt_text_skips_wrapper_and_table_of_contents():
    content_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <office:document-content
        xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
        xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
      <office:body>
        <office:text>
          <text:p>QUESTO E-BOOK</text:p>
          <text:table-of-content>
            <text:index-body><text:p>Indice</text:p></text:index-body>
          </text:table-of-content>
          <text:h>Nuova cronica</text:h>
          <text:p>Corpo<text:s text:c="2"/>del testo.<text:note><text:p>Nota editoriale</text:p></text:note></text:p>
        </office:text>
      </office:body>
    </office:document-content>
    """
    archive = make_zip("content.xml", content_xml.encode("utf-8"))

    text = extract_odt_text(archive)

    assert text == "Nuova cronica\nCorpo  del testo.\n"


def test_strip_liber_liber_boilerplate_starts_at_repeated_work_title():
    text = """Trattatello in laude di Dante
Questo e-book è distribuito da Liber Liber.
Informazioni sul progetto Manuzio: https://www.liberliber.it/

Trattatello in laude di Dante
di Giovanni Boccaccio

Solone fu reputato sapiente.
"""

    cleaned = strip_liber_liber_boilerplate(
        text,
        title="Trattatello in laude di Dante",
    )

    assert cleaned.startswith("Trattatello in laude di Dante\ndi Giovanni Boccaccio")
    assert "progetto Manuzio" not in cleaned


def test_fetch_liber_liber_text_follows_two_step_download():
    archive_bytes = make_zip(
        "opera.txt",
        (
            "Opera\nQuesto e-book di Liber Liber\nhttps://www.liberliber.it/\n"
            "Opera\nCorpo del testo.\n"
        ).encode("utf-8"),
    )
    session = FakeSession(
        [
            FakeResponse(
                text='<a href="/opere/download/?op=1&amp;type=opera_url_txt">TXT</a>'
            ),
            FakeResponse(
                text='<input value="https://media.test/opera.zip" id="myUrl">'
            ),
            FakeResponse(content=archive_bytes, url="https://media.test/opera.zip"),
        ]
    )

    fetched = fetch_liber_liber_text(
        "https://liberliber.it/opera/",
        title="Opera",
        session=session,
    )

    assert fetched.archive_format == "txt_zip"
    assert fetched.archive_url == "https://media.test/opera.zip"
    assert fetched.text == "Opera\nCorpo del testo.\n"
    assert len(session.urls) == 3


def test_fetch_liber_liber_text_falls_back_from_txt_zip_to_odt():
    content_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <office:document-content
        xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
        xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
      <office:body><office:text><text:h>Opera</text:h><text:p>Corpo ODT.</text:p></office:text></office:body>
    </office:document-content>
    """
    odt_bytes = make_zip("content.xml", content_xml.encode("utf-8"))
    session = FakeSession(
        [
            FakeResponse(
                text=(
                    '<a href="/download/?type=opera_url_txt">TXT</a>'
                    '<a href="/download/?type=opera_url_odt">ODT</a>'
                )
            ),
            FakeResponse(text='<input value="" id="myUrl">'),
            FakeResponse(text='<input value="https://media.test/opera.odt">'),
            FakeResponse(content=odt_bytes, url="https://media.test/opera.odt"),
        ]
    )

    fetched = fetch_liber_liber_text(
        "https://liberliber.it/opera/",
        title="Opera",
        session=session,
    )

    assert fetched.archive_format == "odt"
    assert fetched.text == "Opera\nCorpo ODT.\n"
