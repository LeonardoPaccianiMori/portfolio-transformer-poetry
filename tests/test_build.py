from pathlib import Path

from sonnet_corpus.build import build_corpus


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.status_code = 200
        self.headers = {}

    def raise_for_status(self) -> None:
        return None


def test_build_deletes_temp_dirs_on_success(tmp_path: Path, monkeypatch):
    index_html = """
    <div class="mw-parser-output">
      <h2>Sonetti</h2>
      <ul>
        <li><a href="/wiki/Poesie_(Giacomo_da_Lentini)/Sonetti/Lo_giglio_quand%27%C3%A8_colto">XX</a></li>
      </ul>
    </div>
    """
    poem_html = """
    <div class="mw-parser-output">
      <div class="poem">
        uno<br/>due<br/>tre<br/>quattro<br/>cinque<br/>sei<br/>sette<br/>
        otto<br/>nove<br/>dieci<br/>undici<br/>dodici<br/>tredici<br/>quattordici
      </div>
    </div>
    """

    def fake_get(self, url, timeout, params=None):
        if url.endswith("Poesie_(Giacomo_da_Lentini)"):
            return FakeResponse(index_html)
        return FakeResponse(poem_html)

    monkeypatch.setattr("requests.Session.get", fake_get)

    report = build_corpus(
        repo_root=tmp_path,
        sources="giacomo",
        dataset="expanded_with_petrarch",
        force=True,
        keep_temp=False,
        request_delay=0,
        verbose=False,
    )

    assert report["included_rows"] == 1
    assert (tmp_path / "data/processed/poems/giacomo_lo_giglio_quand_è_colto.txt").is_file()
    assert (tmp_path / "data/metadata/poems_manifest.csv").is_file()
    assert not (tmp_path / "data/raw").exists()
    assert not (tmp_path / "data/interim").exists()


def test_excluded_rows_do_not_get_clean_text_path(tmp_path: Path, monkeypatch):
    index_html = """
    <div class="mw-parser-output">
      <h2>Sonetti</h2>
      <ul>
        <li><a href="/wiki/Poesie_(Giacomo_da_Lentini)/Sonetti/Lo_giglio_quand%27%C3%A8_colto">XX</a></li>
      </ul>
    </div>
    """
    poem_html = """
    <div class="mw-parser-output">
      <div class="poem">uno<br/>due<br/>tre</div>
    </div>
    """

    def fake_get(self, url, timeout, params=None):
        if url.endswith("Poesie_(Giacomo_da_Lentini)"):
            return FakeResponse(index_html)
        return FakeResponse(poem_html)

    monkeypatch.setattr("requests.Session.get", fake_get)

    build_corpus(
        repo_root=tmp_path,
        sources="giacomo",
        dataset="expanded_with_petrarch",
        force=True,
        keep_temp=False,
        request_delay=0,
        verbose=False,
    )

    manifest = (tmp_path / "data/metadata/poems_manifest.csv").read_text(encoding="utf-8")
    assert ",False,excluded,excluded," in manifest
    assert "data/processed/poems/giacomo_lo_giglio_quand_è_colto.txt" not in manifest
