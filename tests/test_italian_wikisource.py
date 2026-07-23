import pytest

from sonnet_corpus.italian_wikisource import (
    _sleep_with_progress,
    extract_ordered_direct_text_link_titles,
    extract_ordered_page_namespace_link_titles,
    extract_ordered_subpage_titles,
    extract_wikisource_prose_text,
    fetch_italian_wikisource_work,
    fetch_italian_wikisource_page_collection,
    fetch_italian_wikisource_two_level_page_collection,
    fetch_pinned_italian_wikisource_page_collection,
    select_edition_page_title,
    select_explicit_page_titles,
    select_work_subpage_titles,
    validate_work_boundaries,
    WikisourcePageRevision,
    WikisourceWorkSnapshot,
)


ROOT_TITLE = "Il Saggiatore"
ROOT_HTML = """
<div class="mw-parser-output">
  <div class="ws-noexport"><a title="Il Saggiatore/metadata">metadata</a></div>
  <ul>
    <li><a title="Il Saggiatore/Dedica">Dedica</a></li>
    <li><a title="Il Saggiatore/1">Capitolo I</a></li>
  </ul>
  <a title="Other work/1">unrelated work</a>
</div>
"""


class FakeResponse:
    def __init__(self, payload, status_code=200, headers=None):
        self.payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(
        self,
        revisions,
        rendered_html,
        resolved_titles=None,
        rate_limited_requests=0,
    ):
        self.revisions = revisions
        self.rendered_html = rendered_html
        self.resolved_titles = resolved_titles or {}
        self.rate_limited_requests = rate_limited_requests
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append(params)
        if self.rate_limited_requests:
            self.rate_limited_requests -= 1
            return FakeResponse({}, status_code=429, headers={"Retry-After": "0"})
        if params["action"] == "query":
            pages = []
            for title in params["titles"].split("|"):
                revision_id, timestamp = self.revisions[title]
                pages.append(
                    {
                        "title": self.resolved_titles.get(title, title),
                        "revisions": [
                            {
                                "revid": revision_id,
                                "timestamp": timestamp,
                            }
                        ],
                    }
                )
            return FakeResponse(
                {
                    "query": {
                        "pages": pages
                    }
                }
            )

        revision_id = int(params["oldid"])
        return FakeResponse(
            {
                "parse": {
                    "revid": revision_id,
                    "text": {"*": self.rendered_html[revision_id]},
                }
            }
        )


def test_extract_ordered_subpage_titles_uses_visible_work_order():
    assert extract_ordered_subpage_titles(ROOT_HTML, ROOT_TITLE) == [
        "Il Saggiatore/Dedica",
        "Il Saggiatore/1",
    ]


def test_extract_ordered_direct_text_link_titles_excludes_collection_navigation():
    html = """
    <div class="mw-parser-output">
      <a title="Sonetti romaneschi">root</a>
      <a title="Sonetti romaneschi/Sonetti del 1830">next index</a>
      <a title="A Pippo de R...">poem one</a>
      <a title="Pio Ottavo">poem two</a>
      <a title="Autore:Giuseppe Gioachino Belli">author</a>
      <div class="ws-noexport"><a title="Excluded poem">excluded</a></div>
    </div>
    """

    assert extract_ordered_direct_text_link_titles(
        html,
        collection_root_title="Sonetti romaneschi",
    ) == ["A Pippo de R...", "Pio Ottavo"]


def test_extract_ordered_page_namespace_link_titles_limits_selection_to_one_index():
    html = """
    <div class="mw-parser-output">
      <a title="Pagina:Giannone Vol.1.djvu/i">first</a>
      <a title="Pagina:Giannone Vol.1.djvu/1">second</a>
      <a title="Pagina:Giannone Vol.2.djvu/1">other volume</a>
      <a title="Indice:Giannone Vol.1.djvu">index</a>
    </div>
    """

    assert extract_ordered_page_namespace_link_titles(
        html,
        index_title="Indice:Giannone Vol.1.djvu",
    ) == [
        "Pagina:Giannone Vol.1.djvu/i",
        "Pagina:Giannone Vol.1.djvu/1",
    ]


def test_validate_work_boundaries_rejects_unexpected_last_page():
    with pytest.raises(ValueError, match="unexpected last Wikisource subpage"):
        validate_work_boundaries(
            ["Il Saggiatore/Dedica", "Il Saggiatore/1"],
            expected_first_subpage="Il Saggiatore/Dedica",
            expected_last_subpage="Il Saggiatore/2",
        )


def test_validate_work_boundaries_allows_a_dynamic_last_page():
    validate_work_boundaries(
        ["Il Saggiatore/Dedica", "Il Saggiatore/1"],
        expected_first_subpage="Il Saggiatore/Dedica",
    )


def test_select_work_subpage_titles_excludes_editorial_subtrees():
    discovered_titles = [
        "La scienza nuova - Volume I/Dedica dell'editore",
        "La scienza nuova - Volume I/Introduzione dell'editore/I",
        "La scienza nuova - Volume I/Titolo",
        "La scienza nuova - Volume I/Libro I",
    ]

    selected = select_work_subpage_titles(
        discovered_titles,
        selected_titles=None,
        excluded_prefixes=(
            "La scienza nuova - Volume I/Dedica dell'editore",
            "La scienza nuova - Volume I/Introduzione dell'editore",
        ),
    )

    assert selected == [
        "La scienza nuova - Volume I/Titolo",
        "La scienza nuova - Volume I/Libro I",
    ]


def test_select_work_subpage_titles_can_limit_a_collection_to_approved_prefixes():
    discovered_titles = [
        "Rime (Andreini)/Frontespizio",
        "Rime (Andreini)/Sonetto I",
        "Rime (Andreini)/Sonetti CLXXI-CLXXII",
        "Rime (Andreini)/Canzone I",
    ]

    selected = select_work_subpage_titles(
        discovered_titles,
        selected_titles=None,
        included_prefixes=("Rime (Andreini)/Sonetto", "Rime (Andreini)/Sonetti"),
    )

    assert selected == [
        "Rime (Andreini)/Sonetto I",
        "Rime (Andreini)/Sonetti CLXXI-CLXXII",
    ]


def test_select_work_subpage_titles_can_include_a_nested_approved_subtree():
    discovered_titles = [
        "Rime (Vittoria Colonna)/Sonetto I",
        "Rime (Vittoria Colonna)/Sonetti spirituali",
        "Rime (Vittoria Colonna)/Sonetti spirituali/Sonetto I",
        "Rime (Vittoria Colonna)/Canzone I",
    ]

    selected = select_work_subpage_titles(
        discovered_titles,
        selected_titles=None,
        included_prefixes=(
            "Rime (Vittoria Colonna)/Sonetto",
            "Rime (Vittoria Colonna)/Sonetti spirituali/Sonetto",
        ),
    )

    assert selected == [
        "Rime (Vittoria Colonna)/Sonetto I",
        "Rime (Vittoria Colonna)/Sonetti spirituali/Sonetto I",
    ]


def test_fetch_collection_can_select_direct_root_text_links():
    root_title = "Rime disperse"
    first_poem = "I: Mai non vo' più cantar"
    second_poem = "II: Epitafio"
    revisions = {
        root_title: (100, "2026-07-22T10:00:00Z"),
        first_poem: (101, "2026-07-22T10:01:00Z"),
        second_poem: (102, "2026-07-22T10:02:00Z"),
    }
    rendered_html = {
        100: (
            f'<div class="mw-parser-output"><a title="{first_poem}">first</a>'
            f'<a title="{second_poem}">second</a></div>'
        ),
        101: '<div class="mw-parser-output"><div class="poem">First poem.</div></div>',
        102: '<div class="mw-parser-output"><div class="poem">Second poem.</div></div>',
    }

    collection = fetch_italian_wikisource_page_collection(
        "https://example.test/rime-disperse",
        expected_title=root_title,
        direct_text_links=True,
        request_delay=0,
        session=FakeSession(revisions, rendered_html),
    )

    assert [page.revision.title for page in collection.pages] == [first_poem, second_poem]


def test_fetch_collection_recursively_selects_leaf_text_pages():
    root_title = "Istoria"
    first_book = "Istoria/Libro primo"
    chapter_one = "Istoria/Libro primo/Capitolo I"
    chapter_two = "Istoria/Libro primo/Capitolo II"
    revisions = {
        root_title: (100, "2026-07-23T10:00:00Z"),
        first_book: (101, "2026-07-23T10:01:00Z"),
        chapter_one: (102, "2026-07-23T10:02:00Z"),
        chapter_two: (103, "2026-07-23T10:03:00Z"),
    }
    rendered_html = {
        100: f'<div class="mw-parser-output"><a title="{first_book}">book</a></div>',
        101: (
            '<div class="mw-parser-output">'
            f'<a title="{chapter_one}">one</a>'
            f'<a title="{chapter_two}">two</a></div>'
        ),
        102: '<div class="mw-parser-output">First chapter.</div>',
        103: '<div class="mw-parser-output">Second chapter.</div>',
    }

    collection = fetch_italian_wikisource_page_collection(
        "https://example.test/istoria",
        expected_title=root_title,
        expected_first_subpage=first_book,
        recursive_subpages=True,
        request_delay=0,
        session=FakeSession(revisions, rendered_html),
    )

    assert [page.revision.title for page in collection.pages] == [chapter_one, chapter_two]


def test_select_work_subpage_titles_rejects_an_empty_filtered_scope():
    with pytest.raises(ValueError, match="selected no primary-text subpages"):
        select_work_subpage_titles(
            ["La scienza nuova - Volume I/Dedica dell'editore"],
            selected_titles=None,
            excluded_prefixes=("La scienza nuova - Volume I/Dedica dell'editore",),
        )


def test_select_explicit_page_titles_accepts_non_subpage_index_links_in_order():
    html = """
    <div class="mw-parser-output">
      <ul>
        <li><a title="Opera:Alla Sera">Alla Sera</a></li>
        <li><a title="Opera:A Zacinto">A Zacinto</a></li>
      </ul>
    </div>
    """

    selected = select_explicit_page_titles(
        html,
        ["Opera:Alla Sera", "Opera:A Zacinto"],
    )

    assert selected == ["Opera:Alla Sera", "Opera:A Zacinto"]


def test_select_explicit_page_titles_rejects_a_missing_index_link():
    with pytest.raises(ValueError, match="missing explicit pages"):
        select_explicit_page_titles(
            '<div class="mw-parser-output"><a title="Opera:Alla Sera">x</a></div>',
            ["Opera:A Zacinto"],
        )


def test_select_edition_page_title_returns_the_single_matching_primary_text_link():
    html = """
    <div class="mw-parser-output">
      <a title="Opera:Alla Sera">work record</a>
      <a title="Alla Sera (1803)">1803 edition</a>
      <a title="Alla Sera (1835)">1835 edition</a>
    </div>
    """

    assert select_edition_page_title(html, "(1835)") == "Alla Sera (1835)"


def test_select_edition_page_title_allows_a_colon_inside_a_poem_title():
    html = """
    <div class="mw-parser-output">
      <a title="Opera:Non son chi fui">work record</a>
      <a title="Non son chi fui: perì di noi gran parte (1835)">edition</a>
    </div>
    """

    assert (
        select_edition_page_title(html, "(1835)")
        == "Non son chi fui: perì di noi gran parte (1835)"
    )


def test_select_edition_page_title_accepts_a_source_name_before_the_edition_year():
    html = """
    <div class="mw-parser-output">
      <a title="Indice:Opere scelte di Ugo Foscolo II.djvu">source scan</a>
      <a title="Alla Musa (Foscolo 1835)">edition</a>
    </div>
    """

    assert select_edition_page_title(html, "1835)") == "Alla Musa (Foscolo 1835)"


def test_select_edition_page_title_rejects_missing_or_ambiguous_matches():
    with pytest.raises(ValueError, match="exactly one Wikisource edition page"):
        select_edition_page_title(
            '<div class="mw-parser-output"><a title="Alla Sera (1803)">x</a></div>',
            "(1835)",
        )

    with pytest.raises(ValueError, match="exactly one Wikisource edition page"):
        select_edition_page_title(
            """
            <div class="mw-parser-output">
              <a title="Alla Sera (1835)">first</a>
              <a title="Alla Sera, alternate (1835)">second</a>
            </div>
            """,
            "(1835)",
        )


def test_fetch_collection_follows_a_bibliographic_record_to_its_selected_edition():
    root_title = "Opera:Sonetti (Foscolo)"
    record_title = "Opera:Alla Sera"
    edition_title = "Alla Sera (1835)"
    revisions = {
        root_title: (100, "2026-07-15T10:00:00Z"),
        record_title: (101, "2026-07-15T10:01:00Z"),
        edition_title: (102, "2026-07-15T10:02:00Z"),
    }
    rendered_html = {
        100: f'<div class="mw-parser-output"><a title="{record_title}">Alla Sera</a></div>',
        101: f'<div class="mw-parser-output"><a title="{edition_title}">edition</a></div>',
        102: '<div class="mw-parser-output"><div class="poem">Actual poem text.</div></div>',
    }

    collection = fetch_italian_wikisource_page_collection(
        "https://it.wikisource.org/wiki/Sonetti_(Foscolo)",
        expected_title=root_title,
        explicit_page_titles=[record_title],
        edition_page_title_suffix="(1835)",
        request_delay=0,
        session=FakeSession(revisions, rendered_html),
    )

    assert [page.revision.title for page in collection.pages] == [edition_title]
    assert collection.pages[0].source_record_revision is not None
    assert collection.pages[0].source_record_revision.title == record_title


def test_fetch_two_level_collection_pins_indexes_and_leaf_pages_in_source_order():
    root_title = "Sonetti romaneschi"
    first_index = "Sonetti romaneschi/1818"
    second_index = "Sonetti romaneschi/1819"
    first_leaf = "Uno"
    second_leaf = "Due"
    third_leaf = "Tre"
    revisions = {
        root_title: (100, "2026-07-20T10:00:00Z"),
        first_index: (101, "2026-07-20T10:01:00Z"),
        second_index: (102, "2026-07-20T10:02:00Z"),
        first_leaf: (103, "2026-07-20T10:03:00Z"),
        second_leaf: (104, "2026-07-20T10:04:00Z"),
        third_leaf: (105, "2026-07-20T10:05:00Z"),
    }
    rendered_html = {
        100: (
            f'<div class="mw-parser-output"><a title="{first_index}">first</a>'
            f'<a title="{second_index}">second</a></div>'
        ),
        101: (
            f'<div class="mw-parser-output"><a title="{first_leaf}">one</a>'
            f'<a title="{second_leaf}">two</a>'
            f'<a title="{root_title}">root</a></div>'
        ),
        102: (
            f'<div class="mw-parser-output"><a title="{second_leaf}">two again</a>'
            f'<a title="{third_leaf}">three</a>'
            f'<a title="{first_index}">previous index</a></div>'
        ),
        103: '<div class="mw-parser-output"><div class="poem">One</div></div>',
        104: '<div class="mw-parser-output"><div class="poem">Two</div></div>',
        105: '<div class="mw-parser-output"><div class="poem">Three</div></div>',
    }

    collection = fetch_italian_wikisource_two_level_page_collection(
        "https://example.test/sonetti-romaneschi",
        expected_title=root_title,
        index_page_titles=[first_index, second_index],
        leaf_link_mode="direct_text_links",
        request_delay=0,
        session=FakeSession(revisions, rendered_html),
    )

    assert [revision.title for revision in collection.index_revisions] == [
        first_index,
        second_index,
    ]
    assert [page.revision.title for page in collection.pages] == [
        first_leaf,
        second_leaf,
        third_leaf,
    ]


def test_fetch_pinned_collection_validates_a_record_to_edition_snapshot():
    root_title = "Opera:Sonetti (Foscolo)"
    record_title = "Opera:Alla Sera"
    edition_title = "Alla Sera (1835)"
    revisions = {
        root_title: (100, "2026-07-15T10:00:00Z"),
        record_title: (101, "2026-07-15T10:01:00Z"),
        edition_title: (102, "2026-07-15T10:02:00Z"),
    }
    rendered_html = {
        100: f'<div class="mw-parser-output"><a title="{record_title}">Alla Sera</a></div>',
        101: f'<div class="mw-parser-output"><a title="{edition_title}">edition</a></div>',
        102: '<div class="mw-parser-output"><div class="poem">Pinned poem text.</div></div>',
    }
    snapshot = WikisourceWorkSnapshot(
        source_id="ws_foscolo_sonetti",
        landing_page_url="https://it.wikisource.org/wiki/Sonetti_(Foscolo)",
        title=root_title,
        scope="explicit_edition_pages",
        root_revision=WikisourcePageRevision(root_title, 100, "2026-07-15T10:00:00Z"),
        page_revisions=[WikisourcePageRevision(edition_title, 102, "2026-07-15T10:02:00Z")],
        source_record_revisions=[
            WikisourcePageRevision(record_title, 101, "2026-07-15T10:01:00Z")
        ],
        edition_page_title_suffix="1835)",
    )

    collection = fetch_pinned_italian_wikisource_page_collection(
        snapshot,
        request_delay=0,
        session=FakeSession(revisions, rendered_html),
    )

    assert collection.pages[0].revision.title == edition_title
    assert collection.pages[0].source_record_revision is not None
    assert collection.pages[0].source_record_revision.title == record_title


def test_fetch_pinned_collection_accepts_explicit_standalone_root_links():
    root_title = "Rime disperse"
    poem_title = "Spargi di lauri, palme e mirti foglie"
    revisions = {
        root_title: (100, "2026-07-15T10:00:00Z"),
        poem_title: (101, "2026-07-15T10:01:00Z"),
    }
    rendered_html = {
        100: (
            "<div class='mw-parser-output'>"
            f"<a title='{poem_title}'>first poem</a></div>"
        ),
        101: (
            "<div class='mw-parser-output'>"
            "<div class='poem'>Pinned poem text.</div></div>"
        ),
    }
    snapshot = WikisourceWorkSnapshot(
        source_id="ws_sannazaro_rime_disperse",
        landing_page_url="https://it.wikisource.org/wiki/Rime_disperse",
        title=root_title,
        scope="explicit_linked_pages",
        root_revision=WikisourcePageRevision(
            root_title, 100, "2026-07-15T10:00:00Z"
        ),
        page_revisions=[
            WikisourcePageRevision(
                poem_title, 101, "2026-07-15T10:01:00Z"
            )
        ],
    )

    collection = fetch_pinned_italian_wikisource_page_collection(
        snapshot,
        request_delay=0,
        session=FakeSession(revisions, rendered_html),
    )

    assert collection.pages[0].revision.title == poem_title


def test_extract_wikisource_prose_text_removes_site_wrappers():
    html = """
    <div class="mw-parser-output">
      <div class="quality-ns0">Questo testo e completo.</div>
      <div class="ws-noexport">license and metadata</div>
      <p>Testo primario.</p>
      <sup class="reference">[1]</sup>
      <div id="catlinks">categories</div>
      <p>Secondo paragrafo.</p>
    </div>
    """

    assert extract_wikisource_prose_text(html) == "Testo primario.\nSecondo paragrafo."


def test_fetch_work_pins_root_and_subpage_revisions_and_collects_text():
    revisions = {
        "Il Saggiatore": (100, "2026-07-15T10:00:00Z"),
        "Il Saggiatore/Dedica": (101, "2026-07-15T10:01:00Z"),
        "Il Saggiatore/1": (102, "2026-07-15T10:02:00Z"),
    }
    rendered_html = {
        100: ROOT_HTML,
        101: '<div class="mw-parser-output"><p>Dedica primaria.</p></div>',
        102: '<div class="mw-parser-output"><p>Capitolo primario.</p></div>',
    }
    session = FakeSession(revisions, rendered_html)

    fetched = fetch_italian_wikisource_work(
        "https://it.wikisource.org/wiki/Il_Saggiatore",
        expected_title="Il Saggiatore",
        expected_first_subpage="Il Saggiatore/Dedica",
        expected_last_subpage="Il Saggiatore/1",
        request_delay=0,
        session=session,
    )

    assert fetched.root_revision.revision_id == 100
    assert [revision.revision_id for revision in fetched.page_revisions] == [101, 102]
    assert fetched.text == (
        "## Il Saggiatore/Dedica\n\nDedica primaria.\n\n"
        "## Il Saggiatore/1\n\nCapitolo primario.\n"
    )
    assert fetched.raw_html_character_count == sum(len(value) for value in rendered_html.values())
    assert [call["oldid"] for call in session.calls if call["action"] == "parse"] == [
        "100",
        "101",
        "102",
    ]
    child_revision_query = next(
        call
        for call in session.calls
        if call["action"] == "query" and call["titles"] == "Il Saggiatore/Dedica|Il Saggiatore/1"
    )
    assert "rvlimit" not in child_revision_query


def test_fetch_work_rejects_empty_cleaned_subpage_text():
    revisions = {
        "Il Saggiatore": (100, "2026-07-15T10:00:00Z"),
        "Il Saggiatore/Dedica": (101, "2026-07-15T10:01:00Z"),
        "Il Saggiatore/1": (102, "2026-07-15T10:02:00Z"),
    }
    rendered_html = {
        100: ROOT_HTML,
        101: '<div class="mw-parser-output"><p>Dedica primaria.</p></div>',
        102: '<div class="mw-parser-output"><div class="ws-noexport">only wrapper</div></div>',
    }

    with pytest.raises(ValueError, match="empty primary text after cleaning"):
        fetch_italian_wikisource_work(
            "https://it.wikisource.org/wiki/Il_Saggiatore",
            expected_title="Il Saggiatore",
            expected_first_subpage="Il Saggiatore/Dedica",
            expected_last_subpage="Il Saggiatore/1",
            request_delay=0,
            session=FakeSession(revisions, rendered_html),
        )


def test_fetch_work_rejects_a_page_that_resolves_to_a_different_title():
    revisions = {"Il Saggiatore": (100, "2026-07-15T10:00:00Z")}
    session = FakeSession(
        revisions,
        rendered_html={},
        resolved_titles={"Il Saggiatore": "Another work"},
    )

    with pytest.raises(ValueError, match="unexpected Wikisource page title"):
        fetch_italian_wikisource_work(
            "https://it.wikisource.org/wiki/Il_Saggiatore",
            expected_title="Il Saggiatore",
            expected_first_subpage="Il Saggiatore/Dedica",
            expected_last_subpage="Il Saggiatore/1",
            request_delay=0,
            session=session,
        )


def test_fetch_work_retries_a_rate_limited_request():
    revisions = {
        "Il Saggiatore": (100, "2026-07-15T10:00:00Z"),
        "Il Saggiatore/Dedica": (101, "2026-07-15T10:01:00Z"),
        "Il Saggiatore/1": (102, "2026-07-15T10:02:00Z"),
    }
    rendered_html = {
        100: ROOT_HTML,
        101: '<div class="mw-parser-output"><p>Dedica primaria.</p></div>',
        102: '<div class="mw-parser-output"><p>Capitolo primario.</p></div>',
    }
    session = FakeSession(revisions, rendered_html, rate_limited_requests=1)

    fetched = fetch_italian_wikisource_work(
        "https://it.wikisource.org/wiki/Il_Saggiatore",
        expected_title="Il Saggiatore",
        expected_first_subpage="Il Saggiatore/Dedica",
        expected_last_subpage="Il Saggiatore/1",
        request_delay=0,
        session=session,
    )

    assert fetched.root_revision.revision_id == 100
    assert len(session.calls) == 6


def test_rate_limit_cooldown_emits_heartbeats(monkeypatch):
    sleeps = []
    messages = []
    monkeypatch.setattr("sonnet_corpus.italian_wikisource.sleep", sleeps.append)

    _sleep_with_progress(25, messages.append)

    assert sleeps == [10.0, 10.0, 5.0]
    assert messages == [
        "rate-limit cooldown remaining: 15 seconds",
        "rate-limit cooldown remaining: 5 seconds",
    ]
