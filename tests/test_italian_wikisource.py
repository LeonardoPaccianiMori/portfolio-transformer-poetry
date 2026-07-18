import pytest

from sonnet_corpus.italian_wikisource import (
    _sleep_with_progress,
    extract_ordered_subpage_titles,
    extract_wikisource_prose_text,
    fetch_italian_wikisource_work,
    select_work_subpage_titles,
    validate_work_boundaries,
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


def test_select_work_subpage_titles_rejects_an_empty_filtered_scope():
    with pytest.raises(ValueError, match="selected no primary-text subpages"):
        select_work_subpage_titles(
            ["La scienza nuova - Volume I/Dedica dell'editore"],
            selected_titles=None,
            excluded_prefixes=("La scienza nuova - Volume I/Dedica dell'editore",),
        )


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
