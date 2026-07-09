import pytest

from sonnet_corpus.gutenberg import (
    candidate_plain_text_urls,
    fetch_gutenberg_text,
    strip_gutenberg_boilerplate,
)


class FakeResponse:
    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.headers = {}
        self.urls = []

    def get(self, url, timeout):
        self.urls.append(url)
        return self.responses.pop(0)


def test_candidate_plain_text_urls_starts_with_cache_url():
    urls = candidate_plain_text_urls("44549")

    assert urls[0] == "https://www.gutenberg.org/cache/epub/44549/pg44549.txt"
    assert "https://www.gutenberg.org/files/44549/44549-0.txt" in urls


def test_strip_gutenberg_boilerplate_keeps_only_book_body():
    text = "\n".join(
        [
            "Project Gutenberg header",
            "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***",
            "Capitolo primo",
            "Questo e il corpo.",
            "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***",
            "Project Gutenberg license text",
        ]
    )

    stripped = strip_gutenberg_boilerplate(text)

    assert stripped == "Capitolo primo\nQuesto e il corpo.\n"


def test_strip_gutenberg_boilerplate_returns_stripped_text_without_markers():
    stripped = strip_gutenberg_boilerplate("  Solo corpo  \n")

    assert stripped == "Solo corpo\n"


def test_strip_gutenberg_boilerplate_removes_legacy_footer_line():
    text = "\n".join(
        [
            "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***",
            "Capitolo primo",
            "Questo e il corpo.",
            "End of Project Gutenberg's Libro della divina dottrina",
        ]
    )

    stripped = strip_gutenberg_boilerplate(text)

    assert stripped == "Capitolo primo\nQuesto e il corpo.\n"


def test_fetch_gutenberg_text_tries_next_candidate_after_404():
    session = FakeSession(
        [
            FakeResponse("", status_code=404),
            FakeResponse("book body", status_code=200),
        ]
    )

    fetched = fetch_gutenberg_text("44549", session=session)

    assert fetched.ebook_id == "44549"
    assert fetched.text == "book body"
    assert fetched.url == "https://www.gutenberg.org/files/44549/44549-0.txt"
    assert len(session.urls) == 2


def test_fetch_gutenberg_text_raises_when_no_candidate_succeeds():
    session = FakeSession([FakeResponse("", status_code=404) for _ in range(4)])

    with pytest.raises(FileNotFoundError, match="could not fetch"):
        fetch_gutenberg_text("44549", session=session)
