import json
from collections import namedtuple

from sonnet_corpus.paisa_activation import (
    PaisaActivationAuditConfig,
    audit_paisa_release,
)


DESCRIPTION_HTML = """
<html><body>
  <p>The corpus contains approximately 380,000 documents coming from about 1,000
  different websites, for a total of about 250 million words.</p>
  <p>The compiled Paisà corpus is licensed under a Creative Commons
  Attribution-Noncommercial-ShareAlike license. It is partly used under
  Attribution-ShareAlike and partly used under Attribution-Noncommercial-ShareAlike.</p>
  <p>Documents are marked in the corpus by an XML \"text\" tag with \"id\" and \"url\" attributes.</p>
  <a href=\"https://example.test/download\">download page</a>
  <p>For citing the corpus: Lyding et al. (2014) [link]</p>
</body></html>
"""


class FakeResponse:
    def __init__(
        self,
        text="",
        *,
        url="https://example.test/page",
        status_code=200,
        headers=None,
    ):
        self.text = text
        self.url = url
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, *, description, release, artifact=None):
        self.description = description
        self.release = release
        self.artifact = artifact

    def get(self, url, timeout, allow_redirects=True):
        assert timeout == 30
        assert allow_redirects is True
        if url == "https://example.test/description":
            return self.description
        if url == "https://example.test/release":
            return self.release
        raise AssertionError(f"unexpected GET: {url}")

    def head(self, url, timeout, allow_redirects=True):
        assert url == "https://example.test/files/paisa.tar.gz"
        assert timeout == 30
        assert allow_redirects is True
        return self.artifact


DiskUsage = namedtuple("DiskUsage", "total used free")


def _disk_usage_with_free(free):
    return lambda path: DiskUsage(total=free, used=0, free=free)


def _config(tmp_path):
    return PaisaActivationAuditConfig(
        report_path=tmp_path / "report.json",
        description_url="https://example.test/description",
        release_url="https://example.test/release",
        acquisition_dir=tmp_path / "paisa",
    )


def test_audit_approves_local_acquisition_after_metadata_route_and_storage_checks(tmp_path):
    session = FakeSession(
        description=FakeResponse(DESCRIPTION_HTML),
        release=FakeResponse(
            '<a href="/files/paisa.tar.gz">download release</a>',
            url="https://example.test/release-page",
        ),
        artifact=FakeResponse(
            url="https://example.test/files/paisa.tar.gz",
            headers={"Content-Length": "100", "Content-Type": "application/gzip"},
        ),
    )

    report = audit_paisa_release(
        _config(tmp_path),
        session=session,
        disk_usage=_disk_usage_with_free(1_000),
    )

    assert report["activation_status"] == "approved_for_local_acquisition"
    assert report["release_route"]["artifact_url"] == "https://example.test/files/paisa.tar.gz"
    assert report["storage_preflight"]["required_bytes"] == 250
    assert report["license_and_attribution"]["corpus_license"] == "CC BY-NC-SA"
    assert "Do not commit PAISÀ text" in report["license_and_attribution"]["public_repository_policy"]
    assert json.loads((_config(tmp_path).report_path).read_text(encoding="utf-8"))["activation_status"] == (
        "approved_for_local_acquisition"
    )


def test_audit_records_an_anubis_release_block_without_downloading_payload(tmp_path):
    session = FakeSession(
        description=FakeResponse(DESCRIPTION_HTML),
        release=FakeResponse(
            "<html><body>Oh noes! Anubis bot protection</body></html>",
            url="https://example.test/protected",
        ),
    )

    report = audit_paisa_release(
        _config(tmp_path),
        session=session,
        disk_usage=_disk_usage_with_free(1_000),
    )

    assert report["activation_status"] == "blocked_release_access"
    assert report["release_route"]["status"] == "blocked"
    assert report["release_route"]["access_barrier"] == "anubis"
    assert report["release_route"]["artifact_url"] == ""


def test_audit_blocks_acquisition_when_the_release_size_cannot_fit_locally(tmp_path):
    session = FakeSession(
        description=FakeResponse(DESCRIPTION_HTML),
        release=FakeResponse(
            "",
            url="https://example.test/files/paisa.tar.gz",
            headers={
                "Content-Length": "1_000".replace("_", ""),
                "Content-Type": "application/gzip",
                "Content-Disposition": "attachment; filename=paisa.tar.gz",
            },
        ),
    )

    report = audit_paisa_release(
        _config(tmp_path),
        session=session,
        disk_usage=_disk_usage_with_free(2_499),
    )

    assert report["activation_status"] == "blocked_storage_preflight"
    assert report["storage_preflight"]["status"] == "insufficient_space"
    assert report["storage_preflight"]["required_bytes"] == 2_500


def test_audit_blocks_acquisition_when_the_official_route_hides_artifact_size(tmp_path):
    session = FakeSession(
        description=FakeResponse(DESCRIPTION_HTML),
        release=FakeResponse(
            "",
            url="https://example.test/files/paisa.tar.gz",
            headers={"Content-Type": "application/gzip"},
        ),
    )

    report = audit_paisa_release(
        _config(tmp_path),
        session=session,
        disk_usage=_disk_usage_with_free(10_000),
    )

    assert report["activation_status"] == "blocked_storage_preflight"
    assert report["storage_preflight"]["status"] == "unknown_artifact_size"
