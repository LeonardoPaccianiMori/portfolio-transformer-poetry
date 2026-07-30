"""Audit whether the PAISÀ release can be acquired under the project policy."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from .paisa_probe import PAISA_DESCRIPTION_URL, PaisaMetadataProbeResult, fetch_paisa_metadata


PAISA_RELEASE_URL = "https://hdl.handle.net/20.500.12124/3"
PAISA_RELEASE_ARTIFACT_URL = (
    "https://clarin.eurac.edu/repository/xmlui/bitstream/handle/20.500.12124/3/"
    "paisa.raw.utf8.gz?sequence=1&isAllowed=y"
)
ProgressCallback = Callable[[str], None]
_DOWNLOAD_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".gz", ".xz", ".bz2")
_DOWNLOAD_CONTENT_TYPES = (
    "application/octet-stream",
    "application/zip",
    "application/x-tar",
    "application/gzip",
    "application/x-gzip",
)
_ACCESS_BARRIER_MARKERS = (
    "anubis",
    "access denied",
    "bot protection",
    "you must enable javascript",
    "oh noes",
)


@dataclass(frozen=True)
class PaisaActivationAuditConfig:
    """Inputs and safety limits for one PAISÀ release-readiness audit."""

    report_path: Path = Path("reports/paisa_release_activation_audit.json")
    description_url: str = PAISA_DESCRIPTION_URL
    release_url: str = PAISA_RELEASE_URL
    acquisition_dir: Path = Path("data/local/pretraining/paisa")
    minimum_free_space_multiplier: float = 2.5
    request_timeout_seconds: int = 30


@dataclass(frozen=True)
class PaisaReleaseRoute:
    """What the official release route exposed without downloading a payload."""

    requested_url: str
    resolved_url: str
    status: str
    http_status: int | None
    content_type: str
    content_length_bytes: int | None
    artifact_url: str
    access_barrier: str
    error: str


@dataclass(frozen=True)
class PaisaStoragePreflight:
    """Disk-capacity result for retaining a compressed release and local outputs."""

    acquisition_dir: str
    available_bytes: int
    required_bytes: int | None
    minimum_free_space_multiplier: float
    status: str
    error: str


def audit_paisa_release(
    config: PaisaActivationAuditConfig,
    *,
    session: requests.Session | None = None,
    disk_usage: Callable[[str | Path], shutil._ntuple_diskusage] = shutil.disk_usage,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Write a public, metadata-only PAISÀ activation decision.

    The function never downloads a corpus payload. It confirms the official
    metadata, resolves the official release route, examines only HTTP metadata,
    and checks that the declared release would fit in the configured local area.
    """

    if config.minimum_free_space_multiplier <= 1:
        raise ValueError("minimum_free_space_multiplier must be greater than one")

    started_at = _utc_now()
    http = session or requests.Session()
    _write_progress(progress, f"fetching official description: {config.description_url}")
    try:
        metadata = fetch_paisa_metadata(source_url=config.description_url, session=http)
    except Exception as exc:
        metadata = _metadata_error(config.description_url, str(exc))

    _write_progress(progress, f"resolving official release route: {config.release_url}")
    route = _inspect_release_route(
        config.release_url,
        http=http,
        timeout=config.request_timeout_seconds,
    )
    storage = _check_storage_preflight(
        config.acquisition_dir,
        content_length_bytes=route.content_length_bytes,
        minimum_free_space_multiplier=config.minimum_free_space_multiplier,
        disk_usage=disk_usage,
    )
    activation_status, reason = _activation_decision(metadata, route, storage)

    report = {
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "scope": "release_activation_audit_no_corpus_payload_download",
        "activation_status": activation_status,
        "activation_reason": reason,
        "metadata": asdict(metadata),
        "release_route": asdict(route),
        "storage_preflight": asdict(storage),
        "license_and_attribution": {
            "permitted_project_role": "local_noncommercial_pretraining_corpus_only",
            "corpus_license": metadata.corpus_license,
            "source_license_families": metadata.source_license_families,
            "document_provenance_fields": metadata.document_provenance_fields,
            "document_level_license_classification": (
                "not_reported_on_the_official_description_page; retain each document's "
                "id and url as attribution inventory"
            ),
            "public_repository_policy": (
                "Do not commit PAISÀ text, derivative token files, or PAISÀ-derived "
                "checkpoints. Commit only source links, license notices, code, "
                "configuration, and aggregate reports."
            ),
            "required_credit": _required_credit(metadata),
        },
        "next_action": _next_action(activation_status, route, storage),
    }
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_progress(progress, f"wrote activation report: {config.report_path}")
    return report


def _inspect_release_route(
    release_url: str,
    *,
    http: requests.Session,
    timeout: int,
) -> PaisaReleaseRoute:
    try:
        response = http.get(release_url, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
    except Exception as exc:
        return PaisaReleaseRoute(
            requested_url=release_url,
            resolved_url="",
            status="error",
            http_status=None,
            content_type="",
            content_length_bytes=None,
            artifact_url="",
            access_barrier="",
            error=str(exc),
        )

    resolved_url = str(getattr(response, "url", release_url))
    headers = getattr(response, "headers", {})
    content_type = str(headers.get("Content-Type", "")).split(";", maxsplit=1)[0].lower()
    content_length_bytes = _parse_content_length(headers.get("Content-Length"))
    body = response.text
    barrier = _detect_access_barrier(body)
    if barrier:
        return PaisaReleaseRoute(
            requested_url=release_url,
            resolved_url=resolved_url,
            status="blocked",
            http_status=getattr(response, "status_code", None),
            content_type=content_type,
            content_length_bytes=content_length_bytes,
            artifact_url="",
            access_barrier=barrier,
            error="",
        )

    if _is_direct_download(resolved_url, headers, content_type):
        return PaisaReleaseRoute(
            requested_url=release_url,
            resolved_url=resolved_url,
            status="ok",
            http_status=getattr(response, "status_code", None),
            content_type=content_type,
            content_length_bytes=content_length_bytes,
            artifact_url=resolved_url,
            access_barrier="",
            error="",
        )

    artifact_url = _find_download_artifact_url(body, resolved_url)
    if artifact_url:
        return _inspect_artifact_headers(
            requested_url=release_url,
            resolved_url=resolved_url,
            artifact_url=artifact_url,
            http=http,
            timeout=timeout,
        )
    return PaisaReleaseRoute(
        requested_url=release_url,
        resolved_url=resolved_url,
        status="blocked",
        http_status=getattr(response, "status_code", None),
        content_type=content_type,
        content_length_bytes=content_length_bytes,
        artifact_url="",
        access_barrier="",
        error="official release page did not expose a direct downloadable corpus artifact",
    )


def _inspect_artifact_headers(
    *,
    requested_url: str,
    resolved_url: str,
    artifact_url: str,
    http: requests.Session,
    timeout: int,
) -> PaisaReleaseRoute:
    try:
        response = http.head(artifact_url, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
    except Exception as exc:
        return PaisaReleaseRoute(
            requested_url=requested_url,
            resolved_url=resolved_url,
            status="blocked",
            http_status=None,
            content_type="",
            content_length_bytes=None,
            artifact_url=artifact_url,
            access_barrier="",
            error=f"could not inspect downloadable artifact headers: {exc}",
        )

    headers = getattr(response, "headers", {})
    return PaisaReleaseRoute(
        requested_url=requested_url,
        resolved_url=resolved_url,
        status="ok",
        http_status=getattr(response, "status_code", None),
        content_type=str(headers.get("Content-Type", "")).split(";", maxsplit=1)[0].lower(),
        content_length_bytes=_parse_content_length(headers.get("Content-Length")),
        artifact_url=str(getattr(response, "url", artifact_url)),
        access_barrier="",
        error="",
    )


def _check_storage_preflight(
    acquisition_dir: Path,
    *,
    content_length_bytes: int | None,
    minimum_free_space_multiplier: float,
    disk_usage: Callable[[str | Path], shutil._ntuple_diskusage],
) -> PaisaStoragePreflight:
    acquisition_dir.mkdir(parents=True, exist_ok=True)
    available_bytes = disk_usage(acquisition_dir).free
    if content_length_bytes is None:
        return PaisaStoragePreflight(
            acquisition_dir=str(acquisition_dir),
            available_bytes=available_bytes,
            required_bytes=None,
            minimum_free_space_multiplier=minimum_free_space_multiplier,
            status="unknown_artifact_size",
            error="official release route did not provide Content-Length",
        )

    required_bytes = int(content_length_bytes * minimum_free_space_multiplier)
    if available_bytes < required_bytes:
        return PaisaStoragePreflight(
            acquisition_dir=str(acquisition_dir),
            available_bytes=available_bytes,
            required_bytes=required_bytes,
            minimum_free_space_multiplier=minimum_free_space_multiplier,
            status="insufficient_space",
            error="available disk space is below the configured safety requirement",
        )
    return PaisaStoragePreflight(
        acquisition_dir=str(acquisition_dir),
        available_bytes=available_bytes,
        required_bytes=required_bytes,
        minimum_free_space_multiplier=minimum_free_space_multiplier,
        status="ok",
        error="",
    )


def _activation_decision(
    metadata: PaisaMetadataProbeResult,
    route: PaisaReleaseRoute,
    storage: PaisaStoragePreflight,
) -> tuple[str, str]:
    if metadata.status != "ok":
        return "blocked_metadata_verification", metadata.error
    if route.status != "ok":
        return "blocked_release_access", route.access_barrier or route.error
    if storage.status != "ok":
        return "blocked_storage_preflight", storage.error
    return (
        "approved_for_local_acquisition",
        "official metadata, release route, and disk-capacity preflight passed; "
        "payload acquisition remains local and must retain document id/url attribution",
    )


def _next_action(
    activation_status: str,
    route: PaisaReleaseRoute,
    storage: PaisaStoragePreflight,
) -> str:
    if activation_status == "approved_for_local_acquisition":
        return (
            "Run the PAISÀ local acquisition and document-level attribution inventory builder "
            f"against {route.artifact_url}."
        )
    if route.access_barrier:
        return (
            "Do not attempt a payload download. Record the access barrier and obtain an "
            "officially accessible release route before continuing the PAISÀ rescue."
        )
    if storage.status != "ok":
        return "Do not download the release until the artifact size and local disk capacity are verified."
    return "Resolve the reported metadata or release-route issue before acquiring PAISÀ."


def _is_direct_download(url: str, headers: object, content_type: str) -> bool:
    header_map = headers if isinstance(headers, dict) else dict(headers)
    disposition = str(header_map.get("Content-Disposition", "")).lower()
    return (
        "attachment" in disposition
        or content_type in _DOWNLOAD_CONTENT_TYPES
        or _has_download_suffix(url)
    )


def _find_download_artifact_url(html: str, page_url: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for link in soup.select("a[href]"):
        href = str(link["href"]).strip()
        label = link.get_text(" ", strip=True).lower()
        if _has_download_suffix(href) or "download" in label:
            return urljoin(page_url, href)
    return ""


def _has_download_suffix(value: str) -> bool:
    lowered = value.lower().split("?", maxsplit=1)[0]
    return lowered.endswith(_DOWNLOAD_SUFFIXES)


def _detect_access_barrier(html: str) -> str:
    lowered = html.lower()
    for marker in _ACCESS_BARRIER_MARKERS:
        if marker in lowered:
            return marker
    return ""


def _parse_content_length(value: object) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(str(value))
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _metadata_error(source_url: str, error: str) -> PaisaMetadataProbeResult:
    return PaisaMetadataProbeResult(
        source_url=source_url,
        status="error",
        error=error,
        document_count=None,
        website_count=None,
        reported_word_count=None,
        corpus_license="",
        source_license_families=[],
        document_provenance_fields=[],
        download_page_url="",
        citation="",
    )


def _required_credit(metadata: PaisaMetadataProbeResult) -> str:
    citation = metadata.citation or "Lyding et al. (2014)"
    return (
        f"PAISÀ corpus, {citation}. Corpus license: {metadata.corpus_license}. "
        "Retain document id and URL attribution fields for all acquired documents."
    )


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _write_progress(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)
