"""Build a local, provenance-preserving PAISÀ pretraining component."""

from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import re
import shutil
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from .paisa_activation import PAISA_RELEASE_ARTIFACT_URL


PAISA_DOCUMENT_SEPARATOR = "<|endoftext|>"
ProgressCallback = Callable[[str], None]
_OPEN_TEXT_TAG = re.compile(r"^\s*<text\b(?P<attributes>[^>]*)>\s*$", re.IGNORECASE)
_CLOSE_TEXT_TAG = re.compile(r"^\s*</text>\s*$", re.IGNORECASE)
_ATTRIBUTE = re.compile(
    r"(?P<name>[A-Za-z_:][A-Za-z0-9_.:-]*)\s*=\s*"
    r"(?P<quote>['\"])(?P<value>.*?)(?P=quote)"
)


@dataclass(frozen=True)
class PaisaBuildConfig:
    """Configuration for one local PAISÀ extraction and split build."""

    corpus_version: str = "paisa_modern_italian_v1"
    release_url: str = PAISA_RELEASE_ARTIFACT_URL
    processed_dir: Path = Path("data/local/pretraining/paisa/paisa_modern_italian_v1")
    report_path: Path = Path("reports/paisa_modern_italian_v1_build_report.json")
    temp_dir: Path = Path("data/interim/paisa_modern_italian_v1_build")
    validation_fraction: float = 0.01
    split_salt: str = "paisa_modern_italian_v1"
    request_timeout_seconds: int = 120
    download_chunk_bytes: int = 1_048_576
    download_progress_bytes: int = 33_554_432
    document_progress_interval: int = 10_000
    max_download_attempts: int = 8
    download_retry_delay_seconds: float = 2.0


@dataclass(frozen=True)
class PaisaDocument:
    """One raw PAISÀ document with its required source attribution fields."""

    document_id: str
    url: str
    raw_text: str


@dataclass
class PaisaBuildCounts:
    """Aggregate counters retained in the public build report."""

    parsed_documents: int = 0
    retained_documents: int = 0
    empty_documents: int = 0
    exact_duplicate_documents: int = 0
    retained_characters: int = 0
    retained_words: int = 0
    train_documents: int = 0
    validation_documents: int = 0
    train_characters: int = 0
    validation_characters: int = 0
    train_words: int = 0
    validation_words: int = 0


def build_paisa_corpus(
    config: PaisaBuildConfig,
    *,
    session: requests.Session | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Download, validate, split, and locally publish the PAISÀ corpus.

    The release archive and all text-bearing outputs remain under ``data/local``.
    The separately written report intentionally contains only aggregate metadata.
    """

    _validate_config(config)
    started_at = _utc_now()
    temp_root = config.temp_dir
    raw_dir = temp_root / "raw"
    interim_dir = temp_root / "interim"
    staged_processed_dir = temp_root / "processed"
    archive_path = raw_dir / "paisa.raw.utf8.gz"
    download_part_path = raw_dir / "paisa.raw.utf8.gz.part"
    inventory_path = staged_processed_dir / "document_attribution.jsonl"
    train_path = staged_processed_dir / "train.txt"
    validation_path = staged_processed_dir / "validation.txt"
    local_report_path = staged_processed_dir / "build_report.json"

    _validate_deletable_directory(temp_root, label="temp_dir")
    _validate_deletable_directory(config.processed_dir, label="processed_dir")
    lock_handle = _acquire_build_lock(temp_root)
    try:
        _prepare_temp_tree(
            temp_root,
            raw_dir,
            interim_dir,
            staged_processed_dir,
            reusable_archive_path=archive_path,
            resumable_download_path=download_part_path,
        )
    except Exception:
        _release_build_lock(lock_handle)
        raise

    try:
        _write_progress(
            progress,
            "start "
            f"corpus_version={config.corpus_version} "
            f"validation_fraction={config.validation_fraction:.2%} "
            f"document_progress_interval={config.document_progress_interval}",
        )
        downloaded = _download_release(
            release_url=config.release_url,
            archive_path=archive_path,
            download_part_path=download_part_path,
            session=session,
            timeout=config.request_timeout_seconds,
            chunk_bytes=config.download_chunk_bytes,
            progress_bytes=config.download_progress_bytes,
            max_attempts=config.max_download_attempts,
            retry_delay_seconds=config.download_retry_delay_seconds,
            progress=progress,
        )
        _write_progress(progress, "first pass: validating documents and writing attribution inventory")
        decisions, counts = _inventory_documents(
            archive_path=archive_path,
            inventory_path=inventory_path,
            validation_fraction=config.validation_fraction,
            split_salt=config.split_salt,
            document_progress_interval=config.document_progress_interval,
            progress=progress,
        )
        _require_non_empty_splits(counts)
        _write_progress(progress, "second pass: writing local train and validation text")
        _write_processed_splits(
            archive_path=archive_path,
            decisions=decisions,
            train_path=train_path,
            validation_path=validation_path,
            counts=counts,
            document_progress_interval=config.document_progress_interval,
            progress=progress,
        )
        report = _make_report(
            config=config,
            started_at=started_at,
            downloaded=downloaded,
            counts=counts,
        )
        _write_json(local_report_path, report)
        _publish_processed_tree(
            staged_processed_dir=staged_processed_dir,
            processed_dir=config.processed_dir,
        )
        _write_json(config.report_path, report)
    except Exception:
        _release_build_lock(lock_handle)
        raise
    else:
        try:
            shutil.rmtree(temp_root)
        finally:
            _release_build_lock(lock_handle)

    _write_progress(progress, f"wrote local corpus: {config.processed_dir}")
    _write_progress(progress, f"wrote public aggregate report: {config.report_path}")
    return report


def iter_paisa_documents(archive_path: Path) -> Iterator[PaisaDocument]:
    """Yield documents from PAISÀ's line-delimited ``<text id url>`` raw format."""

    current_id = ""
    current_url = ""
    current_lines: list[str] = []
    pending_close_lines: list[str] = []
    with gzip.open(archive_path, "rt", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            opening_match = _OPEN_TEXT_TAG.match(line)
            if not current_id:
                if opening_match:
                    current_id, current_url = _parse_text_tag_attributes(
                        opening_match.group("attributes"),
                        line_number=line_number,
                    )
                    current_lines = []
                elif _is_preamble_line(line):
                    continue
                else:
                    raise ValueError(
                        "unexpected non-preamble content before a PAISÀ <text> tag "
                        f"at line {line_number}"
                    )
                continue

            if pending_close_lines:
                if not line.strip():
                    pending_close_lines.append(line)
                    continue
                if opening_match:
                    yield PaisaDocument(
                        document_id=current_id,
                        url=current_url,
                        raw_text="".join(current_lines),
                    )
                    current_id, current_url = _parse_text_tag_attributes(
                        opening_match.group("attributes"),
                        line_number=line_number,
                    )
                    current_lines = []
                    pending_close_lines = []
                    continue
                current_lines.extend(pending_close_lines)
                pending_close_lines = []

            if opening_match:
                raise ValueError(f"nested PAISÀ <text> tag at line {line_number}")
            if _CLOSE_TEXT_TAG.match(line):
                pending_close_lines = [line]
            else:
                current_lines.append(line)

    if current_id:
        if pending_close_lines:
            yield PaisaDocument(
                document_id=current_id,
                url=current_url,
                raw_text="".join(current_lines),
            )
        else:
            raise ValueError("PAISÀ archive ended before closing its final <text> tag")


def canonicalize_paisa_document_text(text: str) -> str:
    """Normalize transport whitespace without altering spelling or punctuation."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[\t\f\v ]+", " ", normalized)
    normalized = re.sub(r" *\n *", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def split_for_paisa_fingerprint(
    fingerprint: str,
    *,
    validation_fraction: float,
    split_salt: str,
) -> str:
    """Assign a content fingerprint to one deterministic, leak-free split."""

    _validate_validation_fraction(validation_fraction)
    digest = hashlib.sha256(f"{split_salt}:{fingerprint}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big")
    threshold = int(validation_fraction * (1 << 64))
    return "validation" if value < threshold else "train"


def _download_release(
    *,
    release_url: str,
    archive_path: Path,
    download_part_path: Path,
    session: requests.Session | None,
    timeout: int,
    chunk_bytes: int,
    progress_bytes: int,
    max_attempts: int,
    retry_delay_seconds: float,
    progress: ProgressCallback | None,
) -> dict[str, Any]:
    http = session or requests.Session()
    if archive_path.is_file():
        archive_bytes = _file_size(archive_path)
        _write_progress(
            progress,
            f"reusing completed local archive: {_format_bytes(archive_bytes)}",
        )
        return {
            "requested_url": release_url,
            "resolved_url": release_url,
            "content_length_bytes": archive_bytes,
            "downloaded_bytes": archive_bytes,
            "sha256": _digest_existing_file(archive_path).hexdigest(),
            "download_attempts": 0,
            "acquisition_mode": "reused_complete_local_archive",
        }
    _write_progress(progress, f"downloading official release: {release_url}")
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        existing_bytes = _file_size(download_part_path)
        request_kwargs: dict[str, Any] = {"stream": True, "timeout": timeout}
        if existing_bytes:
            request_kwargs["headers"] = {"Range": f"bytes={existing_bytes}-"}
            _write_progress(
                progress,
                f"resuming partial download at {_format_bytes(existing_bytes)} "
                f"attempt={attempt}/{max_attempts}",
            )
        try:
            response = http.get(release_url, **request_kwargs)
            response.raise_for_status()
            content_type = str(response.headers.get("Content-Type", "")).lower()
            if "text/html" in content_type:
                raise ValueError(
                    "PAISÀ release URL returned HTML rather than the audited gzip artifact: "
                    f"{getattr(response, 'url', release_url)}"
                )
            response_status = int(getattr(response, "status_code", 200))
            if existing_bytes and response_status == 206:
                write_mode = "ab"
                expected_bytes = _expected_total_bytes(response.headers, existing_bytes)
            elif existing_bytes and response_status == 200:
                _write_progress(
                    progress,
                    "release server ignored Range; restarting this download attempt from zero",
                )
                existing_bytes = 0
                write_mode = "wb"
                expected_bytes = _parse_positive_int(response.headers.get("Content-Length"))
            elif existing_bytes:
                raise _RetryableDownloadError(
                    f"unexpected HTTP status for resumed PAISÀ download: {response_status}"
                )
            else:
                write_mode = "wb"
                expected_bytes = _parse_positive_int(response.headers.get("Content-Length"))

            digest = _digest_existing_file(download_part_path) if existing_bytes else hashlib.sha256()
            downloaded_bytes = existing_bytes
            next_progress_bytes = ((downloaded_bytes // progress_bytes) + 1) * progress_bytes
            with download_part_path.open(write_mode) as handle:
                for chunk in response.iter_content(chunk_size=chunk_bytes):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    digest.update(chunk)
                    downloaded_bytes += len(chunk)
                    if downloaded_bytes >= next_progress_bytes:
                        _write_progress(
                            progress,
                            _download_progress_message(downloaded_bytes, expected_bytes),
                        )
                        next_progress_bytes += progress_bytes

            if expected_bytes is not None and downloaded_bytes != expected_bytes:
                raise _RetryableDownloadError(
                    "PAISÀ release byte count does not match the expected total: "
                    f"expected={expected_bytes} actual={downloaded_bytes}"
                )
            download_part_path.replace(archive_path)
            _write_progress(progress, _download_progress_message(downloaded_bytes, expected_bytes))
            return {
                "requested_url": release_url,
                "resolved_url": str(getattr(response, "url", release_url)),
                "content_length_bytes": expected_bytes,
                "downloaded_bytes": downloaded_bytes,
                "sha256": digest.hexdigest(),
                "download_attempts": attempt,
                "acquisition_mode": "downloaded_from_official_release",
            }
        except (requests.RequestException, OSError, _RetryableDownloadError) as exc:
            last_error = exc
            _write_progress(
                progress,
                f"download attempt {attempt}/{max_attempts} interrupted: {exc}",
            )
            if attempt < max_attempts:
                time.sleep(retry_delay_seconds)
                continue
            break

    assert last_error is not None
    raise RuntimeError(
        "PAISÀ release download did not complete after "
        f"{max_attempts} attempts; partial file retained at {download_part_path}"
    ) from last_error


def _inventory_documents(
    *,
    archive_path: Path,
    inventory_path: Path,
    validation_fraction: float,
    split_salt: str,
    document_progress_interval: int,
    progress: ProgressCallback | None,
) -> tuple[dict[str, tuple[str, str]], PaisaBuildCounts]:
    decisions: dict[str, tuple[str, str]] = {}
    seen_fingerprints: dict[str, str] = {}
    seen_document_ids: set[str] = set()
    counts = PaisaBuildCounts()

    with inventory_path.open("w", encoding="utf-8") as inventory_handle:
        for document in iter_paisa_documents(archive_path):
            counts.parsed_documents += 1
            _require_unique_document_id(document.document_id, seen_document_ids)
            cleaned_text = canonicalize_paisa_document_text(document.raw_text)
            if not cleaned_text:
                counts.empty_documents += 1
                _write_inventory_record(
                    inventory_handle,
                    document=document,
                    status="excluded_empty",
                )
            elif PAISA_DOCUMENT_SEPARATOR in cleaned_text:
                raise ValueError(
                    "PAISÀ document contains the reserved document separator: "
                    f"{document.document_id}"
                )
            else:
                fingerprint = _text_fingerprint(cleaned_text)
                original_document_id = seen_fingerprints.get(fingerprint)
                if original_document_id is not None:
                    counts.exact_duplicate_documents += 1
                    _write_inventory_record(
                        inventory_handle,
                        document=document,
                        status="excluded_exact_duplicate",
                        text_sha256=fingerprint,
                        duplicate_of_document_id=original_document_id,
                    )
                else:
                    split = split_for_paisa_fingerprint(
                        fingerprint,
                        validation_fraction=validation_fraction,
                        split_salt=split_salt,
                    )
                    seen_fingerprints[fingerprint] = document.document_id
                    decisions[document.document_id] = (fingerprint, split)
                    character_count = len(cleaned_text)
                    word_count = _count_whitespace_words(cleaned_text)
                    counts.retained_documents += 1
                    counts.retained_characters += character_count
                    counts.retained_words += word_count
                    _record_split_counts(
                        counts,
                        split=split,
                        character_count=character_count,
                        word_count=word_count,
                    )
                    _write_inventory_record(
                        inventory_handle,
                        document=document,
                        status="retained",
                        split=split,
                        text_sha256=fingerprint,
                        character_count=character_count,
                        word_count=word_count,
                    )
            _write_document_progress(
                progress,
                phase="inventory",
                completed=counts.parsed_documents,
                interval=document_progress_interval,
            )
    return decisions, counts


def _write_processed_splits(
    *,
    archive_path: Path,
    decisions: dict[str, tuple[str, str]],
    train_path: Path,
    validation_path: Path,
    counts: PaisaBuildCounts,
    document_progress_interval: int,
    progress: ProgressCallback | None,
) -> None:
    written_documents = 0
    with (
        train_path.open("w", encoding="utf-8") as train_handle,
        validation_path.open("w", encoding="utf-8") as validation_handle,
    ):
        for document in iter_paisa_documents(archive_path):
            decision = decisions.get(document.document_id)
            if decision is None:
                continue
            fingerprint, split = decision
            cleaned_text = canonicalize_paisa_document_text(document.raw_text)
            if _text_fingerprint(cleaned_text) != fingerprint:
                raise ValueError(
                    "PAISÀ document content changed between inventory and output passes: "
                    f"{document.document_id}"
                )
            handle = train_handle if split == "train" else validation_handle
            handle.write(cleaned_text)
            handle.write(f"\n{PAISA_DOCUMENT_SEPARATOR}\n")
            written_documents += 1
            _write_document_progress(
                progress,
                phase="writing",
                completed=written_documents,
                interval=document_progress_interval,
            )

    if written_documents != counts.retained_documents:
        raise ValueError(
            "PAISÀ output pass retained a different number of documents than inventory: "
            f"expected={counts.retained_documents} actual={written_documents}"
        )


def _make_report(
    *,
    config: PaisaBuildConfig,
    started_at: str,
    downloaded: dict[str, Any],
    counts: PaisaBuildCounts,
) -> dict[str, Any]:
    total_output_characters = (
        counts.retained_characters
        + counts.retained_documents * (len(PAISA_DOCUMENT_SEPARATOR) + 2)
    )
    return {
        "corpus_version": config.corpus_version,
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "source": {
            "name": "PAISÀ Corpus of Italian Web Texts",
            "release": downloaded,
            "corpus_license": "CC BY-NC-SA",
            "source_license_families": ["CC BY-SA", "CC BY-NC-SA"],
            "document_provenance_fields": ["id", "url"],
        },
        "cleaning_policy": (
            "normalize line endings and transport whitespace only; preserve spelling, "
            "punctuation, and document-internal paragraph boundaries"
        ),
        "deduplication_policy": "exact SHA-256 fingerprint after whitespace canonicalization",
        "split_policy": {
            "method": "SHA-256 fingerprint assignment",
            "validation_fraction_target": config.validation_fraction,
            "split_salt": config.split_salt,
            "exact_duplicate_split_leakage": "prevented by deduplication before splitting",
        },
        "document_counts": {
            "parsed": counts.parsed_documents,
            "retained": counts.retained_documents,
            "excluded_empty": counts.empty_documents,
            "excluded_exact_duplicate": counts.exact_duplicate_documents,
            "train": counts.train_documents,
            "validation": counts.validation_documents,
        },
        "text_counts": {
            "retained_characters": counts.retained_characters,
            "retained_whitespace_words": counts.retained_words,
            "train_characters": counts.train_characters,
            "validation_characters": counts.validation_characters,
            "train_whitespace_words": counts.train_words,
            "validation_whitespace_words": counts.validation_words,
            "output_characters_with_document_separators": total_output_characters,
        },
        "document_separator": PAISA_DOCUMENT_SEPARATOR,
        "local_artifacts": {
            "train_text_path": _portable_path(config.processed_dir / "train.txt"),
            "validation_text_path": _portable_path(config.processed_dir / "validation.txt"),
            "document_attribution_inventory_path": _portable_path(
                config.processed_dir / "document_attribution.jsonl"
            ),
        },
        "public_repository_policy": (
            "This report contains no document text or document URLs. PAISÀ text, "
            "attribution inventory, derived token files, and derived checkpoints stay local."
        ),
        "temporary_raw_and_interim_deleted_after_success": True,
    }


def _parse_text_tag_attributes(attributes: str, *, line_number: int) -> tuple[str, str]:
    parsed: dict[str, str] = {}
    for match in _ATTRIBUTE.finditer(attributes):
        parsed[match.group("name").lower()] = match.group("value")
    document_id = parsed.get("id", "").strip()
    url = parsed.get("url", "").strip()
    if not document_id or not url:
        raise ValueError(f"PAISÀ <text> tag lacks id or url at line {line_number}")
    return document_id, url


def _is_preamble_line(line: str) -> bool:
    stripped = line.strip()
    return not stripped or stripped.startswith("#")


def _require_unique_document_id(document_id: str, seen_document_ids: set[str]) -> None:
    if document_id in seen_document_ids:
        raise ValueError(f"PAISÀ archive has duplicate document id: {document_id}")
    seen_document_ids.add(document_id)


def _write_inventory_record(
    handle,
    *,
    document: PaisaDocument,
    status: str,
    split: str = "",
    text_sha256: str = "",
    duplicate_of_document_id: str = "",
    character_count: int = 0,
    word_count: int = 0,
) -> None:
    record = {
        "document_id": document.document_id,
        "url": document.url,
        "status": status,
        "split": split,
        "text_sha256": text_sha256,
        "duplicate_of_document_id": duplicate_of_document_id,
        "character_count": character_count,
        "whitespace_word_count": word_count,
    }
    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _record_split_counts(
    counts: PaisaBuildCounts,
    *,
    split: str,
    character_count: int,
    word_count: int,
) -> None:
    if split == "train":
        counts.train_documents += 1
        counts.train_characters += character_count
        counts.train_words += word_count
    elif split == "validation":
        counts.validation_documents += 1
        counts.validation_characters += character_count
        counts.validation_words += word_count
    else:
        raise ValueError(f"unexpected PAISÀ split: {split}")


def _require_non_empty_splits(counts: PaisaBuildCounts) -> None:
    if counts.train_documents == 0 or counts.validation_documents == 0:
        raise ValueError(
            "PAISÀ split must retain at least one train and one validation document; "
            f"train={counts.train_documents} validation={counts.validation_documents}"
        )


def _text_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _count_whitespace_words(text: str) -> int:
    return len(text.split())


def _write_document_progress(
    progress: ProgressCallback | None,
    *,
    phase: str,
    completed: int,
    interval: int,
) -> None:
    if completed % interval == 0:
        _write_progress(progress, f"{phase} documents={completed}")


def _download_progress_message(downloaded_bytes: int, content_length: int | None) -> str:
    message = f"downloaded={_format_bytes(downloaded_bytes)}"
    if content_length is not None:
        message += f"/{_format_bytes(content_length)} ({downloaded_bytes / content_length:.1%})"
    return message


def _format_bytes(value: int) -> str:
    return f"{value / (1024 * 1024):.1f} MiB"


def _parse_positive_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(str(value))
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _expected_total_bytes(headers: object, existing_bytes: int) -> int | None:
    header_map = headers if isinstance(headers, dict) else dict(headers)
    content_range = str(header_map.get("Content-Range", ""))
    match = re.fullmatch(r"bytes\s+\d+-\d+/(\d+)", content_range)
    if match is not None:
        return int(match.group(1))
    remaining_bytes = _parse_positive_int(header_map.get("Content-Length"))
    return existing_bytes + remaining_bytes if remaining_bytes is not None else None


def _file_size(path: Path) -> int:
    return path.stat().st_size if path.is_file() else 0


def _digest_existing_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest


class _RetryableDownloadError(RuntimeError):
    """A transfer did not finish but can continue from its retained partial file."""


def _validate_config(config: PaisaBuildConfig) -> None:
    _validate_validation_fraction(config.validation_fraction)
    if not config.split_salt:
        raise ValueError("split_salt must not be empty")
    if config.request_timeout_seconds <= 0:
        raise ValueError("request_timeout_seconds must be positive")
    if config.download_chunk_bytes <= 0:
        raise ValueError("download_chunk_bytes must be positive")
    if config.download_progress_bytes <= 0:
        raise ValueError("download_progress_bytes must be positive")
    if config.document_progress_interval <= 0:
        raise ValueError("document_progress_interval must be positive")
    if config.max_download_attempts <= 0:
        raise ValueError("max_download_attempts must be positive")
    if config.download_retry_delay_seconds < 0:
        raise ValueError("download_retry_delay_seconds cannot be negative")


def _validate_validation_fraction(validation_fraction: float) -> None:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be greater than 0 and less than 1")


def _prepare_temp_tree(
    temp_root: Path,
    raw_dir: Path,
    interim_dir: Path,
    staged_processed_dir: Path,
    *,
    reusable_archive_path: Path,
    resumable_download_path: Path,
) -> None:
    if temp_root.exists():
        if resumable_download_path.is_file() or _is_valid_gzip_archive(reusable_archive_path):
            shutil.rmtree(interim_dir, ignore_errors=True)
            shutil.rmtree(staged_processed_dir, ignore_errors=True)
            interim_dir.mkdir(parents=True)
            staged_processed_dir.mkdir(parents=True)
            return
        shutil.rmtree(temp_root)
    raw_dir.mkdir(parents=True)
    interim_dir.mkdir(parents=True)
    staged_processed_dir.mkdir(parents=True)


def _is_valid_gzip_archive(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with gzip.open(path, "rb") as handle:
            for _ in iter(lambda: handle.read(1_048_576), b""):
                pass
    except (gzip.BadGzipFile, EOFError, OSError):
        return False
    return True


def _publish_processed_tree(*, staged_processed_dir: Path, processed_dir: Path) -> None:
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    processed_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staged_processed_dir), str(processed_dir))


def _acquire_build_lock(temp_root: Path):
    lock_path = temp_root.parent / f".{temp_root.name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(
            "another PAISÀ build is already using temp_dir: " f"{temp_root}"
        ) from exc
    return handle


def _release_build_lock(handle) -> None:
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()


def _validate_deletable_directory(path: Path, *, label: str) -> None:
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    forbidden = {Path("/").resolve(), cwd, cwd.parent}
    if resolved in forbidden:
        raise ValueError(f"{label} is not safe to delete: {path}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _write_progress(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)
