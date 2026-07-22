"""Run the approved remaining sonnet-source audits and summarize the evidence."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from collections.abc import Callable

from .sonnet_wikisource_probe import probe_sonnet_wikisource_source


REMAINING_SOURCE_AUDIT_IDS = (
    "ws_varchi_infermita",
    "ws_aretino_sonetti_lussuriosi_1792",
)


def run_remaining_sonnet_source_audits(
    *,
    source_manifest_path: Path,
    active_poems_manifest_path: Path,
    repo_root: Path,
    reports_directory: Path,
    summary_path: Path,
    request_delay: float = 6.0,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Audit each approved source and always write a consolidated summary.

    A failed source does not hide results from the others. Individual reports
    remain local because no source is activated by this audit phase.
    """

    started_at = utc_now()
    results: list[dict[str, object]] = []
    reports_directory.mkdir(parents=True, exist_ok=True)
    for index, source_id in enumerate(REMAINING_SOURCE_AUDIT_IDS, start=1):
        report_path = reports_directory / f"{source_id}_probe.json"
        started = monotonic()
        _write_progress(
            progress,
            f"starting source {index}/{len(REMAINING_SOURCE_AUDIT_IDS)}: {source_id}",
        )
        try:
            report = probe_sonnet_wikisource_source(
                source_manifest_path=source_manifest_path,
                active_poems_manifest_path=active_poems_manifest_path,
                repo_root=repo_root,
                source_id=source_id,
                report_path=report_path,
                request_delay=request_delay,
                progress=lambda message: _write_progress(progress, f"{source_id} | {message}"),
            )
        except Exception as error:  # Preserve independent audit outcomes.
            outcome = {
                "source_id": source_id,
                "status": "error",
                "error": str(error),
                "elapsed_seconds": round(monotonic() - started, 1),
            }
            _write_progress(progress, f"source failed: {source_id}: {error}")
        else:
            outcome = {
                "source_id": source_id,
                "status": "ok",
                "activation_status": report["activation_status"],
                "page_count": report["page_count"],
                "candidate_status_counts": report["candidate_status_counts"],
                "report_path": portable_path(report_path, repo_root),
                "elapsed_seconds": round(monotonic() - started, 1),
            }
            _write_progress(
                progress,
                f"source complete: {source_id} pages={report['page_count']} "
                f"elapsed={outcome['elapsed_seconds']:.1f}s",
            )
        results.append(outcome)

    summary = {
        "started_at_utc": started_at,
        "source_ids": list(REMAINING_SOURCE_AUDIT_IDS),
        "request_delay_seconds": request_delay,
        "results": results,
        "finished_at_utc": utc_now(),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_progress(progress, f"wrote consolidated audit summary: {summary_path}")
    return summary


def portable_path(path: Path, repo_root: Path) -> str:
    """Return a repository-relative report path where possible."""

    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def utc_now() -> str:
    """Return a second-precision UTC timestamp."""

    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _write_progress(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
