#!/usr/bin/env python3
"""Run checkpoint 4C extraction, overlap probing, and rendered validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import monotonic, sleep

import requests

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.gutenberg_fulltext_probe import measure_word_shingle_containment
from sonnet_corpus.italian_wikisource import USER_AGENT, extract_wikisource_prose_text
from sonnet_corpus.wikisource_page_extraction import (
    DUMP_FILENAME,
    DUMP_SHA1,
    WikisourcePageExtractionConfig,
    apply_rendered_validation,
    run_wikisource_page_extraction,
    select_rendered_validation_pages,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dump",
        type=Path,
        default=ROOT / "data/local/wikisource/archive_inventory_v1" / DUMP_FILENAME,
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data/local/wikisource/page_extraction_v1",
    )
    parser.add_argument(
        "--extraction",
        type=Path,
        default=ROOT / "data/metadata/italian_wikisource_page_extraction_v1.csv",
    )
    parser.add_argument(
        "--boundaries",
        type=Path,
        default=ROOT / "data/metadata/italian_wikisource_page_boundaries_v1.csv",
    )
    parser.add_argument(
        "--review",
        type=Path,
        default=ROOT / "data/metadata/italian_wikisource_extraction_review_v1.csv",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        default=ROOT / "reports/italian_wikisource_page_extraction_v1.json",
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=ROOT / "reports/italian_wikisource_page_extraction_v1.md",
    )
    parser.add_argument("--progress-interval", type=int, default=25_000)
    parser.add_argument("--validation-root-sample-size", type=int, default=30)
    parser.add_argument("--request-delay", type=float, default=1.0)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--api-retries", type=int, default=5)
    parser.add_argument(
        "--skip-rendered-validation",
        action="store_true",
        help="Run only the local audit; intended for deterministic/offline tests.",
    )
    parser.add_argument(
        "--rendered-validation-only",
        action="store_true",
        help="Resume the bounded API validation from existing local/public audit artifacts.",
    )
    return parser.parse_args()


def _config(args: argparse.Namespace) -> WikisourcePageExtractionConfig:
    return WikisourcePageExtractionConfig(
        repo_root=ROOT,
        dump_path=args.dump,
        resolution_path=ROOT / "data/metadata/italian_wikisource_candidate_resolution_v1.csv",
        hierarchy_path=ROOT / "data/metadata/italian_wikisource_page_hierarchy_v1.csv",
        extraction_path=args.extraction,
        boundaries_path=args.boundaries,
        review_path=args.review,
        json_report_path=args.json_report,
        markdown_report_path=args.markdown_report,
        local_cache_dir=args.cache_dir,
        bibit_record_manifest_path=ROOT / "data/processed/bibit_resolved_v1/records_manifest.csv",
        broader_sources_manifest_path=ROOT / "data/metadata/broader_prose_sources_manifest.csv",
        sonnet_manifest_path=ROOT / "data/metadata/sonnets_expanded_v6_manifest.csv",
        gutenberg_previous_probe_path=ROOT / "data/metadata/project_gutenberg_fulltext_probe_v1.csv",
        gutenberg_previous_cache_dir=ROOT / "data/local/gutenberg/fulltext_gate_v1",
        gutenberg_pass_1b_probe_path=ROOT / "data/metadata/project_gutenberg_fulltext_probe_pass_1b_v1.csv",
        gutenberg_pass_1b_cache_dir=ROOT / "data/local/gutenberg/metadata_review_v1",
        gutenberg_resolved_manifest_path=ROOT / "data/processed/project_gutenberg_resolved_v1/records_manifest.csv",
        expected_dump_sha1=DUMP_SHA1,
        progress_interval=args.progress_interval,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _fetch_rendered_html(
    session: requests.Session,
    *,
    revision_id: int,
    timeout: float,
    retries: int,
) -> str:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(
                "https://it.wikisource.org/w/api.php",
                params={
                    "action": "parse",
                    "oldid": revision_id,
                    "prop": "text",
                    "format": "json",
                    "formatversion": "2",
                },
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
            if "error" in payload:
                raise ValueError(payload["error"])
            return str(payload["parse"]["text"])
        except (requests.RequestException, ValueError, KeyError) as error:
            last_error = error
            if attempt < retries:
                sleep(min(30.0, 2.0 ** (attempt - 1)))
    raise RuntimeError(f"rendered revision fetch failed: {last_error}")


def _run_rendered_validation(args: argparse.Namespace, config: WikisourcePageExtractionConfig) -> None:
    boundaries = _read_csv(config.boundaries_path)
    extraction = _read_csv(config.extraction_path)
    sample = select_rendered_validation_pages(
        boundaries,
        extraction,
        root_sample_size=args.validation_root_sample_size,
    )
    rendered_cache = config.local_cache_dir / "rendered_validation"
    rendered_cache.mkdir(parents=True, exist_ok=True)
    page_text_dir = config.local_cache_dir / "page_texts"
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    results: dict[int, tuple[str, float | None]] = {}
    started = monotonic()
    for index, row in enumerate(sample, start=1):
        page_id = int(row["page_id"])
        revision_id = int(row["dump_revision_id"])
        cache_path = rendered_cache / f"page_{page_id}_rev_{revision_id}.json"
        try:
            if cache_path.is_file():
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                rendered_text = payload["cleaned_text"]
                cache_status = "hit"
            else:
                if index > 1 and args.request_delay:
                    sleep(args.request_delay)
                raw_html = _fetch_rendered_html(
                    session,
                    revision_id=revision_id,
                    timeout=args.request_timeout,
                    retries=args.api_retries,
                )
                rendered_text = extract_wikisource_prose_text(raw_html)
                payload = {
                    "page_id": page_id,
                    "revision_id": revision_id,
                    "title": row["page_title"],
                    "cleaned_text": rendered_text,
                }
                cache_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                cache_status = "downloaded"
            local_text = (page_text_dir / f"{page_id}.txt").read_text(encoding="utf-8")
            metric = measure_word_shingle_containment(local_text, rendered_text)
            containment = metric["containment"]
            status = "pass" if containment >= 0.8 else "hold_low_containment"
            results[page_id] = (status, containment)
        except Exception as error:
            results[page_id] = (f"hold_error:{type(error).__name__}", None)
            cache_status = f"error:{type(error).__name__}:{str(error)[:120]}"
        elapsed = monotonic() - started
        rate = index / elapsed if elapsed else 0.0
        eta = (len(sample) - index) / rate if rate else 0.0
        print(
            "wikisource-page-extraction | rendered-validation "
            f"completed={index:,}/{len(sample):,} percent={index / max(1, len(sample)):.1%} "
            f"cache={cache_status} elapsed={elapsed:.1f}s eta={eta:.1f}s",
            flush=True,
        )
    apply_rendered_validation(
        extraction_path=config.extraction_path,
        boundaries_path=config.boundaries_path,
        review_path=config.review_path,
        json_report_path=config.json_report_path,
        markdown_report_path=config.markdown_report_path,
        results=results,
    )


def main() -> None:
    args = parse_args()
    config = _config(args)
    print(
        "wikisource-page-extraction | start device=cpu "
        "eligible_roots=4641 dump_sha1_pinned=true activation=false "
        f"progress_interval={args.progress_interval} "
        "estimated_runtime=2h-8h_first_run_or_10m-60m_cached",
        flush=True,
    )

    def progress(message: str) -> None:
        print(f"wikisource-page-extraction | {message}", flush=True)

    if args.rendered_validation_only:
        if not config.extraction_path.is_file() or not config.boundaries_path.is_file():
            raise FileNotFoundError("rendered-validation-only requires completed 4C audit artifacts")
        report = json.loads(config.json_report_path.read_text(encoding="utf-8"))
    else:
        report = run_wikisource_page_extraction(config, progress=progress)
    if not args.skip_rendered_validation:
        _run_rendered_validation(args, config)
        report = json.loads(config.json_report_path.read_text(encoding="utf-8"))
    print(
        "wikisource-page-extraction | complete "
        f"roots={report['eligible_root_count']:,} "
        f"characters={report['extracted_character_count']:,} "
        f"reviews={report['review_row_count']:,} activated=0",
        flush=True,
    )
    print(f"wikisource-page-extraction | wrote: {config.extraction_path}", flush=True)
    print(f"wikisource-page-extraction | wrote: {config.json_report_path}", flush=True)


if __name__ == "__main__":
    main()
