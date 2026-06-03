"""Corpus build orchestration."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .cleaning import clean_poem_text, count_poem_lines
from .manifest import ManifestRow, validate_processed_files, write_manifest
from .sources import CandidatePoem, SOURCES, discover_candidates, discover_cino_from_titles, source_keys
from .splits import assign_split
from .wikisource import WikisourceClient, extract_poem_text, iter_index_links


def build_corpus(
    *,
    repo_root: Path,
    sources: str = "all",
    dataset: str = "expanded_with_petrarch",
    force: bool = False,
    keep_temp: bool = False,
    seed: int = 1337,
    request_delay: float = 1.0,
    verbose: bool = True,
) -> dict[str, object]:
    if dataset not in {"core_pre_petrarch", "expanded_with_petrarch"}:
        raise ValueError(f"unknown dataset: {dataset}")

    data_dir = repo_root / "data"
    raw_dir = data_dir / "raw"
    interim_dir = data_dir / "interim"
    processed_dir = data_dir / "processed" / "poems"
    metadata_dir = data_dir / "metadata"

    if force:
        shutil.rmtree(processed_dir, ignore_errors=True)
        shutil.rmtree(metadata_dir, ignore_errors=True)
        shutil.rmtree(raw_dir, ignore_errors=True)
        shutil.rmtree(interim_dir, ignore_errors=True)

    processed_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    client = WikisourceClient(raw_dir=raw_dir, request_delay=request_delay)
    selected_keys = source_keys(sources)

    rows: list[ManifestRow] = []
    source_counts: dict[str, int] = {}
    started_at = datetime.now(UTC).replace(microsecond=0).isoformat()

    try:
        for key in selected_keys:
            spec = SOURCES[key]
            if key == "cino":
                _log(verbose, f"[{key}] fetching category members")
                titles = client.category_members("Categoria:Testi di Cino da Pistoia")
                candidates = discover_cino_from_titles(spec, titles)
            else:
                _log(verbose, f"[{key}] fetching index")
                index_page = client.fetch(spec.index_url)
                candidates = discover_candidates(spec, index_page.html)
            candidates = _expand_cycle_candidates(client, candidates)
            source_counts[key] = len(candidates)
            _log(verbose, f"[{key}] discovered {len(candidates)} candidate pages")
            for index, candidate in enumerate(candidates, start=1):
                if index == 1 or index == len(candidates) or index % 25 == 0:
                    _log(verbose, f"[{key}] processing {index}/{len(candidates)}")
                row = _build_poem(client, candidate, processed_dir, interim_dir, seed)
                if dataset == "core_pre_petrarch":
                    row.include_in_training = (
                        row.include_in_training and row.include_in_core_pre_petrarch
                    )
                rows.append(row)

        manifest_path = metadata_dir / "poems_manifest.csv"
        write_manifest(rows, manifest_path)
        validate_processed_files(rows, repo_root)

        report = {
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "dataset": dataset,
            "sources": selected_keys,
            "seed": seed,
            "candidate_counts": source_counts,
            "manifest_rows": len(rows),
            "included_rows": sum(row.include_in_training for row in rows),
        }
        (metadata_dir / "build_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _write_attribution_summary(rows, metadata_dir / "attribution_summary.md")

        if not keep_temp:
            shutil.rmtree(raw_dir, ignore_errors=True)
            shutil.rmtree(interim_dir, ignore_errors=True)
            _log(verbose, "deleted data/raw and data/interim after successful build")

        return report
    except Exception:
        if not keep_temp:
            # Preserve temp artifacts on failure to make debugging possible.
            pass
        raise


def _expand_cycle_candidates(
    client: WikisourceClient, candidates: list[CandidatePoem]
) -> list[CandidatePoem]:
    expanded: list[CandidatePoem] = []
    for candidate in candidates:
        if candidate.audit_notes != "cycle_page_expand":
            expanded.append(candidate)
            continue

        cycle_page = client.fetch(candidate.source_url)
        for link in iter_index_links(cycle_page.html, candidate.source_url):
            if link.url == candidate.source_url:
                continue
            expanded.append(
                CandidatePoem(
                    **{
                        **asdict(candidate),
                        "poem_id": "",
                        "title_or_first_line": link.title,
                        "source_url": link.url,
                        "source_subcollection": candidate.source_subcollection,
                        "audit_notes": "expanded from sonnet-cycle page",
                    }
                )
            )
            expanded[-1] = CandidatePoem(
                **{
                    **asdict(expanded[-1]),
                    "poem_id": f"folgore_{len(expanded):03d}",
                }
            )
    return expanded


def _build_poem(
    client: WikisourceClient,
    candidate: CandidatePoem,
    processed_dir: Path,
    interim_dir: Path,
    seed: int,
) -> ManifestRow:
    page = client.fetch(candidate.source_url)
    extracted = extract_poem_text(page.html)
    interim_dir.mkdir(parents=True, exist_ok=True)
    (interim_dir / f"{candidate.poem_id}.txt").write_text(extracted + "\n", encoding="utf-8")

    cleaned = clean_poem_text(extracted)
    line_count_clean = count_poem_lines(cleaned)
    include = line_count_clean == 14

    clean_path = processed_dir / f"{candidate.poem_id}.txt"
    if include:
        clean_path.write_text(cleaned, encoding="utf-8")
        clean_text_path = str(clean_path.relative_to(processed_dir.parents[2]))
    else:
        clean_text_path = ""

    split_core = (
        assign_split(candidate.poem_id, seed=seed)
        if include and candidate.include_in_core_pre_petrarch
        else "excluded"
    )
    split_expanded = (
        assign_split(candidate.poem_id, seed=seed)
        if include and candidate.include_in_expanded_with_petrarch
        else "excluded"
    )

    return ManifestRow(
        poem_id=candidate.poem_id,
        title_or_first_line=candidate.title_or_first_line,
        author=candidate.author,
        displayed_author=candidate.displayed_author,
        source_archive=candidate.source_archive,
        source_collection=candidate.source_collection,
        source_subcollection=candidate.source_subcollection,
        source_url=candidate.source_url,
        source_revision_id=page.revision_id,
        source_revision_timestamp=page.revision_timestamp,
        downloaded_at_utc=page.downloaded_at_utc,
        source_edition=candidate.source_edition,
        license_notes=candidate.license_notes,
        period=candidate.period,
        form="sonnet",
        form_evidence=candidate.form_evidence,
        count_method=candidate.count_method if include else "manual_exclusion",
        attribution_status=candidate.attribution_status,
        line_count_raw=count_poem_lines(extracted),
        line_count_clean=line_count_clean,
        raw_text_path="",
        clean_text_path=clean_text_path,
        include_in_core_pre_petrarch=candidate.include_in_core_pre_petrarch,
        include_in_expanded_with_petrarch=candidate.include_in_expanded_with_petrarch,
        include_in_training=include,
        split_core_pre_petrarch=split_core,
        split_expanded_with_petrarch=split_expanded,
        editorial_brackets_removed=True,
        line_markers_removed=True,
        cleaning_notes=(
            "Removed editorial square brackets around letter expansions; "
            "removed displayed line markers; preserved line breaks."
        ),
        audit_notes=(
            candidate.audit_notes
            if include
            else f"excluded after cleaning because line_count_clean={line_count_clean}"
        ),
    )


def _write_attribution_summary(rows: list[ManifestRow], path: Path) -> None:
    source_urls = sorted({row.source_url for row in rows})
    lines = [
        "# Corpus Attribution Summary",
        "",
        "Processed poem files in `data/processed/` are derived from Italian Wikisource pages.",
        "Each manifest row records source URL, edition notes where available, and license notes.",
        "",
        "## Source URLs",
        "",
    ]
    lines.extend(f"- {url}" for url in source_urls)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(message, flush=True)
