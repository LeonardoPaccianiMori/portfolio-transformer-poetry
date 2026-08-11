import csv
import hashlib
import json
from pathlib import Path

import pytest

import sonnet_corpus.gutenberg_resolved_build as build_module
from sonnet_corpus.gutenberg import strip_gutenberg_boilerplate
from sonnet_corpus.gutenberg_resolved_build import (
    ATTRIBUTION_MANIFEST_FIELDS,
    RECORD_MANIFEST_FIELDS,
    SEGMENT_MANIFEST_FIELDS,
    SONNET_MANIFEST_FIELDS,
    GutenbergResolvedBuildConfig,
    build_gutenberg_resolved_corpus,
)


def _write_csv(path: Path, fields, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _gutenberg(text: str) -> str:
    return (
        "Header\n*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
        + text
        + "\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\nFooter\n"
    )


def _sonnet(label: str) -> str:
    return "\n".join(f"{label} verso poetico numero {index}" for index in range(1, 15)) + "\n"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _source_row(
    ebook_id: str,
    text: str,
    *,
    title: str,
    role: str,
    decision: str,
    source_pool: str = "initial_eligible_pool",
    language_route: str = "standard_italian",
    included_characters: int | None = None,
) -> dict[str, str]:
    return {
        "ebook_id": ebook_id,
        "title": title,
        "authors": "Source Record Author",
        "source_pool": source_pool,
        "source_archive": "Project Gutenberg",
        "source_url": f"https://www.gutenberg.org/ebooks/{ebook_id}",
        "period_bucket": (
            "origins_through_1800"
            if role != "nineteenth_century_bridge"
            else "nineteenth_century"
        ),
        "input_role": role,
        "final_role": role,
        "probe_decision": "quality_pass",
        "source_decision": decision,
        "extraction_policy": "test_policy",
        "canonical_reference_ids": "",
        "language_route": language_route,
        "cache_path": f"data/local/gutenberg/test/pg{ebook_id}.txt",
        "cache_sha256": "",
        "cleaned_sha256": _sha(text),
        "cleaned_character_count": str(len(text)),
        "included_record_character_count": str(
            len(text) if included_characters is None else included_characters
        ),
        "excluded_character_count": str(
            0 if included_characters is None else len(text) - included_characters
        ),
        "sonnet_candidate_count": "0",
        "eligible_standard_sonnet_count": "0",
        "unresolved_sonnet_review_count": "0",
        "residual_heldout_overlap_ids": "",
    }


def _segments(
    ebook_id: str,
    text: str,
    parts: list[tuple[int, int, str, str, str]],
) -> list[dict[str, str]]:
    rows = []
    for index, (start, end, decision, role, reason) in enumerate(parts, start=1):
        payload = text[start:end]
        rows.append(
            {
                "segment_id": f"pg{ebook_id}:seg{index:04d}",
                "ebook_id": ebook_id,
                "source_cleaned_sha256": _sha(text),
                "character_start": str(start),
                "character_end": str(end),
                "character_count": str(end - start),
                "segment_sha256": _sha(payload),
                "segment_decision": decision,
                "final_role": role,
                "reason": reason,
                "reference_ids": "",
                "start_anchor": " ".join(payload[:120].split()),
                "end_anchor": " ".join(payload[-120:].split()),
            }
        )
    return rows


def _candidate(
    ebook_id: str,
    title: str,
    text: str,
    start: int,
    end: int,
    decision: str,
    resolution: str,
) -> dict[str, str]:
    raw = text[start:end]
    cleaned = "\n".join(line.strip() for line in raw.splitlines() if line.strip()) + "\n"
    lines = cleaned.strip().splitlines()
    return {
        "candidate_id": f"pg{ebook_id}:char{start}-{end}",
        "ebook_id": ebook_id,
        "title": title,
        "authors": "Source Record Author",
        "source_kind": "structural_14_line",
        "stanza_pattern": "14",
        "line_count": "14",
        "character_start": str(start),
        "character_end": str(end),
        "source_text_sha256": _sha(raw),
        "cleaned_text_sha256": _sha(cleaned),
        "first_line": lines[0],
        "last_line": lines[-1],
        "exact_reference_ids": "",
        "near_reference_ids": "",
        "heldout_reference_ids": "",
        "duplicate_gutenberg_candidate_ids": "",
        "manual_review_resolution": resolution,
        "manual_review_rationale": "Pinned test review evidence.",
        "candidate_decision": decision,
    }


def _refresh_audit_report(config: GutenbergResolvedBuildConfig) -> None:
    report = {
        "audit_version": "project_gutenberg_extraction_audit_v1",
        "unresolved_sonnet_review_count": 0,
        "outputs": {
            "source_csv_sha256": hashlib.sha256(
                config.source_decisions_path.read_bytes()
            ).hexdigest(),
            "segment_csv_sha256": hashlib.sha256(
                config.segment_decisions_path.read_bytes()
            ).hexdigest(),
            "sonnet_csv_sha256": hashlib.sha256(
                config.sonnet_decisions_path.read_bytes()
            ).hexdigest(),
            "review_csv_sha256": hashlib.sha256(
                config.sonnet_review_path.read_bytes()
            ).hexdigest(),
        },
    }
    config.audit_report_path.write_text(json.dumps(report), encoding="utf-8")


def _fixture(tmp_path: Path) -> GutenbergResolvedBuildConfig:
    root = tmp_path
    cache_dir = root / "data/local/gutenberg/test"
    cache_dir.mkdir(parents=True)

    standard_poem = _sonnet("Standard")
    general = (
        "General opening with unique historical words.\n"
        + standard_poem
        + "General closing with other unique words.\n"
    )
    standard_start = general.index(standard_poem)
    standard_end = standard_start + len(standard_poem)

    conditioned_source = "Conditioned dialect source kept physically separate and inactive.\n"
    excluded_source = "Canonical duplicate text that must not be materialized.\n"
    false_positive = _sonnet("Dramatic false positive retained as broader poetry")

    conditioned_poem = _sonnet("Conditioned embedded")
    bridge = (
        "Bridge opening with unique nineteenth century words.\n"
        + conditioned_poem
        + "Bridge closing with distinct words.\n"
    )
    conditioned_start = bridge.index(conditioned_poem)
    conditioned_end = conditioned_start + len(conditioned_poem)

    texts = {
        "1": general,
        "2": conditioned_source,
        "3": excluded_source,
        "4": false_positive,
        "5": bridge,
    }
    raw_texts = {ebook_id: _gutenberg(text) for ebook_id, text in texts.items()}
    for ebook_id, raw in raw_texts.items():
        path = cache_dir / f"pg{ebook_id}.txt"
        path.write_text(raw, encoding="utf-8")
        assert strip_gutenberg_boilerplate(raw) == texts[ebook_id]

    sources = [
        _source_row(
            "1",
            general,
            title="General work",
            role="historical_general",
            decision="eligible_standard_core_pending_processed_build",
            included_characters=len(general) - len(standard_poem),
        ),
        _source_row(
            "2",
            conditioned_source,
            title="Conditioned work",
            role="historical_non_sonnet_poetry",
            decision="conditioned_candidate_not_activated",
            source_pool="conditioned_metadata_pool",
            language_route="conditioned_separate",
            included_characters=0,
        ),
        _source_row(
            "3",
            excluded_source,
            title="Duplicate work",
            role="historical_general",
            decision="exclude_canonical_cross_corpus_duplicate",
            included_characters=0,
        ),
        _source_row(
            "4",
            false_positive,
            title="Poetry work",
            role="historical_non_sonnet_poetry",
            decision="eligible_standard_core_pending_processed_build",
        ),
        _source_row(
            "5",
            bridge,
            title="Bridge work",
            role="nineteenth_century_bridge",
            decision="eligible_standard_core_pending_processed_build",
            included_characters=len(bridge) - len(conditioned_poem),
        ),
    ]
    for row in sources:
        row["cache_sha256"] = _sha(raw_texts[row["ebook_id"]])
    source_path = root / "data/metadata/source.csv"
    _write_csv(source_path, RECORD_MANIFEST_FIELDS[:0] or tuple(sources[0]), sources)

    segments = []
    segments.extend(
        _segments(
            "1",
            general,
            [
                (0, standard_start, "include_record_text", "historical_general", "retained"),
                (
                    standard_start,
                    standard_end,
                    "quarantine_sonnet_candidate",
                    "standard_sonnets",
                    "sonnet",
                ),
                (
                    standard_end,
                    len(general),
                    "include_record_text",
                    "historical_general",
                    "retained",
                ),
            ],
        )
    )
    segments.extend(
        _segments(
            "2",
            conditioned_source,
            [
                (
                    0,
                    len(conditioned_source),
                    "conditioned_not_activated",
                    "conditioned_language_variant",
                    "inactive",
                )
            ],
        )
    )
    segments.extend(
        _segments(
            "3",
            excluded_source,
            [(0, len(excluded_source), "exclude_full_source_duplicate", "excluded", "duplicate")],
        )
    )
    segments.extend(
        _segments(
            "4",
            false_positive,
            [
                (
                    0,
                    len(false_positive),
                    "include_record_text",
                    "historical_non_sonnet_poetry",
                    "retained_false_positive",
                )
            ],
        )
    )
    segments.extend(
        _segments(
            "5",
            bridge,
            [
                (
                    0,
                    conditioned_start,
                    "include_record_text",
                    "nineteenth_century_bridge",
                    "retained",
                ),
                (
                    conditioned_start,
                    conditioned_end,
                    "quarantine_conditioned_sonnet_candidate",
                    "conditioned_language_variant",
                    "inactive_sonnet",
                ),
                (
                    conditioned_end,
                    len(bridge),
                    "include_record_text",
                    "nineteenth_century_bridge",
                    "retained",
                ),
            ],
        )
    )
    segment_path = root / "data/metadata/segments.csv"
    _write_csv(segment_path, tuple(segments[0]), segments)

    candidates = [
        _candidate(
            "1",
            "General work",
            general,
            standard_start,
            standard_end,
            "eligible_standard_sonnet_pending_processed_build",
            "accept_structurally_verified_standard_sonnet",
        ),
        _candidate(
            "4",
            "Poetry work",
            false_positive,
            0,
            len(false_positive),
            "exclude_manual_not_sonnet",
            "exclude_not_sonnet",
        ),
        _candidate(
            "5",
            "Bridge work",
            bridge,
            conditioned_start,
            conditioned_end,
            "conditioned_sonnet_candidate_not_activated",
            "exclude_nonstandard_language_sonnet",
        ),
    ]
    sonnet_path = root / "data/metadata/sonnets.csv"
    _write_csv(sonnet_path, tuple(candidates[0]), candidates)
    review_path = root / "data/metadata/review.csv"
    _write_csv(
        review_path,
        (
            "candidate_id",
            "ebook_id",
            "source_text_sha256",
            "review_resolution",
            "review_rationale",
        ),
        [
            {
                "candidate_id": row["candidate_id"],
                "ebook_id": row["ebook_id"],
                "source_text_sha256": row["source_text_sha256"],
                "review_resolution": row["manual_review_resolution"],
                "review_rationale": row["manual_review_rationale"],
            }
            for row in candidates
        ],
    )

    inventory_path = root / "data/metadata/inventory.csv"
    inventory_fields = (
        "ebook_id",
        "copyright",
        "media_type",
        "plain_text_url",
    )
    _write_csv(
        inventory_path,
        inventory_fields,
        [
            {
                "ebook_id": ebook_id,
                "copyright": "False",
                "media_type": "Text",
                "plain_text_url": f"https://example.test/{ebook_id}.txt",
            }
            for ebook_id in texts
        ],
    )
    bibit_manifest = root / "data/processed/bibit/records_manifest.csv"
    _write_csv(
        bibit_manifest,
        ("object_id", "artifact_status", "shard_path", "byte_start", "byte_end"),
        [],
    )
    broader_manifest = root / "data/metadata/broader.csv"
    _write_csv(broader_manifest, ("source_id", "expected_clean_text_path"), [])

    config = GutenbergResolvedBuildConfig(
        repo_root=root,
        source_decisions_path=source_path,
        segment_decisions_path=segment_path,
        sonnet_decisions_path=sonnet_path,
        sonnet_review_path=review_path,
        audit_report_path=root / "reports/audit.json",
        inventory_path=inventory_path,
        bibit_record_manifest_path=bibit_manifest,
        broader_sources_manifest_path=broader_manifest,
        output_dir=root / "data/processed/project_gutenberg_resolved_v1",
        markdown_report_path=root / "reports/build.md",
        max_shard_bytes=2048,
        expected_source_count=5,
        expected_eligible_source_count=3,
        expected_conditioned_source_count=1,
        expected_standard_sonnet_count=1,
        expected_conditioned_sonnet_count=1,
        progress_interval=1,
    )
    config.audit_report_path.parent.mkdir(parents=True, exist_ok=True)
    _refresh_audit_report(config)
    return config


def _slice(root: Path, row: dict[str, str]) -> str:
    payload = (root / row["shard_path"]).read_bytes()
    return payload[int(row["byte_start"]) : int(row["byte_end"])].decode("utf-8")


def test_build_materializes_recoverable_separated_artifacts(tmp_path):
    config = _fixture(tmp_path)
    messages = []

    report = build_gutenberg_resolved_corpus(config, progress=messages.append)

    assert report["source_count"] == 5
    assert report["materialized_source_count"] == 4
    assert report["materialized_sonnet_count"] == 2
    assert report["retained_source_characters_by_role"][
        "conditioned_source_variants"
    ] == len("Conditioned dialect source kept physically separate and inactive.\n")
    assert report["retained_source_characters_by_role"][
        "historical_non_sonnet_poetry"
    ] == len(_sonnet("Dramatic false positive retained as broader poetry"))
    assert report["deduplication"]["internal_near_duplicate_pair_count"] == 0
    assert report["policy"]["v7_split_assigned"] is False
    assert report["policy"]["conditioned_experiment_authorized"] is False
    assert "source-build 5/5" in " ".join(messages)

    records = {
        row["ebook_id"]: row
        for row in _read_csv(config.output_dir / "records_manifest.csv")
    }
    sonnets = {
        row["ebook_id"]: row
        for row in _read_csv(config.output_dir / "sonnets_manifest.csv")
    }
    general = _slice(config.repo_root, records["1"])
    poetry = _slice(config.repo_root, records["4"])
    bridge = _slice(config.repo_root, records["5"])
    conditioned_source = _slice(config.repo_root, records["2"])
    assert "Standard verso poetico" not in general
    assert "Dramatic false positive" in poetry
    assert "Conditioned embedded" not in bridge
    assert "Conditioned dialect source" in conditioned_source
    assert "conditioned_source_variants" in records["2"]["shard_path"]
    assert sonnets["1"]["poem_author"] == ""
    assert sonnets["1"]["poem_author_resolution"] == "pending_candidate_level_attribution_audit"
    assert "conditioned_sonnet_variants" in sonnets["5"]["shard_path"]
    assert not sonnets["4"]["shard_path"]

    attribution = _read_csv(config.output_dir / "attribution_manifest.csv")
    assert len(attribution) == 5
    assert all(row["catalog_copyright"] == "False" for row in attribution)
    segments = _read_csv(config.output_dir / "segments_manifest.csv")
    for row in segments:
        if not row["output_shard_path"]:
            continue
        payload = (config.repo_root / row["output_shard_path"]).read_bytes()
        recovered = payload[
            int(row["output_byte_start"]) : int(row["output_byte_end"])
        ]
        assert hashlib.sha256(recovered).hexdigest() == row["output_sha256"]
    assert all(
        path.stat().st_size <= config.max_shard_bytes
        for path in config.output_dir.rglob("part-*.txt")
    )
    assert (config.repo_root / "data/local/gutenberg/test/pg1.txt").is_file()


def test_build_is_deterministic(tmp_path):
    config = _fixture(tmp_path)

    first = build_gutenberg_resolved_corpus(config)
    first_files = {
        path.relative_to(config.output_dir): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in config.output_dir.rglob("*")
        if path.is_file()
    }
    second = build_gutenberg_resolved_corpus(config)

    assert second == first
    assert {
        path.relative_to(config.output_dir): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in config.output_dir.rglob("*")
        if path.is_file()
    } == first_files


def test_failed_directory_swap_restores_previous_verified_build(tmp_path, monkeypatch):
    config = _fixture(tmp_path)
    build_gutenberg_resolved_corpus(config)
    previous_hashes = {
        path.relative_to(config.output_dir): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in config.output_dir.rglob("*")
        if path.is_file()
    }
    real_replace = build_module.os.replace

    def fail_new_install(source, destination):
        source_path = Path(source)
        if source_path.name.startswith(
            f".{config.output_dir.name}."
        ) and not source_path.name.endswith(".previous"):
            raise OSError("simulated verified-output install failure")
        real_replace(source, destination)

    monkeypatch.setattr(build_module.os, "replace", fail_new_install)

    with pytest.raises(OSError, match="simulated verified-output install failure"):
        build_gutenberg_resolved_corpus(config)

    assert {
        path.relative_to(config.output_dir): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in config.output_dir.rglob("*")
        if path.is_file()
    } == previous_hashes


def test_build_rejects_changed_cached_primary_text(tmp_path):
    config = _fixture(tmp_path)
    cache_path = config.repo_root / "data/local/gutenberg/test/pg1.txt"
    cache_path.write_text(cache_path.read_text(encoding="utf-8") + "changed", encoding="utf-8")

    with pytest.raises(ValueError, match="cached primary-text hash mismatch"):
        build_gutenberg_resolved_corpus(config)

    assert not config.output_dir.exists()


def test_build_rejects_stale_checkpoint_3a_hash(tmp_path):
    config = _fixture(tmp_path)
    with config.sonnet_review_path.open("a", encoding="utf-8") as handle:
        handle.write("changed\n")

    with pytest.raises(ValueError, match="checkpoint-3A audit hash mismatch"):
        build_gutenberg_resolved_corpus(config)


def test_build_rejects_new_final_exact_duplicate(tmp_path):
    config = _fixture(tmp_path)
    sources = _read_csv(config.source_decisions_path)
    segments = _read_csv(config.segment_decisions_path)
    inventory = _read_csv(config.inventory_path)
    source_one = dict(sources[0])
    source_one["ebook_id"] = "6"
    source_one["title"] = "Duplicated residual work"
    source_one["cache_path"] = "data/local/gutenberg/test/pg6.txt"
    source_one["source_url"] = "https://www.gutenberg.org/ebooks/6"
    raw = (config.repo_root / sources[0]["cache_path"]).read_text(encoding="utf-8")
    (config.repo_root / source_one["cache_path"]).write_text(raw, encoding="utf-8")
    source_one["cache_sha256"] = _sha(raw)
    sources.append(source_one)
    source_segments = [dict(row) for row in segments if row["ebook_id"] == "1"]
    for index, row in enumerate(source_segments, start=1):
        row["ebook_id"] = "6"
        row["segment_id"] = f"pg6:seg{index:04d}"
    segments.extend(source_segments)
    inventory.append(
        {
            "ebook_id": "6",
            "copyright": "False",
            "media_type": "Text",
            "plain_text_url": "https://example.test/6.txt",
        }
    )
    _write_csv(config.source_decisions_path, tuple(sources[0]), sources)
    _write_csv(config.segment_decisions_path, tuple(segments[0]), segments)
    _write_csv(config.inventory_path, tuple(inventory[0]), inventory)
    config = GutenbergResolvedBuildConfig(
        **{
            **config.__dict__,
            "expected_source_count": 6,
            "expected_eligible_source_count": 4,
        }
    )
    _refresh_audit_report(config)

    with pytest.raises(ValueError, match="normalized exact duplicates"):
        build_gutenberg_resolved_corpus(config)


def test_public_manifest_field_contracts_are_unique():
    for fields in (
        RECORD_MANIFEST_FIELDS,
        SEGMENT_MANIFEST_FIELDS,
        SONNET_MANIFEST_FIELDS,
        ATTRIBUTION_MANIFEST_FIELDS,
    ):
        assert len(fields) == len(set(fields))
