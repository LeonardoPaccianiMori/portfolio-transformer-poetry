import csv
import hashlib
import json
from pathlib import Path

import pytest

from sonnet_corpus.bibit_processed_build import (
    BibItProcessedBuildConfig,
    build_bibit_processed_corpus,
)
from sonnet_corpus.biblioteca_italiana import parse_bibit_tei


def _tei(object_id: str, prose: str, sonnet_lines: list[str]) -> bytes:
    sonnet = "".join(f"<l>{line}</l>" for line in sonnet_lines)
    return f"""<TEI.2>
      <teiHeader><fileDesc>
        <titleStmt><title>Opera {object_id}</title><author>Autore</author></titleStmt>
        <publicationStmt><publisher>BibIt</publisher><idno>{object_id}</idno>
          <availability><p>Uso scientifico.</p></availability></publicationStmt>
        <sourceDesc><bibl><title>Edizione</title><idno>TEST</idno></bibl></sourceDesc>
      </fileDesc></teiHeader>
      <text><body><p>{prose}</p><lg type="sonetto">{sonnet}</lg></body></text>
    </TEI.2>""".encode()


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_shard_slice(root: Path, row: dict[str, str]) -> str:
    payload = (root / row["shard_path"]).read_bytes()
    return payload[int(row["byte_start"]) : int(row["byte_end"])].decode("utf-8")


def _fixture(tmp_path: Path) -> tuple[BibItProcessedBuildConfig, str, str]:
    root = tmp_path
    tei_dir = root / "data/local/bibit/tei"
    tei_dir.mkdir(parents=True)
    general_lines = [f"Verso generale {index}" for index in range(1, 15)]
    bridge_lines = [f"Verso ponte {index}" for index in range(1, 16)]
    documents = {
        "bibit000001": _tei(
            "bibit000001",
            "Testo [1] generale con parola [restituita].",
            general_lines,
        ),
        "bibit000002": _tei(
            "bibit000002",
            "Testo ottocentesco preservato.",
            bridge_lines,
        ),
    }
    parsed = {}
    for object_id, xml in documents.items():
        (tei_dir / f"{object_id}.xml").write_bytes(xml)
        parsed[object_id] = parse_bibit_tei(xml, object_id=object_id)

    record_decisions = root / "record_decisions.csv"
    record_rows = []
    for object_id, role, policy in (
        (
            "bibit000001",
            "historical_general",
            "strip_editorial_square_delimiters_and_labels",
        ),
        (
            "bibit000002",
            "nineteenth_century_bridge",
            "preserve_rendered_tei_text",
        ),
    ):
        xml = documents[object_id]
        record_rows.append(
            {
                "object_id": object_id,
                "title": f"Opera {object_id}",
                "authors": "Autore",
                "decision": "activate_core",
                "final_role": role,
                "cleaning_policy": policy,
                "included_characters": str(len(parsed[object_id].sonnet_candidate_safe_text)),
                "landing_page_url": f"https://example.test/{object_id}",
                "tei_sha256": hashlib.sha256(xml).hexdigest(),
            }
        )
    _write_csv(record_decisions, record_rows)

    sonnet_decisions = root / "sonnet_decisions.csv"
    sonnet_rows = []
    for object_id, role, policy in (
        (
            "bibit000001",
            "sonnet_core_standard_14_line",
            "strip_editorial_square_delimiters_and_labels",
        ),
        (
            "bibit000002",
            "sonnet_variant_conditioned_auxiliary",
            "preserve_rendered_tei_text",
        ),
    ):
        unit = parsed[object_id].sonnets[0]
        sonnet_rows.append(
            {
                "candidate_id": f"{object_id}:{unit.unit_id}",
                "object_id": object_id,
                "title": f"Opera {object_id}",
                "candidate_author": "Autore",
                "author_resolution": "catalog_author",
                "periods": "500",
                "source_kind": "explicit_tei_sonnet",
                "tei_type": unit.sonnet_type,
                "heading_path": "",
                "line_count": str(unit.line_count),
                "stanza_pattern": str(unit.line_count),
                "decision": "activate_standard_explicit_sonnet",
                "final_role": role,
                "cleaning_policy": policy,
                "character_count": str(len(unit.text)),
                "text_sha256": hashlib.sha256(unit.text.encode()).hexdigest(),
                "landing_page_url": f"https://example.test/{object_id}",
            }
        )
    _write_csv(sonnet_decisions, sonnet_rows)

    config = BibItProcessedBuildConfig(
        repo_root=root,
        record_decisions_path=record_decisions,
        sonnet_decisions_path=sonnet_decisions,
        tei_cache_dir=tei_dir,
        output_dir=root / "data/processed/bibit_resolved_v1",
        markdown_report_path=root / "reports/bibit_resolved_v1_build.md",
        max_shard_bytes=1024,
        progress_interval=1,
    )
    return config, "\n".join(general_lines), "\n".join(bridge_lines)


def test_build_bibit_processed_corpus_writes_recoverable_role_shards(tmp_path):
    config, general_sonnet, bridge_sonnet = _fixture(tmp_path)
    messages = []

    report = build_bibit_processed_corpus(config, progress=messages.append)

    assert report["record_count"] == 2
    assert report["sonnet_count"] == 2
    assert report["policy"]["v7_split_assigned"] is False
    assert report["policy"]["training_mixture_weight_assigned"] is False
    assert config.markdown_report_path.is_file()
    assert "record 2/2" in " ".join(messages)
    with (config.output_dir / "records_manifest.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        records = {row["object_id"]: row for row in csv.DictReader(handle)}
    with (config.output_dir / "sonnets_manifest.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        sonnets = {row["object_id"]: row for row in csv.DictReader(handle)}

    general_text = _read_shard_slice(config.repo_root, records["bibit000001"])
    assert "[1]" not in general_text
    assert "restituita" in general_text
    assert general_sonnet not in general_text
    assert bridge_sonnet not in _read_shard_slice(
        config.repo_root, records["bibit000002"]
    )
    assert _read_shard_slice(config.repo_root, sonnets["bibit000001"]).count("\n") == 14
    assert sonnets["bibit000002"]["final_role"] == "sonnet_variant_conditioned_auxiliary"
    assert all(
        path.stat().st_size <= config.max_shard_bytes
        for path in config.output_dir.rglob("part-*.txt")
    )


def test_build_bibit_processed_corpus_is_deterministic(tmp_path):
    config, _, _ = _fixture(tmp_path)

    first = build_bibit_processed_corpus(config)
    first_report = (config.output_dir / "build_report.json").read_bytes()
    first_files = {
        path.relative_to(config.output_dir): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in config.output_dir.rglob("*")
        if path.is_file()
    }
    second = build_bibit_processed_corpus(config)

    assert second == first
    assert (config.output_dir / "build_report.json").read_bytes() == first_report
    assert {
        path.relative_to(config.output_dir): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in config.output_dir.rglob("*")
        if path.is_file()
    } == first_files
    assert json.loads(first_report)["build_version"] == "bibit_resolved_v1"


def test_build_bibit_processed_corpus_rejects_changed_cached_tei(tmp_path):
    config, _, _ = _fixture(tmp_path)
    cache_path = config.tei_cache_dir / "bibit000001.xml"
    cache_path.write_bytes(cache_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="cached TEI hash mismatch"):
        build_bibit_processed_corpus(config)

    assert not config.output_dir.exists()


def test_build_bibit_processed_corpus_retains_empty_sonnet_source_lineage(tmp_path):
    config, _, _ = _fixture(tmp_path)
    cache_path = config.tei_cache_dir / "bibit000002.xml"
    xml = _tei(
        "bibit000002",
        "",
        [f"Verso ponte {index}" for index in range(1, 16)],
    )
    cache_path.write_bytes(xml)
    parsed = parse_bibit_tei(xml, object_id="bibit000002")
    with config.record_decisions_path.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    records[1]["included_characters"] = str(len(parsed.sonnet_candidate_safe_text))
    records[1]["tei_sha256"] = hashlib.sha256(xml).hexdigest()
    _write_csv(config.record_decisions_path, records)
    with config.sonnet_decisions_path.open(encoding="utf-8", newline="") as handle:
        sonnets = list(csv.DictReader(handle))
    unit = parsed.sonnets[0]
    sonnets[1]["character_count"] = str(len(unit.text))
    sonnets[1]["text_sha256"] = hashlib.sha256(unit.text.encode()).hexdigest()
    _write_csv(config.sonnet_decisions_path, sonnets)

    report = build_bibit_processed_corpus(config)

    assert report["record_count"] == 2
    assert report["record_text_count"] == 1
    assert report["empty_record_text_count"] == 1
    with (config.output_dir / "records_manifest.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        bridge = next(
            row for row in csv.DictReader(handle) if row["object_id"] == "bibit000002"
        )
    assert bridge["artifact_status"] == "sonnet_source_without_residual_record_text"
    assert bridge["shard_path"] == ""
    with (config.output_dir / "sonnets_manifest.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        assert any(row["object_id"] == "bibit000002" for row in csv.DictReader(handle))
