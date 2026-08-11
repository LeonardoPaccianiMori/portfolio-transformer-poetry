import csv
import hashlib
import json
from pathlib import Path

import pytest

from sonnet_corpus.canonical_corpus_reader import (
    CanonicalCorpusReader,
    render_acceptance_markdown,
    write_acceptance_reports,
)


RECORD_FIELDS = (
    "unit_id", "source_group", "source_id", "title", "author", "source_archive",
    "source_url", "epoch_bucket", "final_role", "attribution_id",
    "logical_character_count", "logical_byte_count", "logical_sha256", "storage_kind",
    "storage_path", "byte_start", "byte_end", "activation_status", "training_eligible",
)
SONNET_FIELDS = (
    "unit_id", "source_group", "source_id", "title", "author", "source_archive",
    "source_url", "epoch_bucket", "original_split", "attribution_id", "line_count",
    "logical_character_count", "logical_byte_count", "logical_sha256", "storage_kind",
    "storage_path", "byte_start", "byte_end", "activation_status", "training_eligible",
)
STORAGE_FIELDS = (
    "unit_id", "unit_kind", "final_role", "storage_kind", "storage_path", "byte_start",
    "byte_end", "logical_character_count", "logical_byte_count", "logical_sha256",
    "physical_file_sha256", "public_repository_status",
)
ATTRIBUTION_FIELDS = ("attribution_id", "activation_status")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _base_row(unit_id: str, text: str, start: int, *, training: bool = True) -> dict[str, object]:
    payload = text.encode("utf-8")
    return {
        "unit_id": unit_id,
        "source_group": "fixture",
        "source_id": unit_id,
        "title": unit_id,
        "author": "Autore",
        "source_archive": "Fixture",
        "source_url": "https://example.test/work",
        "epoch_bucket": "1400",
        "attribution_id": "attr:fixture",
        "logical_character_count": len(text),
        "logical_byte_count": len(payload),
        "logical_sha256": _sha(text),
        "storage_kind": "existing_committed_slice",
        "storage_path": "data/processed/fixture/part-0001.txt",
        "byte_start": start,
        "byte_end": start + len(payload),
        "activation_status": (
            "inactive_pending_v7" if training else "protected_v6_validation_test"
        ),
        "training_eligible": str(training).lower(),
    }


def _build_fixture(tmp_path: Path) -> tuple[CanonicalCorpusReader, Path]:
    corpus_dir = tmp_path / "data/processed/canonical"
    shard_path = tmp_path / "data/processed/fixture/part-0001.txt"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    texts = ("Città\n", "Amor\n", "Segreto\n")
    shard = "".join(texts)
    shard_path.write_text(shard, encoding="utf-8")
    physical_sha = _sha(shard)

    record = _base_row("record:1", texts[0], 0)
    record["final_role"] = "historical_general"
    excluded = dict(record)
    excluded.update(
        {
            "unit_id": "record:excluded",
            "source_id": "record:excluded",
            "title": "Excluded",
            "logical_character_count": 0,
            "logical_byte_count": 0,
            "logical_sha256": "",
            "storage_kind": "none",
            "storage_path": "",
            "byte_start": "",
            "byte_end": "",
            "activation_status": "inactive_excluded",
            "training_eligible": "false",
        }
    )
    first_end = len(texts[0].encode("utf-8"))
    sonnet = _base_row("sonnet:train", texts[1], first_end)
    sonnet.update({"original_split": "", "line_count": 14})
    protected_start = first_end + len(texts[1].encode("utf-8"))
    protected = _base_row(
        "sonnet:protected", texts[2], protected_start, training=False
    )
    protected.update({"original_split": "validation", "line_count": 14})

    _write_csv(corpus_dir / "records_manifest.csv", RECORD_FIELDS, [record, excluded])
    _write_csv(corpus_dir / "sonnets_manifest.csv", SONNET_FIELDS, [sonnet, protected])
    storage_rows = []
    for row, kind, role in (
        (record, "broader", "historical_general"),
        (sonnet, "standard_sonnet", "standard_sonnets"),
        (protected, "standard_sonnet", "standard_sonnets"),
    ):
        storage_rows.append(
            {
                "unit_id": row["unit_id"],
                "unit_kind": kind,
                "final_role": role,
                "storage_kind": row["storage_kind"],
                "storage_path": row["storage_path"],
                "byte_start": row["byte_start"],
                "byte_end": row["byte_end"],
                "logical_character_count": row["logical_character_count"],
                "logical_byte_count": row["logical_byte_count"],
                "logical_sha256": row["logical_sha256"],
                "physical_file_sha256": physical_sha,
                "public_repository_status": "committed_or_checkpoint_delta",
            }
        )
    _write_csv(corpus_dir / "storage_manifest.csv", STORAGE_FIELDS, storage_rows)
    _write_csv(
        corpus_dir / "attribution_manifest.csv",
        ATTRIBUTION_FIELDS,
        [{"attribution_id": "attr:fixture", "activation_status": "inactive_pending_v7"}],
    )
    report = {
        "build_version": "fixture_v1",
        "activation_status": "inactive_pending_v7",
        "record_universe_count": 2,
        "sonnet_universe_count": 2,
        "training_record_count": 1,
        "training_sonnet_count": 1,
        "protected_v6_sonnet_count": 1,
        "logical_character_count": len(texts[0]) + len(texts[1]),
        "logical_role_characters": {
            "historical_general": len(texts[0]),
            "standard_sonnets": len(texts[1]),
        },
        "verification": {
            "conditioned_material_included": False,
            "v7_created": False,
            "mixture_weights_assigned": False,
            "gpu_work_started": False,
        },
    }
    (corpus_dir / "build_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    return CanonicalCorpusReader(
        tmp_path, corpus_dir, expected_protected_v6_count=1
    ), shard_path


def test_reader_defaults_to_training_units_and_requires_explicit_protected_mode(tmp_path):
    reader, _ = _build_fixture(tmp_path)

    assert [unit.unit_id for unit in reader.iter_records()] == ["record:1"]
    assert [unit.unit_id for unit in reader.iter_sonnets()] == ["sonnet:train"]
    assert [unit.unit_id for unit in reader.iter_sonnets(eligibility="protected")] == [
        "sonnet:protected"
    ]
    assert len(list(reader.iter_units(eligibility="stored"))) == 3


def test_reader_reads_multibyte_utf8_slice_without_flattening_corpus(tmp_path):
    reader, _ = _build_fixture(tmp_path)

    unit = next(reader.iter_records())

    assert reader.read_text(unit) == "Città\n"


def test_reader_rejects_unknown_role_kind_and_eligibility(tmp_path):
    reader, _ = _build_fixture(tmp_path)

    with pytest.raises(ValueError, match="unknown canonical corpus role"):
        list(reader.iter_units(role="conditioned"))
    with pytest.raises(ValueError, match="unknown unit kind"):
        list(reader.iter_units(unit_kind="poem"))
    with pytest.raises(ValueError, match="unknown eligibility mode"):
        list(reader.iter_units(eligibility="all"))


def test_exhaustive_verify_checks_every_slice_and_freezes_identity(tmp_path):
    reader, _ = _build_fixture(tmp_path)
    progress = []

    report = reader.verify(
        progress=lambda completed, total, path: progress.append((completed, total, path))
    )

    assert report["acceptance_status"] == "pass"
    assert report["stored_unit_count"] == 3
    assert report["physical_file_count"] == 1
    assert report["training_logical_character_count"] == 11
    assert report["protected_v6_sonnet_count"] == 1
    assert report["verification"]["protected_v6_training_excluded"] is True
    assert progress == [(1, 1, "data/processed/fixture/part-0001.txt")]
    assert len(report["logical_identity_sha256"]) == 64
    assert len(report["physical_identity_sha256"]) == 64


def test_exhaustive_verify_rejects_physical_file_tampering(tmp_path):
    reader, shard_path = _build_fixture(tmp_path)
    shard_path.write_text("tampered", encoding="utf-8")

    with pytest.raises(ValueError, match="physical file hash mismatch"):
        reader.verify()


def test_reader_rejects_missing_physical_file(tmp_path):
    reader, shard_path = _build_fixture(tmp_path)
    shard_path.unlink()

    with pytest.raises(FileNotFoundError):
        reader.read_text(next(reader.iter_records()))


def test_reader_rejects_local_cache_storage_reference(tmp_path):
    _, _ = _build_fixture(tmp_path)
    corpus_dir = tmp_path / "data/processed/canonical"
    for name in ("records_manifest.csv", "storage_manifest.csv"):
        path = corpus_dir / name
        text = path.read_text(encoding="utf-8")
        path.write_text(
            text.replace(
                "data/processed/fixture/part-0001.txt",
                "data/local/fixture/part-0001.txt",
            ),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="local-cache storage path is forbidden"):
        CanonicalCorpusReader(
            tmp_path, corpus_dir, expected_protected_v6_count=1
        )


def test_reader_rejects_manifest_storage_join_mismatch(tmp_path):
    _, _ = _build_fixture(tmp_path)
    corpus_dir = tmp_path / "data/processed/canonical"
    path = corpus_dir / "records_manifest.csv"
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text.replace("historical_general", "historical_non_sonnet_poetry"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown role|mismatch"):
        CanonicalCorpusReader(
            tmp_path, corpus_dir, expected_protected_v6_count=1
        )


def test_acceptance_report_writers_are_deterministic(tmp_path):
    reader, _ = _build_fixture(tmp_path)
    report = reader.verify()
    json_path = tmp_path / "reports/acceptance.json"
    markdown_path = tmp_path / "reports/acceptance.md"

    write_acceptance_reports(report, json_path, markdown_path)
    first = (json_path.read_bytes(), markdown_path.read_bytes())
    write_acceptance_reports(report, json_path, markdown_path)

    assert first == (json_path.read_bytes(), markdown_path.read_bytes())
    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    assert render_acceptance_markdown(report).endswith("\n")
    assert "creates no V7 split" in markdown_path.read_text(encoding="utf-8")


def test_real_canonical_reader_freezes_expected_default_boundaries():
    root = Path(__file__).resolve().parents[1]
    reader = CanonicalCorpusReader(
        root, root / "data/processed/canonical_italian_corpora_v1"
    )

    assert len(reader.units) == 26_934
    assert sum(1 for _ in reader.iter_records()) == 4_544
    assert sum(1 for _ in reader.iter_sonnets()) == 22_003
    assert sum(1 for _ in reader.iter_sonnets(eligibility="protected")) == 387
    assert all("data/local" not in unit.storage_path for unit in reader.units)

    report = json.loads(
        (root / "reports/canonical_italian_corpus_acceptance_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["acceptance_status"] == "pass"
    assert report["stored_unit_count"] == 26_934
    assert report["training_logical_character_count"] == 643_822_187
    assert report["logical_identity_sha256"] == (
        "0aeb0ee8ffed91c294b31f27fa85471418acf4e5ff47cf84a17a5e2deb666b57"
    )
    assert report["verification"]["v7_created"] is False
