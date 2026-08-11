"""Stream and verify the inactive canonical Italian logical corpus.

Checkpoint 7B deliberately stores logical units as byte slices of committed
shards.  This module is the single consumer for those slices: it preserves
document boundaries, validates paths and manifest joins, and can exhaustively
verify every referenced byte range without concatenating the corpus.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal


ACCEPTANCE_VERSION = "canonical_italian_corpus_acceptance_v1"
ACCEPTANCE_DATE = "2026-08-11"
CORPUS_ROLES = frozenset(
    {
        "historical_general",
        "historical_non_sonnet_poetry",
        "nineteenth_century_bridge",
        "standard_sonnets",
    }
)
ELIGIBILITY_MODES = frozenset({"training", "protected", "stored"})

Progress = Callable[[int, int, str], None]
EligibilityMode = Literal["training", "protected", "stored"]


@dataclass(frozen=True)
class CanonicalTextUnit:
    """One manifest identity and its verified logical storage location."""

    unit_id: str
    unit_kind: str
    source_group: str
    source_id: str
    title: str
    author: str
    source_archive: str
    source_url: str
    epoch_bucket: str
    final_role: str
    attribution_id: str
    activation_status: str
    training_eligible: bool
    storage_kind: str
    storage_path: str
    byte_start: int
    byte_end: int
    logical_character_count: int
    logical_byte_count: int
    logical_sha256: str
    physical_file_sha256: str
    original_split: str = ""
    line_count: int | None = None


class CanonicalCorpusReader:
    """Read document-level units from the checkpoint-7B logical manifests.

    The default iterators expose only training-eligible units.  Protected V6
    validation/test sonnets require the explicit ``eligibility="protected"``
    mode, which prevents accidental use as training text.
    """

    def __init__(
        self,
        repo_root: Path,
        corpus_dir: Path,
        *,
        expected_protected_v6_count: int = 387,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.corpus_dir = (
            corpus_dir.resolve()
            if corpus_dir.is_absolute()
            else (self.repo_root / corpus_dir).resolve()
        )
        if not self.corpus_dir.is_relative_to(self.repo_root):
            raise ValueError("canonical corpus directory must be inside the repository")
        if expected_protected_v6_count < 0:
            raise ValueError("expected protected V6 count must be non-negative")
        self.expected_protected_v6_count = expected_protected_v6_count

        self.record_manifest_path = self.corpus_dir / "records_manifest.csv"
        self.sonnet_manifest_path = self.corpus_dir / "sonnets_manifest.csv"
        self.storage_manifest_path = self.corpus_dir / "storage_manifest.csv"
        self.attribution_manifest_path = self.corpus_dir / "attribution_manifest.csv"
        self.build_report_path = self.corpus_dir / "build_report.json"

        self._record_rows = _read_csv(
            self.record_manifest_path,
            {
                "unit_id", "source_group", "source_id", "title", "author",
                "source_archive", "source_url", "epoch_bucket", "final_role",
                "attribution_id", "logical_character_count", "logical_byte_count",
                "logical_sha256", "storage_kind", "storage_path", "byte_start",
                "byte_end", "activation_status", "training_eligible",
            },
        )
        self._sonnet_rows = _read_csv(
            self.sonnet_manifest_path,
            {
                "unit_id", "source_group", "source_id", "title", "author",
                "source_archive", "source_url", "epoch_bucket", "original_split",
                "attribution_id", "line_count", "logical_character_count",
                "logical_byte_count", "logical_sha256", "storage_kind",
                "storage_path", "byte_start", "byte_end", "activation_status",
                "training_eligible",
            },
        )
        self._storage_rows = _read_csv(
            self.storage_manifest_path,
            {
                "unit_id", "unit_kind", "final_role", "storage_kind",
                "storage_path", "byte_start", "byte_end", "logical_character_count",
                "logical_byte_count", "logical_sha256", "physical_file_sha256",
                "public_repository_status",
            },
        )
        self._attribution_rows = _read_csv(
            self.attribution_manifest_path,
            {"attribution_id", "activation_status"},
        )
        if not self.build_report_path.is_file():
            raise FileNotFoundError(self.build_report_path)
        self.build_report = json.loads(self.build_report_path.read_text(encoding="utf-8"))

        self._attributions = _unique_rows(
            self._attribution_rows, "attribution_id", "attribution manifest"
        )
        self._storage = _unique_rows(
            self._storage_rows, "unit_id", "storage manifest"
        )
        self._units = self._build_units()

    @property
    def units(self) -> tuple[CanonicalTextUnit, ...]:
        """Return the immutable, unit-id-sorted stored-unit inventory."""

        return self._units

    def iter_units(
        self,
        *,
        unit_kind: str | None = None,
        role: str | None = None,
        eligibility: EligibilityMode = "training",
    ) -> Iterator[CanonicalTextUnit]:
        """Yield stored units while making protected-text access explicit."""

        if eligibility not in ELIGIBILITY_MODES:
            raise ValueError(f"unknown eligibility mode: {eligibility}")
        if role is not None and role not in CORPUS_ROLES:
            raise ValueError(f"unknown canonical corpus role: {role}")
        if unit_kind not in {None, "broader", "standard_sonnet"}:
            raise ValueError(f"unknown unit kind: {unit_kind}")

        for unit in self._units:
            if unit_kind is not None and unit.unit_kind != unit_kind:
                continue
            if role is not None and unit.final_role != role:
                continue
            if eligibility == "training" and not unit.training_eligible:
                continue
            if eligibility == "protected" and unit.activation_status != "protected_v6_validation_test":
                continue
            yield unit

    def iter_records(
        self,
        role: str | None = None,
        *,
        eligibility: EligibilityMode = "training",
    ) -> Iterator[CanonicalTextUnit]:
        """Yield broader document units, training-eligible by default."""

        return self.iter_units(
            unit_kind="broader", role=role, eligibility=eligibility
        )

    def iter_sonnets(
        self,
        *,
        eligibility: EligibilityMode = "training",
    ) -> Iterator[CanonicalTextUnit]:
        """Yield standard-sonnet units, training-eligible by default."""

        return self.iter_units(
            unit_kind="standard_sonnet",
            role="standard_sonnets",
            eligibility=eligibility,
        )

    def read_text(self, unit: CanonicalTextUnit) -> str:
        """Read one byte slice and verify its UTF-8, length, and logical hash."""

        path = self._resolve_storage_path(unit.storage_path)
        with path.open("rb") as handle:
            handle.seek(unit.byte_start)
            payload = handle.read(unit.logical_byte_count)
        return _verify_logical_payload(unit, payload)

    def verify(self, progress: Progress | None = None) -> dict[str, Any]:
        """Exhaustively verify all physical files and every logical byte slice."""

        by_path: dict[str, list[CanonicalTextUnit]] = defaultdict(list)
        for unit in self._units:
            by_path[unit.storage_path].append(unit)

        physical_bytes = 0
        physical_identity = hashlib.sha256()
        verified_logical_hashes: dict[str, str] = {}
        paths = sorted(by_path)
        for index, portable in enumerate(paths, 1):
            units = by_path[portable]
            expected_hashes = {unit.physical_file_sha256 for unit in units}
            if len(expected_hashes) != 1:
                raise ValueError(f"conflicting physical hashes for {portable}")
            expected_hash = next(iter(expected_hashes))
            path = self._resolve_storage_path(portable)
            payload = path.read_bytes()
            physical_bytes += len(payload)
            actual_hash = hashlib.sha256(payload).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(f"physical file hash mismatch: {portable}")
            physical_identity.update(portable.encode("utf-8"))
            physical_identity.update(b"\0")
            physical_identity.update(actual_hash.encode("ascii"))
            physical_identity.update(b"\n")

            for unit in units:
                if unit.byte_end > len(payload):
                    raise ValueError(f"byte range exceeds physical file: {unit.unit_id}")
                logical = payload[unit.byte_start:unit.byte_end]
                _verify_logical_payload(unit, logical)
                verified_logical_hashes[unit.unit_id] = unit.logical_sha256
            if progress is not None:
                progress(index, len(paths), portable)

        logical_identity = hashlib.sha256()
        for unit in self._units:
            logical_identity.update(unit.unit_id.encode("utf-8"))
            logical_identity.update(b"\0")
            logical_identity.update(unit.unit_kind.encode("ascii"))
            logical_identity.update(b"\0")
            logical_identity.update(unit.final_role.encode("ascii"))
            logical_identity.update(b"\0")
            logical_identity.update(verified_logical_hashes[unit.unit_id].encode("ascii"))
            logical_identity.update(b"\n")

        training = [unit for unit in self._units if unit.training_eligible]
        protected = [
            unit
            for unit in self._units
            if unit.activation_status == "protected_v6_validation_test"
        ]
        training_roles = Counter()
        for unit in training:
            training_roles[unit.final_role] += unit.logical_character_count

        report = {
            "acceptance_version": ACCEPTANCE_VERSION,
            "acceptance_date": ACCEPTANCE_DATE,
            "corpus_build_version": self.build_report["build_version"],
            "acceptance_status": "pass",
            "activation_status": "inactive_pending_v7",
            "record_universe_count": len(self._record_rows),
            "sonnet_universe_count": len(self._sonnet_rows),
            "stored_unit_count": len(self._units),
            "physical_file_count": len(paths),
            "physical_bytes_verified": physical_bytes,
            "training_record_count": sum(
                unit.training_eligible and unit.unit_kind == "broader"
                for unit in self._units
            ),
            "training_sonnet_count": sum(
                unit.training_eligible and unit.unit_kind == "standard_sonnet"
                for unit in self._units
            ),
            "protected_v6_sonnet_count": len(protected),
            "training_logical_character_count": sum(
                unit.logical_character_count for unit in training
            ),
            "stored_logical_character_count": sum(
                unit.logical_character_count for unit in self._units
            ),
            "stored_logical_byte_count": sum(
                unit.logical_byte_count for unit in self._units
            ),
            "training_role_characters": dict(sorted(training_roles.items())),
            "manifest_sha256": {
                _portable(path, self.repo_root): _file_sha256(path)
                for path in (
                    self.record_manifest_path,
                    self.sonnet_manifest_path,
                    self.storage_manifest_path,
                    self.attribution_manifest_path,
                    self.build_report_path,
                )
            },
            "logical_identity_sha256": logical_identity.hexdigest(),
            "physical_identity_sha256": physical_identity.hexdigest(),
            "verification": {
                "all_physical_files_hash_verified": True,
                "all_logical_slices_hash_verified": True,
                "all_utf8_boundaries_verified": True,
                "all_attribution_links_resolved": True,
                "all_paths_repository_relative": True,
                "local_cache_references_absent": True,
                "conditioned_material_absent": True,
                "protected_v6_training_excluded": all(
                    not unit.training_eligible for unit in protected
                ),
                "v7_created": False,
                "tokenization_performed": False,
                "mixture_weights_assigned": False,
                "gpu_work_started": False,
            },
        }
        self._verify_report_totals(report)
        return report

    def _build_units(self) -> tuple[CanonicalTextUnit, ...]:
        manifest_ids: set[str] = set()
        units: list[CanonicalTextUnit] = []
        for unit_kind, rows in (
            ("broader", self._record_rows),
            ("standard_sonnet", self._sonnet_rows),
        ):
            for row in rows:
                unit_id = row["unit_id"]
                if unit_id in manifest_ids:
                    raise ValueError(f"duplicate unit ID across manifests: {unit_id}")
                manifest_ids.add(unit_id)
                training_eligible = _parse_bool(
                    row["training_eligible"], f"training_eligible for {unit_id}"
                )
                if row["attribution_id"] not in self._attributions:
                    raise ValueError(f"missing attribution for {unit_id}")
                storage = self._storage.get(unit_id)
                if row["storage_kind"] == "none":
                    if storage is not None:
                        raise ValueError(f"excluded unit unexpectedly has storage: {unit_id}")
                    if training_eligible:
                        raise ValueError(f"training unit has no storage: {unit_id}")
                    continue
                if storage is None:
                    raise ValueError(f"stored manifest unit lacks storage row: {unit_id}")
                self._validate_manifest_storage_join(row, storage, unit_kind)
                role = storage["final_role"]
                if role not in CORPUS_ROLES:
                    raise ValueError(f"unknown role for {unit_id}: {role}")
                self._resolve_storage_path(storage["storage_path"])
                activation = row["activation_status"]
                if training_eligible and activation != "inactive_pending_v7":
                    raise ValueError(f"training unit is not inactive_pending_v7: {unit_id}")
                if activation == "protected_v6_validation_test" and training_eligible:
                    raise ValueError(f"protected sonnet is training eligible: {unit_id}")
                units.append(
                    CanonicalTextUnit(
                        unit_id=unit_id,
                        unit_kind=unit_kind,
                        source_group=row["source_group"],
                        source_id=row["source_id"],
                        title=row["title"],
                        author=row["author"],
                        source_archive=row["source_archive"],
                        source_url=row["source_url"],
                        epoch_bucket=row["epoch_bucket"],
                        final_role=role,
                        attribution_id=row["attribution_id"],
                        activation_status=activation,
                        training_eligible=training_eligible,
                        storage_kind=storage["storage_kind"],
                        storage_path=storage["storage_path"],
                        byte_start=int(storage["byte_start"]),
                        byte_end=int(storage["byte_end"]),
                        logical_character_count=int(storage["logical_character_count"]),
                        logical_byte_count=int(storage["logical_byte_count"]),
                        logical_sha256=storage["logical_sha256"],
                        physical_file_sha256=storage["physical_file_sha256"],
                        original_split=row.get("original_split", ""),
                        line_count=(int(row["line_count"]) if row.get("line_count") else None),
                    )
                )

        extra_storage = sorted(set(self._storage) - manifest_ids)
        if extra_storage:
            raise ValueError(f"storage rows lack manifest units: {extra_storage[:3]}")
        return tuple(sorted(units, key=lambda unit: unit.unit_id))

    def _validate_manifest_storage_join(
        self, row: dict[str, str], storage: dict[str, str], unit_kind: str
    ) -> None:
        expected_storage_kind = "broader" if unit_kind == "broader" else "standard_sonnet"
        if storage["unit_kind"] != expected_storage_kind:
            raise ValueError(f"unit-kind mismatch for {row['unit_id']}")
        if unit_kind == "broader" and row["final_role"] != storage["final_role"]:
            raise ValueError(f"manifest/storage final_role mismatch for {row['unit_id']}")
        if storage["public_repository_status"] != "committed_or_checkpoint_delta":
            raise ValueError(f"non-public storage status for {row['unit_id']}")
        for field in (
            "storage_kind", "storage_path", "byte_start", "byte_end",
            "logical_character_count", "logical_byte_count", "logical_sha256",
        ):
            if row[field] != storage[field]:
                raise ValueError(f"manifest/storage {field} mismatch for {row['unit_id']}")
        start = int(storage["byte_start"])
        end = int(storage["byte_end"])
        logical_bytes = int(storage["logical_byte_count"])
        if start < 0 or end <= start or end - start != logical_bytes:
            raise ValueError(f"invalid byte range for {row['unit_id']}")
        if len(storage["logical_sha256"]) != 64 or len(storage["physical_file_sha256"]) != 64:
            raise ValueError(f"invalid SHA-256 field for {row['unit_id']}")

    def _resolve_storage_path(self, portable: str) -> Path:
        pure = PurePosixPath(portable)
        if pure.is_absolute() or not pure.parts or ".." in pure.parts:
            raise ValueError(f"unsafe storage path: {portable}")
        if len(pure.parts) >= 2 and pure.parts[:2] == ("data", "local"):
            raise ValueError(f"local-cache storage path is forbidden: {portable}")
        resolved = (self.repo_root / Path(*pure.parts)).resolve()
        if not resolved.is_relative_to(self.repo_root):
            raise ValueError(f"storage path escapes repository: {portable}")
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        return resolved

    def _verify_report_totals(self, report: dict[str, Any]) -> None:
        expected_pairs = {
            "record_universe_count": "record_universe_count",
            "sonnet_universe_count": "sonnet_universe_count",
            "training_record_count": "training_record_count",
            "training_sonnet_count": "training_sonnet_count",
            "protected_v6_sonnet_count": "protected_v6_sonnet_count",
            "training_logical_character_count": "logical_character_count",
        }
        for acceptance_key, build_key in expected_pairs.items():
            if report[acceptance_key] != self.build_report[build_key]:
                raise ValueError(
                    f"acceptance/build report mismatch: {acceptance_key}"
                )
        if report["training_role_characters"] != self.build_report["logical_role_characters"]:
            raise ValueError("acceptance/build report role totals mismatch")
        if report["protected_v6_sonnet_count"] != self.expected_protected_v6_count:
            raise ValueError("protected V6 sonnet count changed")
        if self.build_report.get("activation_status") != "inactive_pending_v7":
            raise ValueError("canonical build is not inactive_pending_v7")
        build_verification = self.build_report.get("verification", {})
        expected_false = (
            "conditioned_material_included",
            "v7_created",
            "mixture_weights_assigned",
            "gpu_work_started",
        )
        if any(build_verification.get(key) is not False for key in expected_false):
            raise ValueError("canonical build safety boundary changed")


def write_acceptance_reports(
    report: dict[str, Any], json_path: Path, markdown_path: Path
) -> None:
    """Write deterministic machine-readable and portfolio-readable reports."""

    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    _replace_text(
        json_path,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _replace_text(markdown_path, render_acceptance_markdown(report))


def render_acceptance_markdown(report: dict[str, Any]) -> str:
    """Render the checkpoint-7C acceptance result without runtime-dependent data."""

    roles = "\n".join(
        f"- `{role}`: {count:,} training characters."
        for role, count in report["training_role_characters"].items()
    )
    return (
        "# Canonical Italian Corpus Acceptance Freeze v1\n\n"
        "## Result\n\n"
        f"Checkpoint 7C exhaustively verifies {report['stored_unit_count']:,} stored logical units "
        f"across {report['physical_file_count']:,} committed physical files. Every physical-file hash, "
        "logical byte range, UTF-8 boundary, character count, and logical hash passes.\n\n"
        f"The accepted inactive training view contains {report['training_record_count']:,} broader records, "
        f"{report['training_sonnet_count']:,} standard sonnets, and "
        f"{report['training_logical_character_count']:,} logical characters.\n\n"
        f"{roles}\n\n"
        "## Safety Boundary\n\n"
        f"All {report['protected_v6_sonnet_count']:,} protected V6 validation/test sonnets remain readable "
        "only through the explicit protected-audit iterator and are excluded from the default training iterator. "
        "All paths are repository-relative, no logical storage points into `data/local/`, and conditioned material "
        "is absent.\n\n"
        "The corpus remains inactive. This checkpoint creates no V7 split, performs no Minerva tokenization, "
        "assigns no mixture weight, starts no GPU work, and deletes no reusable cache.\n\n"
        "## Frozen Identities\n\n"
        f"- Logical identity SHA-256: `{report['logical_identity_sha256']}`\n"
        f"- Physical identity SHA-256: `{report['physical_identity_sha256']}`\n"
    )


def _read_csv(path: Path, required_fields: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = sorted(required_fields - fields)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {', '.join(missing)}")
        return list(reader)


def _unique_rows(
    rows: list[dict[str, str]], key: str, label: str
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        value = row[key]
        if not value or value in result:
            raise ValueError(f"duplicate or empty {key} in {label}: {value}")
        result[value] = row
    return result


def _parse_bool(value: str, label: str) -> bool:
    if value not in {"true", "false"}:
        raise ValueError(f"{label} must be true or false")
    return value == "true"


def _verify_logical_payload(unit: CanonicalTextUnit, payload: bytes) -> str:
    if len(payload) != unit.logical_byte_count:
        raise ValueError(f"logical byte-count mismatch: {unit.unit_id}")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"invalid UTF-8 slice boundary: {unit.unit_id}") from error
    if len(text) != unit.logical_character_count:
        raise ValueError(f"logical character-count mismatch: {unit.unit_id}")
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != unit.logical_sha256:
        raise ValueError(f"logical SHA-256 mismatch: {unit.unit_id}")
    return text


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root).as_posix()


def _replace_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
