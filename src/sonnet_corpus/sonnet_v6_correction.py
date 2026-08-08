"""Build the deterministic V6 correction of the V5 sonnet corpus."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from sonnet_corpus.manifest import ManifestRow, write_manifest
from sonnet_corpus.sonnet_expansion_build import (
    normalize_for_duplicate_check,
    read_manifest_rows,
    split_counts,
)


V5_MANIFEST_SHA256 = "d71abe5bbc048392b7579702124f25bd6dedb400a47a0c171e3f4e6e0aae6275"
V6_DATASET_ID = "sonnets_expanded_v6"
V6_EXPECTED_COUNTS = {"train": 1481, "validation": 190, "test": 197}
V6_REMOVALS = {
    "cavalcanti_la_genealogia_dei_manoscritti": (
        "editorial apparatus page misclassified as a sonnet"
    ),
    "dante_cx_dante_quando_per_caso_s_abbandona": (
        "exact duplicate; retained stronger Cino attribution"
    ),
    "dante_cxii_cercando_di_trovar_minera_in_oro": (
        "exact duplicate; retained stronger Cino attribution"
    ),
    "dante_xcvii_dante_i_non_so_in_qual_albergo_soni": (
        "exact duplicate; retained stronger Cino attribution"
    ),
    "dante_xcviii_dante_i_ho_preso_l_abito_di_doglia": (
        "exact duplicate; retained stronger Cino attribution"
    ),
    "guittone_ben_si_conosce_lo_servente_e_vede": (
        "exact duplicate whose ID and title do not match the retained text"
    ),
    "guittone_de_vertù_de_scienzia_il_cui_podere": (
        "exact duplicate whose ID and title do not match the retained text"
    ),
}
V6_CANONICAL_DUPLICATE_RECORDS = (
    "cino_rime_dantecx_dante_quando_per_caso_s_abbandona",
    "cino_rime_dantecxii_cercando_di_trovar_minera_in_oro",
    "cino_rime_dantexcvii_dante_i_non_so_in_qual_albergo_soni",
    "cino_rime_dantexcviii_dante_i_ho_preso_l_abito_di_doglia",
    "guittone_non_per_meo_fallo_lasso_mi_convene",
    "guittone_tu_costante_e_sicuro_fondamento",
)


def build_sonnets_expanded_v6(
    *,
    repo_root: Path,
    source_manifest_path: Path,
    source_attribution_path: Path,
    validation_prompt_path: Path,
    final_test_prompt_path: Path,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Copy V5, remove seven fixed defects, and publish an isolated V6."""
    source_manifest = _resolve(repo_root, source_manifest_path)
    source_attribution = _resolve(repo_root, source_attribution_path)
    if _sha256(source_manifest) != V5_MANIFEST_SHA256:
        raise ValueError("V5 source manifest hash does not match the frozen input")

    source_rows = read_manifest_rows(source_manifest)
    selected_rows = select_v6_rows(source_rows)
    source_rows_by_id = {row.poem_id: row for row in source_rows}
    _validate_canonical_records(selected_rows)
    _validate_prompt_isolation(
        selected_rows,
        validation_prompt_path=_resolve(repo_root, validation_prompt_path),
        final_test_prompt_path=_resolve(repo_root, final_test_prompt_path),
    )

    output_root = repo_root / "data" / "processed" / V6_DATASET_ID
    output_manifest = repo_root / "data" / "metadata" / f"{V6_DATASET_ID}_manifest.csv"
    output_report = repo_root / "data" / "metadata" / f"{V6_DATASET_ID}_build_report.json"
    output_attribution = (
        repo_root / "data" / "metadata" / f"{V6_DATASET_ID}_attribution.md"
    )
    output_paths = (output_root, output_manifest, output_report, output_attribution)
    if any(path.exists() for path in output_paths):
        raise FileExistsError(f"versioned corpus already exists: {V6_DATASET_ID}")

    staging_root = repo_root / "data" / "interim" / f"{V6_DATASET_ID}_build"
    staging_manifest = repo_root / "data" / "interim" / f"{V6_DATASET_ID}_manifest.csv"
    staging_report = repo_root / "data" / "interim" / f"{V6_DATASET_ID}_build_report.json"
    staging_attribution = (
        repo_root / "data" / "interim" / f"{V6_DATASET_ID}_attribution.md"
    )
    staging_paths = (
        staging_root,
        staging_manifest,
        staging_report,
        staging_attribution,
    )
    _cleanup(staging_paths)

    try:
        poem_dir = staging_root / "poems"
        poem_dir.mkdir(parents=True)
        copied_rows: list[ManifestRow] = []
        _report(progress, f"copying {len(selected_rows)} corrected V6 poems")
        for index, row in enumerate(selected_rows, start=1):
            source_path = repo_root / row.clean_text_path
            destination = poem_dir / source_path.name
            shutil.copyfile(source_path, destination)
            audit_note = "V6 canonical record retained after exact-text audit."
            if row.poem_id not in V6_CANONICAL_DUPLICATE_RECORDS:
                audit_note = row.audit_notes
            copied_rows.append(
                replace(
                    row,
                    clean_text_path=str(destination.relative_to(repo_root)),
                    audit_notes=_append_note(row.audit_notes, audit_note),
                )
            )
            if index % 250 == 0 or index == len(selected_rows):
                _report(progress, f"copied poem {index}/{len(selected_rows)}")

        final_root = repo_root / "data" / "processed" / V6_DATASET_ID
        final_rows = [
            replace(
                row,
                clean_text_path=str(
                    final_root.relative_to(repo_root)
                    / Path(row.clean_text_path).relative_to(
                        staging_root.relative_to(repo_root)
                    )
                ),
            )
            for row in copied_rows
        ]
        write_manifest(final_rows, staging_manifest)
        _validate_unique_texts(final_rows, staging_root=staging_root, repo_root=repo_root)

        removed_rows = [source_rows_by_id[poem_id] for poem_id in V6_REMOVALS]
        report = {
            "dataset_id": V6_DATASET_ID,
            "correction_version": "sonnets_expanded_v6_correction_v1",
            "source_manifest_path": _portable(source_manifest, repo_root),
            "source_manifest_sha256": V5_MANIFEST_SHA256,
            "output_manifest_path": _portable(output_manifest, repo_root),
            "source_poem_count": len(source_rows),
            "removed_poem_count": len(removed_rows),
            "output_poem_count": len(final_rows),
            "split_counts": split_counts(final_rows),
            "removed_records": [
                {
                    "poem_id": row.poem_id,
                    "split": row.split_expanded_with_petrarch,
                    "reason": V6_REMOVALS[row.poem_id],
                }
                for row in removed_rows
            ],
            "canonical_duplicate_records": list(V6_CANONICAL_DUPLICATE_RECORDS),
            "validation_prompt_count": len(_load_prompt_ids(_resolve(repo_root, validation_prompt_path))),
            "final_test_prompt_count": len(_load_prompt_ids(_resolve(repo_root, final_test_prompt_path))),
            "exact_duplicate_group_count": 0,
        }
        if report["split_counts"] != V6_EXPECTED_COUNTS:
            raise ValueError(f"unexpected V6 split counts: {report['split_counts']}")
        staging_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _write_attribution(
            staging_attribution,
            source_attribution=source_attribution,
        )

        output_root.parent.mkdir(parents=True, exist_ok=True)
        staging_root.replace(output_root)
        staging_manifest.replace(output_manifest)
        report["output_manifest_sha256"] = _sha256(output_manifest)
        output_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        staging_report.unlink(missing_ok=True)
        staging_attribution.replace(output_attribution)
        _report(progress, f"wrote versioned corpus: {V6_DATASET_ID}")
        return report
    except Exception:
        _cleanup(output_paths)
        _cleanup(staging_paths)
        raise


def select_v6_rows(rows: list[ManifestRow]) -> list[ManifestRow]:
    """Apply only the seven approved V6 removals."""
    row_ids = {row.poem_id for row in rows}
    missing = sorted(set(V6_REMOVALS) - row_ids)
    if missing:
        raise ValueError("V5 manifest is missing fixed V6 removals: " + ", ".join(missing))
    selected = [row for row in rows if row.poem_id not in V6_REMOVALS]
    if len(rows) - len(selected) != len(V6_REMOVALS):
        raise ValueError("V6 correction removed an unexpected number of records")
    return selected


def _validate_canonical_records(rows: list[ManifestRow]) -> None:
    row_ids = {row.poem_id for row in rows}
    missing = sorted(set(V6_CANONICAL_DUPLICATE_RECORDS) - row_ids)
    if missing:
        raise ValueError("V6 canonical records are missing: " + ", ".join(missing))


def _validate_prompt_isolation(
    rows: list[ManifestRow], *, validation_prompt_path: Path, final_test_prompt_path: Path
) -> None:
    rows_by_id = {row.poem_id: row for row in rows}
    for prompt_path, expected_split in (
        (validation_prompt_path, "validation"),
        (final_test_prompt_path, "test"),
    ):
        for poem_id in _load_prompt_ids(prompt_path):
            row = rows_by_id.get(poem_id)
            if row is None:
                raise ValueError(f"V6 removed frozen prompt poem: {poem_id}")
            if row.split_expanded_with_petrarch != expected_split:
                raise ValueError(
                    f"V6 changed frozen prompt split for {poem_id}: "
                    f"{row.split_expanded_with_petrarch}"
                )


def _validate_unique_texts(
    rows: list[ManifestRow], *, staging_root: Path, repo_root: Path
) -> None:
    fingerprints: dict[str, str] = {}
    final_relative = Path("data") / "processed" / V6_DATASET_ID
    for row in rows:
        final_path = Path(row.clean_text_path)
        staged_path = staging_root / final_path.relative_to(final_relative)
        text = staged_path.read_text(encoding="utf-8")
        fingerprint = normalize_for_duplicate_check(text)
        previous = fingerprints.get(fingerprint)
        if previous is not None:
            raise ValueError(f"V6 exact duplicate remained: {previous}; {row.poem_id}")
        fingerprints[fingerprint] = row.poem_id


def _load_prompt_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"prompt config must contain a non-empty list: {path}")
    poem_ids = [row.get("poem_id") for row in payload if isinstance(row, dict)]
    if len(poem_ids) != len(payload) or any(not poem_id for poem_id in poem_ids):
        raise ValueError(f"prompt config contains an invalid poem_id: {path}")
    return poem_ids


def _write_attribution(path: Path, *, source_attribution: Path) -> None:
    inherited = source_attribution.read_text(encoding="utf-8")
    inherited_body = inherited.split("\n", 1)[1] if "\n" in inherited else inherited
    path.write_text(
        "# Sonnets Expanded V6 Attribution\n\n"
        "V6 contains no new source acquisition. It inherits V5 source texts and "
        "license obligations after removing one editorial apparatus page and six "
        "exact duplicate records. Source URLs, editions, revision metadata, and "
        "license notes remain unchanged in the V6 manifest.\n\n"
        "## Inherited V5 Source Records\n"
        f"{inherited_body}",
        encoding="utf-8",
    )


def _append_note(existing: str, note: str) -> str:
    existing = existing.strip()
    if not note or note in existing:
        return existing
    return f"{existing}; {note}" if existing else note


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable(path: Path, repo_root: Path) -> str:
    return str(path.resolve().relative_to(repo_root.resolve()))


def _resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _cleanup(paths: tuple[Path, ...]) -> None:
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)


def _report(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
