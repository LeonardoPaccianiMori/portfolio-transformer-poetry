#!/usr/bin/env python3
"""Build and verify the corrected V8 logical corpus view."""

from pathlib import Path

from sonnet_corpus.sonnet_v8_correction import (
    SonnetV8CorrectionConfig,
    build_sonnet_v8_corpus,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    report = build_sonnet_v8_corpus(
        SonnetV8CorrectionConfig(
            repo_root=root,
            canonical_corpus_dir=root / "data/processed/canonical_italian_corpora_v1",
            v7_manifest_path=root / "data/metadata/sonnets_expanded_v7_manifest.csv",
            correction_ledger_path=root / "data/metadata/sonnets_expanded_v8_corrections_v1.csv",
            alias_evidence_path=root / "data/metadata/sonnets_expanded_v8_alias_evidence_v1.csv",
            policy_path=root / "data/metadata/sonnets_expanded_v8_policy_v1.json",
            tei_cache_dir=root / "data/local/bibit/tei",
            output_dir=root / "data/processed/sonnets_expanded_v8",
            overlap_report_path=root / "reports/sonnets_expanded_v8_broader_heldout_overlap_v1.json",
            json_report_path=root / "reports/sonnets_expanded_v8_correction_v1.json",
            markdown_report_path=root / "reports/sonnets_expanded_v8_correction_v1.md",
        )
    )
    print(
        f"{report['status']}: {report['sonnet_count']} sonnets, "
        f"{report['broader_training_record_count']} broader records"
    )


if __name__ == "__main__":
    main()
