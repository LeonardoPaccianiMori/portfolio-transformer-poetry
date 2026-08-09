import csv
from pathlib import Path

from sonnet_corpus.gutenberg import FetchedGutenbergText
from sonnet_corpus.gutenberg_fulltext_probe import (
    GutenbergFullTextProbeConfig,
    fingerprint_text,
    measure_word_shingle_containment,
    run_gutenberg_fulltext_probe,
)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_fingerprint_and_containment_normalize_accents_and_punctuation():
    left = "L'umanità vive nella città, e parla della sua storia ogni giorno. " * 20
    right = "L umanita vive nella citta e parla della sua storia ogni giorno " * 20
    extended = right + ("ma il nuovo capitolo continua con altre parole. " * 20)

    left_fingerprint, _ = fingerprint_text(left, anchor_mask=7)
    right_fingerprint, _ = fingerprint_text(right, anchor_mask=7)
    containment = measure_word_shingle_containment(left, extended)

    assert left_fingerprint.normalized_word_sha256 == right_fingerprint.normalized_word_sha256
    assert containment["containment"] == 1.0
    assert containment["left_containment"] == 1.0
    assert containment["right_containment"] < 1.0
    assert containment["matching_shingles"] == containment["denominator"]


def test_run_fulltext_probe_measures_duplicates_and_heldout_leakage(tmp_path):
    inventory = tmp_path / "inventory.csv"
    _write_csv(
        inventory,
        [
            {
                "ebook_id": ebook_id,
                "title": f"Libro {ebook_id}",
                "authors": "Autore",
                "preliminary_role": "historical_general_candidate",
                "period_bucket": "origins_through_1800",
                "inventory_status": "audit_then_deduplicate",
                "landing_page_url": f"https://example.test/{ebook_id}",
                "plain_text_url": f"https://example.test/{ebook_id}.txt",
                "download_count": "10",
                "possible_existing_work_matches": (
                    "bibit:bibit000001" if ebook_id == "1" else ""
                ),
                "intra_gutenberg_duplicate_ids": "2" if ebook_id == "1" else "",
            }
            for ebook_id in ("1", "2", "3", "4")
        ],
    )
    sonnet = (
        "Nel cuore della sera parla amore\n"
        "e la città risponde alla sua voce\n"
        "mentre la vita passa e non si tace\n"
        "e ogni pensiero torna nel mio cuore\n"
        "La storia antica vive nel chiarore\n"
        "del giorno che nel cielo nasce in pace\n"
        "e il tempo con la luce si compiace\n"
        "quando la mente cerca il suo valore\n"
        "Così la lingua narra la memoria\n"
        "di chi cammina ancora nella via\n"
        "e trova nelle parole la ragione\n"
        "poi custodisce intera questa storia\n"
        "perché nel canto resti compagnia\n"
        "e viva in ogni tempo la canzone\n"
    )
    shared_body = sonnet + ("Il testo italiano che parla della vita e della storia. " * 30)
    unique_body = " ".join(
        "Nel capitolo "
        + chr(97 + index // 26)
        + chr(97 + index % 26)
        + " la cronaca narra il viaggio della città con parole nuove."
        for index in range(200)
    )
    other_body = "La filosofia considera il pensiero e discute ogni ragione umana. " * 40

    shard = tmp_path / "bibit.txt"
    shard.write_text(shared_body, encoding="utf-8")
    bibit_manifest = tmp_path / "bibit.csv"
    _write_csv(
        bibit_manifest,
        [
            {
                "object_id": "bibit000001",
                "artifact_status": "text_materialized",
                "shard_path": str(shard.relative_to(tmp_path)),
                "byte_start": "0",
                "byte_end": str(shard.stat().st_size),
            }
        ],
    )
    current_text = tmp_path / "current.txt"
    current_text.write_text(shared_body, encoding="utf-8")
    broader_manifest = tmp_path / "broader.csv"
    _write_csv(
        broader_manifest,
        [
            {
                "source_id": "existing_work",
                "expected_clean_text_path": str(current_text.relative_to(tmp_path)),
            }
        ],
    )
    sonnet_path = tmp_path / "heldout.txt"
    sonnet_path.write_text(sonnet, encoding="utf-8")
    sonnet_manifest = tmp_path / "sonnets.csv"
    _write_csv(
        sonnet_manifest,
        [
            {
                "poem_id": "heldout_one",
                "clean_text_path": str(sonnet_path.relative_to(tmp_path)),
                "split_expanded_with_petrarch": "test",
            }
        ],
    )
    config = GutenbergFullTextProbeConfig(
        repo_root=tmp_path,
        inventory_csv_path=inventory,
        cache_dir=tmp_path / "cache",
        output_csv_path=tmp_path / "probe.csv",
        json_report_path=tmp_path / "probe.json",
        markdown_report_path=tmp_path / "probe.md",
        bibit_record_manifest_path=bibit_manifest,
        broader_sources_manifest_path=broader_manifest,
        sonnet_manifest_path=sonnet_manifest,
        request_delay_seconds=0,
        min_cleaned_characters=100,
        anchor_mask=7,
    )

    def fetch(ebook_id, **kwargs):
        if ebook_id in {"1", "2"}:
            body = shared_body
        elif ebook_id == "3":
            body = sonnet + unique_body
        else:
            body = other_body
        return FetchedGutenbergText(
            ebook_id=ebook_id,
            url=f"https://example.test/{ebook_id}.txt",
            text=(
                "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
                + body
                + "\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
            ),
        )

    report = run_gutenberg_fulltext_probe(config, fetch_text=fetch)

    assert report["candidate_count"] == 4
    assert report["probe_status_counts"] == {"quality_pass": 4}
    assert report["heldout_sonnet_reference_count"] == 1
    assert report["intra_gutenberg_exact_duplicate_groups"] == [["1", "2"]]
    assert any(pair["reference_id"] == "bibit:bibit000001" for pair in report["cross_corpus_duplicate_pairs"])
    with config.output_csv_path.open(encoding="utf-8", newline="") as handle:
        rows = {row["ebook_id"]: row for row in csv.DictReader(handle)}
    assert rows["1"]["intra_gutenberg_exact_duplicate_ids"] == "2"
    assert rows["2"]["intra_gutenberg_exact_duplicate_ids"] == "1"
    assert "heldout_one|containment=1.000000" in rows["1"]["heldout_sonnet_overlap_metrics"]
    assert rows["1"]["probe_decision"] == "exclude_cross_corpus_duplicate_candidate"
    assert rows["3"]["probe_decision"] == "quarantine_heldout_sonnet_segment_before_activation"
    assert rows["4"]["probe_decision"] == "quality_pass_pending_editorial_activation_review"
    assert rows["4"]["manual_review_resolution"] == ""
    assert report["policy"]["activation_authorized"] is False
