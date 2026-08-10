import csv
from pathlib import Path

from sonnet_corpus.gutenberg import FetchedGutenbergText
from sonnet_corpus.gutenberg_fulltext_probe import (
    GutenbergFullTextProbeConfig,
    fingerprint_text,
    measure_word_shingle_containment,
    run_gutenberg_fulltext_probe,
    select_authoritative_probe_candidates,
)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _gutenberg_raw(body: str) -> str:
    return (
        "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
        + body
        + "\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
    )


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


def test_authoritative_selection_excludes_conditioned_and_uses_final_role():
    inventory = [
        {"ebook_id": ebook_id, "preliminary_role": "date_and_role_review"}
        for ebook_id in ("1", "2", "3")
    ]
    authoritative = [
        {
            "ebook_id": "1",
            "title": "Selected",
            "resolution_pass": "pass_1b",
            "final_activation_class": "eligible_probe",
            "final_role": "historical_general_candidate",
            "final_decision": "eligible_historical_candidate",
        },
        {
            "ebook_id": "2",
            "title": "Conditioned",
            "resolution_pass": "pass_1b",
            "final_activation_class": "conditioned_probe",
            "final_role": "conditioned_romanesco_sonnet_candidate",
            "final_decision": "route_conditioned_romanesco_sonnets",
        },
        {
            "ebook_id": "3",
            "title": "Earlier pass",
            "resolution_pass": "pass_1a",
            "final_activation_class": "eligible_probe",
            "final_role": "nineteenth_century_bridge_candidate",
            "final_decision": "eligible_nineteenth_century_candidate",
        },
    ]

    selected, summary = select_authoritative_probe_candidates(
        inventory,
        authoritative_rows=authoritative,
        required_resolution_pass="pass_1b",
        required_activation_class="eligible_probe",
        expected_candidate_count=1,
        conditioned_activation_class="conditioned_probe",
        expected_conditioned_count=1,
    )

    assert [row["ebook_id"] for row in selected] == ["1"]
    assert selected[0]["final_role"] == "historical_general_candidate"
    assert summary["conditioned_count"] == 1
    assert summary["conditioned_records"][0]["ebook_id"] == "2"


def test_pass1b_probe_reuses_cache_and_compares_complete_prior_pool(tmp_path):
    body = "La storia italiana narra la vita della città con parole antiche. " * 80
    other = "Il pensiero umano considera ogni ragione e ogni nuova memoria. " * 80
    inventory = tmp_path / "inventory.csv"
    _write_csv(
        inventory,
        [
            {
                "ebook_id": "10",
                "title": "Nuovo libro",
                "authors": "Autore",
                "preliminary_role": "date_and_role_review",
                "period_bucket": "unknown",
                "inventory_status": "review_work_publication_date",
                "landing_page_url": "https://example.test/10",
                "plain_text_url": "https://example.test/10.txt",
                "possible_existing_work_matches": "",
                "intra_gutenberg_duplicate_ids": "5",
            }
        ],
    )
    authoritative = tmp_path / "authoritative.csv"
    _write_csv(
        authoritative,
        [
            {
                "ebook_id": "10",
                "title": "Nuovo libro",
                "resolution_pass": "pass_1b",
                "final_period_bucket": "origins_through_1800",
                "final_role": "historical_general_candidate",
                "final_decision": "eligible_historical_candidate",
                "final_resolution_status": "pass_1b_authoritative_resolved",
                "final_activation_class": "eligible_probe",
            },
            {
                "ebook_id": "11",
                "title": "Testo condizionato",
                "resolution_pass": "pass_1b",
                "final_period_bucket": "conditioned",
                "final_role": "conditioned_bolognese_prose_and_drama_candidate",
                "final_decision": "route_conditioned_bolognese_prose_and_drama",
                "final_resolution_status": "pass_1b_language_variety_routed",
                "final_activation_class": "conditioned_probe",
            },
        ],
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "pg10.txt").write_text(_gutenberg_raw(body), encoding="utf-8")
    prior_cache = tmp_path / "prior-cache"
    prior_cache.mkdir()
    (prior_cache / "pg5.txt").write_text(_gutenberg_raw(body), encoding="utf-8")
    prior_probe = tmp_path / "prior.csv"
    _write_csv(prior_probe, [{"ebook_id": "5"}])

    bibit_text = tmp_path / "bibit.txt"
    bibit_text.write_text(other, encoding="utf-8")
    bibit = tmp_path / "bibit.csv"
    _write_csv(
        bibit,
        [
            {
                "object_id": "bibit-one",
                "artifact_status": "text_materialized",
                "shard_path": bibit_text.name,
                "byte_start": "0",
                "byte_end": str(bibit_text.stat().st_size),
            }
        ],
    )
    current_text = tmp_path / "current.txt"
    current_text.write_text(other + " altra conclusione", encoding="utf-8")
    broader = tmp_path / "broader.csv"
    _write_csv(
        broader,
        [{"source_id": "current-one", "expected_clean_text_path": current_text.name}],
    )
    sonnet_path = tmp_path / "heldout.txt"
    sonnet_path.write_text("Verso distinto della prova\n" * 14, encoding="utf-8")
    sonnets = tmp_path / "sonnets.csv"
    _write_csv(
        sonnets,
        [
            {
                "poem_id": "heldout",
                "clean_text_path": sonnet_path.name,
                "split_expanded_with_petrarch": "test",
            }
        ],
    )
    config = GutenbergFullTextProbeConfig(
        repo_root=tmp_path,
        inventory_csv_path=inventory,
        cache_dir=cache,
        output_csv_path=tmp_path / "probe.csv",
        json_report_path=tmp_path / "probe.json",
        markdown_report_path=tmp_path / "probe.md",
        bibit_record_manifest_path=bibit,
        broader_sources_manifest_path=broader,
        sonnet_manifest_path=sonnets,
        authoritative_resolution_csv_path=authoritative,
        required_resolution_pass="pass_1b",
        required_activation_class="eligible_probe",
        expected_candidate_count=1,
        conditioned_activation_class="conditioned_probe",
        expected_conditioned_count=1,
        prior_gutenberg_probe_csv_path=prior_probe,
        prior_gutenberg_cache_dir=prior_cache,
        expected_prior_gutenberg_count=1,
        min_cleaned_characters=100,
        anchor_mask=7,
    )

    def unexpected_fetch(*args, **kwargs):
        raise AssertionError("cached pass-1B text must not be downloaded")

    report = run_gutenberg_fulltext_probe(config, fetch_text=unexpected_fetch)

    assert report["candidate_count"] == 1
    assert report["selection"]["conditioned_count"] == 1
    assert report["prior_gutenberg_reference_count"] == 1
    assert len(report["prior_gutenberg_duplicate_pairs"]) == 1
    assert report["cross_corpus_reference_count"] == 2
    with config.output_csv_path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["cache_status"] == "hit"
    assert row["final_role"] == "historical_general_candidate"
    assert "prior_gutenberg:pg5" in row["prior_gutenberg_overlap_metrics"]
    assert row["probe_decision"] == "resolve_cross_pool_gutenberg_canonical_edition"


def test_manual_review_ledger_requires_resolution_and_rationale(tmp_path):
    inventory = tmp_path / "inventory.csv"
    _write_csv(
        inventory,
        [
            {
                "ebook_id": "20",
                "title": "Libro in dialetto romanesco",
                "authors": "Autore",
                "preliminary_role": "historical_general_candidate",
                "period_bucket": "origins_through_1800",
                "inventory_status": "audit_then_deduplicate",
                "landing_page_url": "https://example.test/20",
                "plain_text_url": "https://example.test/20.txt",
                "possible_existing_work_matches": "",
                "intra_gutenberg_duplicate_ids": "",
            }
        ],
    )
    body = "La storia italiana narra la vita della città con parole antiche. " * 80
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "pg20.txt").write_text(_gutenberg_raw(body), encoding="utf-8")
    reference = tmp_path / "reference.txt"
    reference.write_text("Pensiero e memoria del mondo umano. " * 80, encoding="utf-8")
    bibit = tmp_path / "bibit.csv"
    _write_csv(
        bibit,
        [
            {
                "object_id": "bibit-one",
                "artifact_status": "text_materialized",
                "shard_path": reference.name,
                "byte_start": "0",
                "byte_end": str(reference.stat().st_size),
            }
        ],
    )
    broader = tmp_path / "broader.csv"
    _write_csv(
        broader,
        [{"source_id": "current-one", "expected_clean_text_path": reference.name}],
    )
    sonnet = tmp_path / "sonnet.txt"
    sonnet.write_text("Verso separato dalla fonte\n" * 14, encoding="utf-8")
    sonnets = tmp_path / "sonnets.csv"
    _write_csv(
        sonnets,
        [
            {
                "poem_id": "heldout",
                "clean_text_path": sonnet.name,
                "split_expanded_with_petrarch": "validation",
            }
        ],
    )
    review_path = tmp_path / "review.csv"
    base_config = dict(
        repo_root=tmp_path,
        inventory_csv_path=inventory,
        cache_dir=cache,
        output_csv_path=tmp_path / "probe.csv",
        json_report_path=tmp_path / "probe.json",
        markdown_report_path=tmp_path / "probe.md",
        bibit_record_manifest_path=bibit,
        broader_sources_manifest_path=broader,
        sonnet_manifest_path=sonnets,
        review_decisions_csv_path=review_path,
        min_cleaned_characters=100,
        anchor_mask=7,
    )

    detection_report = run_gutenberg_fulltext_probe(
        GutenbergFullTextProbeConfig(
            **base_config,
            require_review_resolutions=False,
        )
    )
    assert detection_report["manual_review_anomaly_count"] == 1
    assert detection_report["manual_review_unresolved_count"] == 1

    _write_csv(
        review_path,
        [
            {
                "ebook_id": "20",
                "title": "Libro in dialetto romanesco",
                "quality_review_flags": "",
                "language_variety_flags": "review_language_variety_marker",
                "manual_review_resolution": "retain_conditioned_review",
                "manual_review_rationale": "The title explicitly requires a conditioned language route.",
            }
        ],
    )
    final_report = run_gutenberg_fulltext_probe(
        GutenbergFullTextProbeConfig(**base_config)
    )
    assert final_report["manual_review_unresolved_count"] == 0
    with (tmp_path / "probe.csv").open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["manual_review_resolution"] == "retain_conditioned_review"
