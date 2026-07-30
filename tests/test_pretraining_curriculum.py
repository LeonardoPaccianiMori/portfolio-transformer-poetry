import hashlib
import json

import pytest

from sonnet_corpus.paisa_build import PAISA_DOCUMENT_SEPARATOR
from sonnet_corpus.pretraining_curriculum import (
    load_paisa_historical_curriculum_config,
    prepare_paisa_historical_curriculum,
    split_historical_source_text,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_document(handle, text):
    handle.write(f"{text}\n{PAISA_DOCUMENT_SEPARATOR}\n")


def _config_path(tmp_path):
    paisa_train = tmp_path / "local/paisa/train.txt"
    paisa_validation = tmp_path / "local/paisa/validation.txt"
    paisa_train.parent.mkdir(parents=True)
    with paisa_train.open("w", encoding="utf-8") as handle:
        for index in range(30):
            _write_document(handle, f"PAISA train document {index} con testo italiano.")
    with paisa_validation.open("w", encoding="utf-8") as handle:
        _write_document(handle, "PAISA_VALIDATION_ONLY")

    paisa_report_path = tmp_path / "reports/paisa.json"
    _write_json(
        paisa_report_path,
        {
            "source": {"release": {"sha256": "release-sha"}},
            "document_counts": {"train": 30, "validation": 1},
            "text_counts": {"train_characters": 1200, "validation_characters": 21},
            "local_artifacts": {
                "train_text_path": str(paisa_train),
                "validation_text_path": str(paisa_validation),
            },
        },
    )

    source_paths = []
    for source_id in ("historical_one", "historical_two"):
        source_path = tmp_path / f"historical/{source_id}.txt"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(
            "Inizio storico.\n" * 60 + f"{source_id.upper()}_VALIDATION_ONLY\n",
            encoding="utf-8",
        )
        source_paths.append((source_id, source_path))
    historical_report_path = tmp_path / "reports/historical.json"
    _write_json(
        historical_report_path,
        {
            "corpus_version": "pretraining_historical_italian_v2",
            "sources": [
                {"source_id": source_id, "source_path": str(source_path)}
                for source_id, source_path in source_paths
            ],
        },
    )

    config_path = tmp_path / "config.json"
    _write_json(
        config_path,
        {
            "curriculum_id": "test_rescue",
            "paisa_build_report_path": str(paisa_report_path),
            "historical_mixture_report_path": str(historical_report_path),
            "expected_paisa_build_report_sha256": _sha256(paisa_report_path),
            "expected_historical_mixture_report_sha256": _sha256(historical_report_path),
            "expected_paisa_release_sha256": "release-sha",
            "local_output_dir": str(tmp_path / "local/curriculum"),
            "report_path": str(tmp_path / "reports/curriculum.json"),
            "historical_source_validation_fraction": 0.1,
            "tokenizer": {
                "vocab_size": 16000,
                "special_tokens": ["<|endoftext|>"],
                "paisa_train_sample_characters": 10_000,
                "historical_train_sample_characters": 1_000,
                "sample_seed": "test-sample",
            },
            "stages": [
                {
                    "stage_id": "modern_italian_pretraining",
                    "dataset": "paisa_train",
                    "max_passes": 3,
                },
                {
                    "stage_id": "historical_italian_annealing",
                    "dataset": "historical_train",
                    "max_passes": 12,
                },
            ],
        },
    )
    return config_path


def test_prepare_curriculum_uses_only_training_text_for_tokenizer_sample(tmp_path):
    config = load_paisa_historical_curriculum_config(_config_path(tmp_path))

    report = prepare_paisa_historical_curriculum(config)

    sample_path = config.local_output_dir / "tokenizer_training_sample.txt"
    sample_text = sample_path.read_text(encoding="utf-8")
    historical_validation = (config.local_output_dir / "historical_validation.txt").read_text(
        encoding="utf-8"
    )
    assert "PAISA_VALIDATION_ONLY" not in sample_text
    assert "HISTORICAL_ONE_VALIDATION_ONLY" not in sample_text
    assert "HISTORICAL_TWO_VALIDATION_ONLY" not in sample_text
    assert "HISTORICAL_ONE_VALIDATION_ONLY" in historical_validation
    assert report["tokenizer"]["vocab_size"] == 16000
    assert report["tokenizer"]["training_policy"].endswith("no validation text")
    assert report["stages"][0]["max_passes"] == 3
    assert report["stages"][1]["max_passes"] == 12
    assert "PAISA_VALIDATION_ONLY" not in config.report_path.read_text(encoding="utf-8")


def test_prepare_curriculum_rejects_report_provenance_drift(tmp_path):
    config_path = _config_path(tmp_path)
    config = load_paisa_historical_curriculum_config(config_path)
    config.paisa_build_report_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="PAISÀ build report SHA-256"):
        prepare_paisa_historical_curriculum(config)


def test_historical_split_uses_a_final_newline_bounded_suffix():
    train_text, validation_text = split_historical_source_text(
        "prima riga\nseconda riga\nterza riga\n",
        validation_fraction=0.25,
    )

    assert train_text.endswith("seconda riga")
    assert validation_text == "terza riga"
