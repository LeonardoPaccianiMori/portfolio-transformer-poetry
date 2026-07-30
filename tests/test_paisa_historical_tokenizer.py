import hashlib
import json
from pathlib import Path

import pytest

from sonnet_corpus.bpe import BytePairEncodingTokenizer
from sonnet_corpus.paisa_build import PAISA_DOCUMENT_SEPARATOR
from sonnet_corpus.paisa_historical_tokenizer import PaisaHistoricalTokenizerConfig
from sonnet_corpus.paisa_historical_tokenizer import train_paisa_historical_rescue_tokenizer


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_document(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{text}\n{PAISA_DOCUMENT_SEPARATOR}\n", encoding="utf-8")


def _write_test_curriculum(
    tmp_path: Path,
    *,
    validation_text: str = "validazione",
    historical_validation_text: str = "testo valido\n",
) -> Path:
    paisa_train_path = tmp_path / "local/paisa/train.txt"
    paisa_validation_path = tmp_path / "local/paisa/validation.txt"
    _write_document(
        paisa_train_path,
        "alfabeto italiano per il modello con z ripetuta z z z e parole diverse h cronica",
    )
    _write_document(paisa_validation_path, validation_text)
    historical_train_path = tmp_path / "local/curriculum/historical_train.txt"
    historical_validation_path = tmp_path / "local/curriculum/historical_validation.txt"
    historical_train_path.parent.mkdir(parents=True, exist_ok=True)
    historical_train_path.write_text(
        "testo storico italiano con parole diverse e lettere valide\n",
        encoding="utf-8",
    )
    historical_validation_path.write_text(historical_validation_text, encoding="utf-8")
    sample_path = tmp_path / "local/curriculum/tokenizer_training_sample.txt"
    sample_path.write_text(
        "amor amore amabile amicizia\n"
        "cronica cronache cronista\n"
        f"{PAISA_DOCUMENT_SEPARATOR}\n",
        encoding="utf-8",
    )

    paisa_report_path = tmp_path / "reports/paisa.json"
    _write_json(
        paisa_report_path,
        {
            "source": {"release": {"sha256": "release-sha"}},
            "local_artifacts": {
                "train_text_path": str(paisa_train_path),
                "validation_text_path": str(paisa_validation_path),
            },
        },
    )
    historical_report_path = tmp_path / "reports/historical.json"
    _write_json(historical_report_path, {"corpus_version": "pretraining_historical_italian_v2"})
    curriculum_report_path = tmp_path / "reports/curriculum_report.json"
    _write_json(
        curriculum_report_path,
        {
            "local_artifacts": {
                "historical_train_path": str(historical_train_path),
                "historical_validation_path": str(historical_validation_path),
                "tokenizer_training_sample_path": str(sample_path),
            }
        },
    )
    config_path = tmp_path / "configs/curriculum.json"
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
            "report_path": str(curriculum_report_path),
            "historical_source_validation_fraction": 0.01,
            "tokenizer": {
                "vocab_size": 64,
                "special_tokens": [PAISA_DOCUMENT_SEPARATOR],
                "paisa_train_sample_characters": 1,
                "historical_train_sample_characters": 1,
                "sample_seed": "test",
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


def test_rescue_tokenizer_is_train_only_and_resumable(tmp_path: Path):
    config_path = _write_test_curriculum(tmp_path)
    tokenizer_path = tmp_path / "local/curriculum/tokenizer.json"
    checkpoint_path = tmp_path / "local/curriculum/state.json"
    run_config = PaisaHistoricalTokenizerConfig(
        curriculum_config_path=config_path,
        tokenizer_path=tokenizer_path,
        training_checkpoint_path=checkpoint_path,
        max_merges_per_run=1,
        merge_progress_interval=1,
    )

    incomplete = train_paisa_historical_rescue_tokenizer(run_config)

    assert incomplete["status"] == "incomplete"
    assert checkpoint_path.is_file()
    completed = train_paisa_historical_rescue_tokenizer(
        PaisaHistoricalTokenizerConfig(
            curriculum_config_path=config_path,
            tokenizer_path=tokenizer_path,
            training_checkpoint_path=checkpoint_path,
            max_merges_per_run=100,
            merge_progress_interval=1,
        )
    )

    tokenizer = BytePairEncodingTokenizer.load(tokenizer_path)
    sample_path = tmp_path / "local/curriculum/tokenizer_training_sample.txt"
    sample = sample_path.read_text(encoding="utf-8")
    assert completed["status"] == "complete"
    assert not checkpoint_path.exists()
    assert tokenizer.decode(tokenizer.encode(sample)) == sample
    assert tokenizer.token_to_id[PAISA_DOCUMENT_SEPARATOR] == 0
    assert completed["sample"]["sha256"] == _sha256(sample_path)
    assert completed["character_coverage"]["uncovered_validation_character_count"] == 0
    assert completed["paisa_validation_sanitization"]["excluded_documents"] == 0
    assert ("z", "z") not in tokenizer.merges


def test_rescue_tokenizer_excludes_unrepresentable_paisa_validation_documents(tmp_path: Path):
    config_path = _write_test_curriculum(tmp_path, validation_text="o Ω")
    paisa_validation_path = tmp_path / "local/paisa/validation.txt"
    with paisa_validation_path.open("a", encoding="utf-8") as handle:
        handle.write(f"testo valido\n{PAISA_DOCUMENT_SEPARATOR}\n")

    report = train_paisa_historical_rescue_tokenizer(
        PaisaHistoricalTokenizerConfig(curriculum_config_path=config_path)
    )

    tokenizable_path = tmp_path / "local/curriculum/paisa_validation_tokenizable.txt"
    assert report["paisa_validation_sanitization"]["excluded_documents"] == 1
    assert report["paisa_validation_sanitization"]["excluded_codepoints"] == ["U+03A9"]
    assert "Ω" not in tokenizable_path.read_text(encoding="utf-8")


def test_rescue_tokenizer_rejects_historical_validation_character_absent_from_training(
    tmp_path: Path,
):
    config_path = _write_test_curriculum(
        tmp_path,
        historical_validation_text="testo Ω\n",
    )

    with pytest.raises(ValueError, match=r"U\+03A9"):
        train_paisa_historical_rescue_tokenizer(
            PaisaHistoricalTokenizerConfig(curriculum_config_path=config_path)
        )
