import json
from array import array
from pathlib import Path

import pytest
import torch

from sonnet_corpus.bpe import BytePairEncodingTokenizer
from sonnet_corpus.paisa_build import PAISA_DOCUMENT_SEPARATOR
from sonnet_corpus.paisa_historical_encoding import BoundedPretokenEncoder
from sonnet_corpus.paisa_historical_encoding import PaisaHistoricalEncodingConfig
from sonnet_corpus.paisa_historical_encoding import encode_paisa_historical_splits
from sonnet_corpus.paisa_historical_encoding import load_memory_mapped_token_ids


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_documents(path: Path, documents: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            f"{document.strip()}\n{PAISA_DOCUMENT_SEPARATOR}\n"
            for document in documents
        ),
        encoding="utf-8",
    )


def _write_fixture(
    tmp_path: Path,
) -> tuple[PaisaHistoricalEncodingConfig, dict[str, list[str]]]:
    documents = {
        "paisa_train": ["testo moderno", "altra pagina"],
        "paisa_validation": ["testo valido"],
        "historical_train": ["testo antico", "seconda opera"],
        "historical_validation": ["finale antico", "finale secondo"],
    }
    paths = {
        split_id: tmp_path / "inputs" / f"{split_id}.txt"
        for split_id in documents
    }
    for split_id, split_documents in documents.items():
        _write_documents(paths[split_id], split_documents)

    characters = sorted({
        character
        for split_documents in documents.values()
        for document in split_documents
        for character in document
    })
    tokenizer = BytePairEncodingTokenizer(
        token_to_id={
            PAISA_DOCUMENT_SEPARATOR: 0,
            **{
                character: index
                for index, character in enumerate(characters, start=1)
            },
        },
        merges=[],
        special_tokens=[PAISA_DOCUMENT_SEPARATOR],
    )
    tokenizer_path = tmp_path / "local/tokenizer.json"
    tokenizer.save(tokenizer_path)
    tokenizer_report_path = tmp_path / "reports/tokenizer_report.json"
    _write_json(
        tokenizer_report_path,
        {
            "curriculum_id": "test_rescue",
            "tokenizer": {"actual_vocab_size": tokenizer.vocab_size},
            "paisa_validation_sanitization": {"retained_documents": 1},
            "local_artifacts": {
                "tokenizer_path": str(tokenizer_path),
                "paisa_train_path": str(paths["paisa_train"]),
                "paisa_validation_tokenizable_path": str(
                    paths["paisa_validation"]
                ),
                "historical_train_path": str(paths["historical_train"]),
                "historical_validation_path": str(paths["historical_validation"]),
            },
        },
    )
    curriculum_report_path = tmp_path / "reports/curriculum_report.json"
    _write_json(
        curriculum_report_path,
        {
            "paisa": {"train_documents": 2},
            "historical": {"source_count": 2},
        },
    )
    config = PaisaHistoricalEncodingConfig(
        tokenizer_report_path=tokenizer_report_path,
        curriculum_report_path=curriculum_report_path,
        output_dir=tmp_path / "local/encoded",
        local_report_path=tmp_path / "local/encoded_report.json",
        public_report_path=tmp_path / "reports/encoded_report.json",
        progress_interval_documents=1,
        checkpoint_interval_documents=1,
        pretoken_cache_entries=4,
    )
    return config, documents


def test_streaming_encoder_writes_four_memory_mapped_splits(tmp_path: Path):
    config, documents = _write_fixture(tmp_path)

    report = encode_paisa_historical_splits(config)

    assert report["status"] == "complete"
    assert len(report["splits"]) == 4
    assert report["split_policy"].startswith("encode the four existing")
    tokenizer = BytePairEncodingTokenizer.load(
        Path(
            json.loads(config.tokenizer_report_path.read_text())["local_artifacts"][
                "tokenizer_path"
            ]
        )
    )
    for split in report["splits"]:
        token_ids = load_memory_mapped_token_ids(
            Path(split["output_path"]),
            token_count=split["tokens"],
        )
        expected_text = "".join(
            document + PAISA_DOCUMENT_SEPARATOR
            for document in documents[split["split_id"]]
        )
        assert token_ids.dtype == torch.uint16
        assert tokenizer.decode(token_ids.tolist()) == expected_text
        assert split["document_separator_token_count"] == len(
            documents[split["split_id"]]
        )


def test_streaming_encoder_resumes_and_truncates_uncheckpointed_bytes(
    tmp_path: Path,
):
    config, documents = _write_fixture(tmp_path)
    interrupted_config = PaisaHistoricalEncodingConfig(
        **{
            **config.__dict__,
            "max_documents_per_split_run": 1,
        }
    )

    incomplete = encode_paisa_historical_splits(interrupted_config)

    assert incomplete["status"] == "incomplete"
    part_path = config.output_dir / "paisa_train.uint16.bin.part"
    checkpoint_path = config.output_dir / ".paisa_train.checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    with part_path.open("ab") as handle:
        array("H", [999]).tofile(handle)
    assert part_path.stat().st_size > checkpoint["output_bytes"]

    complete = encode_paisa_historical_splits(config)

    split = complete["splits"][0]
    tokenizer_path = Path(
        json.loads(config.tokenizer_report_path.read_text())["local_artifacts"][
            "tokenizer_path"
        ]
    )
    tokenizer = BytePairEncodingTokenizer.load(tokenizer_path)
    token_ids = load_memory_mapped_token_ids(
        Path(split["output_path"]),
        token_count=split["tokens"],
    )
    assert tokenizer.decode(token_ids.tolist()) == "".join(
        document + PAISA_DOCUMENT_SEPARATOR
        for document in documents["paisa_train"]
    )
    assert not checkpoint_path.exists()
    assert not part_path.exists()


def test_streaming_encoder_rejects_unterminated_document(tmp_path: Path):
    config, _ = _write_fixture(tmp_path)
    tokenizer_report = json.loads(config.tokenizer_report_path.read_text())
    paisa_train_path = Path(
        tokenizer_report["local_artifacts"]["paisa_train_path"]
    )
    paisa_train_path.write_text("unterminated", encoding="utf-8")

    with pytest.raises(ValueError, match="unterminated document"):
        encode_paisa_historical_splits(config)


def test_memory_mapped_loader_rejects_wrong_token_count(tmp_path: Path):
    path = tmp_path / "tokens.bin"
    with path.open("wb") as handle:
        array("H", [1, 2, 3]).tofile(handle)

    with pytest.raises(ValueError, match="size mismatch"):
        load_memory_mapped_token_ids(path, token_count=4)


def test_bounded_pretoken_encoder_matches_regular_bpe_encoding():
    tokenizer = BytePairEncodingTokenizer(
        token_to_id={
            PAISA_DOCUMENT_SEPARATOR: 0,
            "a": 1,
            "m": 2,
            "o": 3,
            "r": 4,
            " ": 5,
            "am": 6,
            "or": 7,
            "amor": 8,
        },
        merges=[("a", "m"), ("o", "r"), ("am", "or")],
        special_tokens=[PAISA_DOCUMENT_SEPARATOR],
    )
    encoder = BoundedPretokenEncoder(tokenizer, max_cache_entries=2)
    text = "amor amor"

    assert encoder.encode(text) == tokenizer.encode(text)
    assert len(encoder.cache) <= 2
