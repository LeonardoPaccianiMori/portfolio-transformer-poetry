"""Verified private reference export for exact V7 sonnet-training memorization checks."""

from __future__ import annotations

import array
import hashlib
import json
import os
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from sonnet_analysis.minerva_v7_runtime import tokenizer_sha256


REFERENCE_VERSION = "minerva_7b_v7_sonnet_train_memorization_reference_v1"
EXPECTED_INDEX_SHA256 = "3544154a798565caee80e89bd307840b5de0a8684288e6f6b164ac5c5d332dfa"
EXPECTED_SHARD_SHA256 = "167da9677d44b6a50fa66afd68ede38be07bf3d951f20764c27692f801f7b341"
EXPECTED_TOKENIZER_SHA256 = "11fbe803977e9d6dc1a50e6bb088be5b550f5e26da2a82fbfd7b41a045853a8c"
EXPECTED_DOCUMENTS = 19_899
EXPECTED_TOKENS = 3_551_021
DEFAULT_NGRAM_SIZE = 40


def normalize_for_memorization(text: str) -> str:
    return " ".join(text.lower().split())


def character_ngram_set(text: str, ngram_size: int) -> set[str]:
    if ngram_size <= 0:
        raise ValueError("ngram_size must be greater than 0")
    return {
        text[index : index + ngram_size]
        for index in range(max(0, len(text) - ngram_size + 1))
    }


def _nearest_training_record(
    generated_text: str,
    training_records: Sequence[Mapping[str, str]],
    ngram_size: int,
) -> dict[str, Any]:
    generated_normalized = normalize_for_memorization(generated_text)
    generated_grams = character_ngram_set(generated_normalized, ngram_size)
    best: dict[str, Any] | None = None
    for record in training_records:
        reference_normalized = normalize_for_memorization(str(record["text"]))
        if not generated_grams:
            containment = 0.0
        else:
            reference_grams = character_ngram_set(reference_normalized, ngram_size)
            containment = len(generated_grams & reference_grams) / len(generated_grams)
        if containment == 0:
            continue
        longest = SequenceMatcher(
            None, generated_normalized, reference_normalized, autojunk=False
        ).find_longest_match(
            0, len(generated_normalized), 0, len(reference_normalized)
        ).size
        risk = "high" if containment >= 0.30 or longest >= 160 else (
            "medium" if containment >= 0.15 or longest >= 80 else "low"
        )
        row = {
            "nearest_poem_id": record["poem_id"],
            "nearest_title_or_first_line": record["title_or_first_line"],
            "nearest_author": record["author"],
            "nearest_clean_text_path": record["clean_text_path"],
            "ngram_containment": containment,
            "longest_common_substring_chars": longest,
            "longest_common_substring_is_exact": True,
            "longest_common_substring_upper_bound": None,
            "risk_level": risk,
        }
        if best is None or (containment, longest) > (
            best["ngram_containment"], best["longest_common_substring_chars"]
        ):
            best = row
    return best or {
        "nearest_poem_id": None,
        "nearest_title_or_first_line": None,
        "nearest_author": None,
        "nearest_clean_text_path": None,
        "ngram_containment": 0.0,
        "longest_common_substring_chars": None,
        "longest_common_substring_is_exact": False,
        "longest_common_substring_upper_bound": ngram_size - 1,
        "risk_level": "low",
    }


def build_sonnet_train_reference(
    *,
    encoded_report_path: Path,
    tokenizer: Any,
    output_dir: Path,
    repo_root: Path,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Decode only the hash-pinned V7 sonnet training pool into a private JSONL."""

    report = json.loads(encoded_report_path.read_text(encoding="utf-8"))
    pool = _resolve_train_pool(report)
    index_path = repo_root / pool["document_index"]["path"]
    shard_rows = pool["shards"]
    if len(shard_rows) != 1:
        raise ValueError("the frozen sonnets_train reference requires exactly one shard")
    shard_path = repo_root / shard_rows[0]["path"]
    _verify_authoritative_inputs(
        pool=pool, index_path=index_path, shard_path=shard_path, tokenizer=tokenizer
    )
    documents = _read_documents(index_path)
    tokens = _read_int32(shard_path, EXPECTED_TOKENS)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(eos_token_id, int):
        raise ValueError("verified tokenizer lacks an EOS token ID")

    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "sonnets_train.records.jsonl"
    temporary = records_path.with_suffix(records_path.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("wb") as handle:
        for number, document in enumerate(documents, start=1):
            start = _token_offset(document, "token_start")
            end = _token_offset(document, "token_end")
            token_ids = tokens[start:end]
            if not token_ids or int(token_ids[-1]) != eos_token_id:
                raise ValueError(f"sonnets_train document lacks its final EOS: {number - 1}")
            text = tokenizer.decode(
                list(token_ids[:-1]), skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            row = {
                "record_id": str(document["unit_id"]),
                "source_id": str(document["source_id"]),
                "document_index": int(document["document_index"]),
                "logical_sha256": str(document["logical_sha256"]),
                "decoded_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "text": text,
            }
            encoded = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
            handle.write(encoded)
            digest.update(encoded)
            if progress and (number == 1 or number % 1000 == 0 or number == len(documents)):
                progress(f"record={number}/{len(documents)} progress={100 * number / len(documents):.1f}%")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(records_path)
    manifest = {
        "reference_version": REFERENCE_VERSION,
        "source_pool_id": "sonnets_train",
        "source_split": "train",
        "record_count": len(documents),
        "token_count": EXPECTED_TOKENS,
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        "eos_token_id": eos_token_id,
        "encoded_report_sha256": _sha256(encoded_report_path),
        "document_index_sha256": EXPECTED_INDEX_SHA256,
        "token_shard_sha256": EXPECTED_SHARD_SHA256,
        "records_path": records_path.name,
        "records_bytes": records_path.stat().st_size,
        "records_sha256": digest.hexdigest(),
        "v7_test_accessed": False,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return load_verified_sonnet_train_reference(manifest_path)[1]


def load_verified_sonnet_train_reference(
    manifest_path: Path,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Load the private export only after verifying its frozen source lineage and hash."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "reference_version": REFERENCE_VERSION,
        "source_pool_id": "sonnets_train",
        "source_split": "train",
        "record_count": EXPECTED_DOCUMENTS,
        "token_count": EXPECTED_TOKENS,
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        "document_index_sha256": EXPECTED_INDEX_SHA256,
        "token_shard_sha256": EXPECTED_SHARD_SHA256,
        "v7_test_accessed": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"memorization reference lineage mismatch: {key}")
    relative = Path(str(manifest.get("records_path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("memorization reference contains an unsafe records path")
    records_path = manifest_path.parent / relative
    if (
        not records_path.is_file()
        or records_path.stat().st_size != int(manifest["records_bytes"])
        or _sha256(records_path) != manifest["records_sha256"]
    ):
        raise ValueError("memorization reference records hash mismatch")
    records = list(_iter_records(records_path))
    if len(records) != EXPECTED_DOCUMENTS:
        raise ValueError("memorization reference record count mismatch")
    return records, {**manifest, "manifest_sha256": _sha256(manifest_path)}


def score_texts_against_reference(
    texts: Sequence[str], records: Sequence[Mapping[str, str]],
    *, ngram_size: int = DEFAULT_NGRAM_SIZE,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Bound matching by scanning training records once for generated 40-grams."""

    generated_grams = [
        character_ngram_set(normalize_for_memorization(text), ngram_size)
        for text in texts
    ]
    union = set().union(*generated_grams) if generated_grams else set()
    gram_to_outputs: dict[str, list[int]] = {}
    for output_index, grams in enumerate(generated_grams):
        for gram in grams:
            gram_to_outputs.setdefault(gram, []).append(output_index)
    candidates: list[dict[str, Mapping[str, str]]] = [dict() for _ in texts]
    total = len(records)
    for number, record in enumerate(records, start=1):
        normalized = normalize_for_memorization(str(record["text"]))
        matched_outputs = set()
        for start in range(max(0, len(normalized) - ngram_size + 1)):
            gram = normalized[start : start + ngram_size]
            if gram in union:
                matched_outputs.update(gram_to_outputs[gram])
        for output_index in matched_outputs:
            candidates[output_index][str(record["record_id"])] = record
        if progress and (number == 1 or number % 1000 == 0 or number == total):
            progress(f"training_record={number}/{total} progress={100 * number / total:.1f}%")
    results = []
    for text, matched in zip(texts, candidates):
        if matched:
            compatible = [
                {
                    "poem_id": row["record_id"],
                    "title_or_first_line": row["source_id"],
                    "author": "",
                    "clean_text_path": "private_v7_sonnet_train_reference",
                    "text": row["text"],
                }
                for row in matched.values()
            ]
            results.append(_nearest_training_record(text, compatible, ngram_size))
        else:
            results.append(_nearest_training_record(text, [_no_match_record()], ngram_size))
    return results


def _resolve_train_pool(report: Mapping[str, Any]) -> Mapping[str, Any]:
    matches = [row for row in report.get("pools", []) if row.get("pool_id") == "sonnets_train"]
    if len(matches) != 1 or matches[0].get("split") != "train":
        raise ValueError("encoded report does not contain exactly one sonnets_train pool")
    return matches[0]


def _verify_authoritative_inputs(*, pool: Mapping[str, Any], index_path: Path, shard_path: Path, tokenizer: Any) -> None:
    if int(pool.get("documents", -1)) != EXPECTED_DOCUMENTS or int(pool.get("tokens", -1)) != EXPECTED_TOKENS:
        raise ValueError("sonnets_train pool count mismatch")
    if pool["document_index"].get("sha256") != EXPECTED_INDEX_SHA256 or pool["shards"][0].get("sha256") != EXPECTED_SHARD_SHA256:
        raise ValueError("sonnets_train report hashes differ from the frozen reference")
    if _sha256(index_path) != EXPECTED_INDEX_SHA256 or _sha256(shard_path) != EXPECTED_SHARD_SHA256:
        raise ValueError("sonnets_train local encoded input hash mismatch")
    if tokenizer_sha256(tokenizer) != EXPECTED_TOKENIZER_SHA256:
        raise ValueError("Minerva tokenizer fingerprint mismatch")


def _read_documents(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != EXPECTED_DOCUMENTS:
        raise ValueError("sonnets_train document index count mismatch")
    for index, row in enumerate(rows):
        if row.get("pool_id") != "sonnets_train" or row.get("split") != "train" or int(row.get("document_index", -1)) != index:
            raise ValueError("sonnets_train document index lineage mismatch")
    return rows


def _read_int32(path: Path, count: int) -> array.array[int]:
    if sys.byteorder != "little" or array.array("i").itemsize != 4:
        raise RuntimeError("the V7 int32 reference requires a little-endian 32-bit integer host")
    values = array.array("i")
    with path.open("rb") as handle:
        values.fromfile(handle, count)
        if handle.read(1):
            raise ValueError("sonnets_train shard has trailing bytes")
    if len(values) != count:
        raise ValueError("sonnets_train shard token count mismatch")
    return values


def _token_offset(row: Mapping[str, Any], key: str) -> int:
    position = row[key]
    if int(position["shard_index"]) != 0:
        raise ValueError("unexpected multi-shard sonnets_train position")
    return int(position["token_offset"])


def _iter_records(path: Path) -> Iterator[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            text = str(row["text"])
            if hashlib.sha256(text.encode("utf-8")).hexdigest() != row["decoded_text_sha256"]:
                raise ValueError("memorization record text hash mismatch")
            yield row


def _no_match_record() -> dict[str, str]:
    return {
        "poem_id": "no_match_sentinel", "title_or_first_line": "",
        "author": "", "clean_text_path": "", "text": "",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
