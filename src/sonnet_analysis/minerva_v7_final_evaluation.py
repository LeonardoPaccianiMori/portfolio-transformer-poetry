"""One-time, protocol-gated matched V7 final-test generation."""

from __future__ import annotations

import array
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from sonnet_analysis.minerva_v7_dpo_validation import generate_matched_validation
from sonnet_analysis.minerva_v7_runtime import tokenizer_sha256


FINAL_PROTOCOL_VERSION = "minerva_7b_v7_one_time_final_protocol_v1"
FINAL_GENERATION_VERSION = "minerva_7b_v7_one_time_final_generation_v1"
EXPECTED_DOCUMENTS = 1_244
EXPECTED_TOKENS = 217_364
EXPECTED_INDEX_SHA256 = "000b587a1e8b363c23a676ec10c1263fd913ccdf0def537f505ac85ccd717ce8"
EXPECTED_SHARD_SHA256 = "16f253f0ef85ed3bd2feda7f69ab07a82b60c019b1c1ad3e903196d4ab5ca5e1"
EXPECTED_TOKENIZER_SHA256 = "11fbe803977e9d6dc1a50e6bb088be5b550f5e26da2a82fbfd7b41a045853a8c"
EXPECTED_SEEDS = (6200, 6201)


def load_frozen_final_protocol(path: Path) -> dict[str, Any]:
    """Validate the pre-unsealing contract without reading test material."""

    protocol = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "protocol_version": FINAL_PROTOCOL_VERSION,
        "protocol_status": "frozen_before_first_test_access",
        "selected_final_system": "dpo",
        "comparator_system": "stage_3",
        "test_document_count": EXPECTED_DOCUMENTS,
        "test_token_count": EXPECTED_TOKENS,
        "test_document_index_sha256": EXPECTED_INDEX_SHA256,
        "test_token_shard_sha256": EXPECTED_SHARD_SHA256,
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        "test_opening_selection": "all_documents_first_nonempty_line",
        "seeds": list(EXPECTED_SEEDS),
        "planned_output_count": EXPECTED_DOCUMENTS * len(EXPECTED_SEEDS) * 2,
        "systems": ["stage_3", "dpo"],
        "v7_test_access_authorized": True,
        "retuning_after_test_forbidden": True,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"frozen final protocol mismatch: {key}")
    for key in (
        "stage_3_state_identity_sha256", "dpo_adapter_sha256",
        "validation_analysis_sha256", "blinded_summary_sha256",
        "preservation_evaluation_sha256", "selection_record_sha256",
    ):
        value = protocol.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"frozen final protocol lacks identity: {key}")
    recipe = protocol.get("recipe")
    if recipe != {
        "recipe_id": "no_labels_creative", "temperature": 0.85,
        "top_p": 0.95, "top_k": None, "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 4, "max_new_tokens": 512,
        "continuation_line_target": 13,
    }:
        raise ValueError("frozen final recipe mismatch")
    return protocol


def open_final_test_prompts(
    *, protocol: Mapping[str, Any], encoded_report_path: Path,
    tokenizer: Any, repo_root: Path,
) -> list[dict[str, Any]]:
    """First authorized test access: verify hashes, then decode every opening."""

    if protocol.get("v7_test_access_authorized") is not True:
        raise PermissionError("V7 final-test access is not authorized")
    report = json.loads(encoded_report_path.read_text(encoding="utf-8"))
    pools = [row for row in report.get("pools", []) if row.get("pool_id") == "sonnets_test"]
    if len(pools) != 1:
        raise ValueError("encoded report lacks the exact sonnets_test pool")
    pool = pools[0]
    if (
        pool.get("split") != "test"
        or int(pool.get("documents", -1)) != EXPECTED_DOCUMENTS
        or int(pool.get("tokens", -1)) != EXPECTED_TOKENS
        or len(pool.get("shards", [])) != 1
    ):
        raise ValueError("sonnets_test metadata differs from the frozen contract")
    index_path = repo_root / str(pool["document_index"]["path"])
    shard_path = repo_root / str(pool["shards"][0]["path"])
    if _sha256(index_path) != EXPECTED_INDEX_SHA256 or _sha256(shard_path) != EXPECTED_SHARD_SHA256:
        raise ValueError("sonnets_test input hash mismatch")
    if tokenizer_sha256(tokenizer) != EXPECTED_TOKENIZER_SHA256:
        raise ValueError("final-test tokenizer fingerprint mismatch")
    documents = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line]
    token_ids = _read_int32(shard_path, EXPECTED_TOKENS)
    if len(documents) != EXPECTED_DOCUMENTS:
        raise ValueError("sonnets_test document count mismatch")
    prompts = []
    for index, document in enumerate(documents):
        if (
            document.get("pool_id") != "sonnets_test"
            or document.get("split") != "test"
            or int(document.get("document_index", -1)) != index
        ):
            raise ValueError("sonnets_test document lineage mismatch")
        start, end = _offset(document, "token_start"), _offset(document, "token_end")
        span = token_ids[start:end]
        if not span or int(span[-1]) != tokenizer.eos_token_id:
            raise ValueError("final-test document lacks its frozen EOS")
        text = tokenizer.decode(
            list(span[:-1]), skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        opening = next((line.strip() for line in text.splitlines() if line.strip()), "")
        if not opening or "\n" in opening or "\r" in opening:
            raise ValueError("final-test document lacks one usable opening line")
        identity = hashlib.sha256(
            f"{FINAL_GENERATION_VERSION}|{document['unit_id']}|{document['logical_sha256']}".encode()
        ).hexdigest()
        prompts.append({
            "id": f"final_{identity[:16]}",
            "source_identity": str(document["unit_id"]),
            "source_split": "sonnets_test",
            "source_logical_sha256": str(document["logical_sha256"]),
            "author_key": str(document["author_key"]),
            "work_key": str(document["work_key"]),
            "period": str(document["epoch_key"]),
            "opening_line": opening,
        })
    return prompts


def _offset(row: Mapping[str, Any], key: str) -> int:
    value = row[key]
    if int(value["shard_index"]) != 0:
        raise ValueError("final-test pool unexpectedly crosses shards")
    return int(value["token_offset"])


def _read_int32(path: Path, count: int) -> array.array[int]:
    values: array.array[int] = array.array("i")
    with path.open("rb") as handle:
        values.fromfile(handle, count)
    if sys_byteorder_big_endian():
        values.byteswap()
    if len(values) != count:
        raise ValueError("final-test shard is truncated")
    return values


def sys_byteorder_big_endian() -> bool:
    import sys
    return sys.byteorder == "big"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
