"""Deterministic validation-only prompts for high-volume V7 generation."""

from __future__ import annotations

import array
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from sonnet_analysis.minerva_v7_runtime import tokenizer_sha256


PROMPT_VERSION = "minerva_7b_v7_exploratory_prompts_v1"
PROMPT_COUNT = 120
SELECTION_SEED = 8307
EXPECTED_INDEX_SHA256 = "51b0ee44e9d76e3fb5d4942fccedc9ba6a07e90778dc4340bdcedad3f7d0d192"
EXPECTED_SHARD_SHA256 = "4c9d9e071ac1a293e9831504d7e692ffbf7efeb587fb4f61d47eba6b175834f5"
EXPECTED_TOKENIZER_SHA256 = "11fbe803977e9d6dc1a50e6bb088be5b550f5e26da2a82fbfd7b41a045853a8c"
EXPECTED_DOCUMENTS = 1_247
EXPECTED_TOKENS = 219_470


def build_exploratory_prompt_manifest(
    *, encoded_report_path: Path, tokenizer: Any, repo_root: Path,
) -> dict[str, Any]:
    """Select 120 period/author/work-balanced openings from V7 validation only."""

    report = json.loads(encoded_report_path.read_text(encoding="utf-8"))
    pools = [row for row in report.get("pools", []) if row.get("pool_id") == "sonnets_validation"]
    if len(pools) != 1 or pools[0].get("split") != "validation":
        raise ValueError("encoded report lacks the exact sonnets_validation pool")
    pool = pools[0]
    if int(pool.get("documents", -1)) != EXPECTED_DOCUMENTS or int(pool.get("tokens", -1)) != EXPECTED_TOKENS:
        raise ValueError("sonnets_validation counts differ from the frozen contract")
    if len(pool.get("shards", [])) != 1:
        raise ValueError("sonnets_validation must contain exactly one frozen shard")
    index_path = repo_root / pool["document_index"]["path"]
    shard_path = repo_root / pool["shards"][0]["path"]
    if pool["document_index"].get("sha256") != EXPECTED_INDEX_SHA256 or pool["shards"][0].get("sha256") != EXPECTED_SHARD_SHA256:
        raise ValueError("sonnets_validation report hashes changed")
    if _sha256(index_path) != EXPECTED_INDEX_SHA256 or _sha256(shard_path) != EXPECTED_SHARD_SHA256:
        raise ValueError("sonnets_validation local input hash mismatch")
    if tokenizer_sha256(tokenizer) != EXPECTED_TOKENIZER_SHA256:
        raise ValueError("exploratory prompt tokenizer fingerprint mismatch")
    documents = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line]
    if len(documents) != EXPECTED_DOCUMENTS:
        raise ValueError("sonnets_validation document index count mismatch")
    token_ids = _read_int32(shard_path, EXPECTED_TOKENS)
    eos = getattr(tokenizer, "eos_token_id", None)
    candidates = []
    for index, document in enumerate(documents):
        if (
            document.get("pool_id") != "sonnets_validation"
            or document.get("split") != "validation"
            or int(document.get("document_index", -1)) != index
        ):
            raise ValueError("sonnets_validation document lineage mismatch")
        start = _offset(document, "token_start")
        end = _offset(document, "token_end")
        span = token_ids[start:end]
        if not span or int(span[-1]) != eos:
            raise ValueError("validation document lacks its frozen final EOS")
        text = tokenizer.decode(
            list(span[:-1]), skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        opening = next((line.strip() for line in text.splitlines() if line.strip()), "")
        if not opening or "\n" in opening or "\r" in opening:
            raise ValueError("validation document lacks one usable opening line")
        candidates.append({**document, "opening_line": opening})
    selected = _balanced_selection(candidates, PROMPT_COUNT)
    prompts = []
    for row in selected:
        identity = hashlib.sha256(
            f"{PROMPT_VERSION}|{row['unit_id']}|{row['logical_sha256']}".encode("utf-8")
        ).hexdigest()
        prompts.append(
            {
                "id": f"exploratory_{identity[:16]}",
                "source_identity": str(row["unit_id"]),
                "source_split": "sonnets_validation",
                "source_logical_sha256": str(row["logical_sha256"]),
                "author_key": str(row["author_key"]),
                "work_key": str(row["work_key"]),
                "period": str(row["epoch_key"]),
                "opening_line": str(row["opening_line"]),
            }
        )
    period_counts = Counter(row["period"] for row in prompts)
    author_counts = Counter(row["author_key"] for row in prompts)
    work_counts = Counter(row["work_key"] for row in prompts)
    return {
        "prompt_version": PROMPT_VERSION,
        "prompt_count": len(prompts),
        "selection_seed": SELECTION_SEED,
        "selection_policy": "waterfill_period_quota_then_minimize_global_author_and_work_reuse",
        "source_pool_id": "sonnets_validation",
        "source_split": "validation",
        "source_document_index_sha256": EXPECTED_INDEX_SHA256,
        "source_token_shard_sha256": EXPECTED_SHARD_SHA256,
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        "period_counts": dict(sorted(period_counts.items())),
        "unique_authors": len(author_counts),
        "maximum_prompts_per_author": max(author_counts.values()),
        "unique_works": len(work_counts),
        "maximum_prompts_per_work": max(work_counts.values()),
        "prompts": prompts,
        "v7_test_accessed": False,
    }


def validate_exploratory_prompt_manifest(
    path: Path, *, expected_sha256: str | None = None,
) -> dict[str, Any]:
    if expected_sha256 and _sha256(path) != expected_sha256:
        raise ValueError("exploratory prompt manifest hash mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "prompt_version": PROMPT_VERSION,
        "prompt_count": PROMPT_COUNT,
        "source_pool_id": "sonnets_validation",
        "source_split": "validation",
        "source_document_index_sha256": EXPECTED_INDEX_SHA256,
        "source_token_shard_sha256": EXPECTED_SHARD_SHA256,
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        "v7_test_accessed": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"exploratory prompt contract mismatch: {key}")
    prompts = manifest.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != PROMPT_COUNT:
        raise ValueError("exploratory prompt rows are incomplete")
    identities = set()
    for row in prompts:
        if row.get("source_split") != "sonnets_validation" or "test" in json.dumps(row).lower():
            raise ValueError("exploratory prompt contains non-validation or test material")
        identity = (row.get("id"), row.get("source_identity"), row.get("source_logical_sha256"))
        if identity in identities:
            raise ValueError("duplicate exploratory prompt identity")
        identities.add(identity)
        if not str(row.get("opening_line", "")).strip() or "\n" in row["opening_line"]:
            raise ValueError("invalid exploratory opening line")
    return manifest


def _balanced_selection(candidates: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    by_period: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_period[str(row["epoch_key"])].append(row)
    quotas = {period: 0 for period in by_period}
    while sum(quotas.values()) < count:
        available = [period for period in sorted(by_period) if quotas[period] < len(by_period[period])]
        if not available:
            raise ValueError("insufficient validation prompts")
        period = min(available, key=lambda value: (quotas[value], value))
        quotas[period] += 1
    author_counts: Counter[str] = Counter()
    work_counts: Counter[str] = Counter()
    selected = []
    for period in sorted(by_period):
        remaining = list(by_period[period])
        for _ in range(quotas[period]):
            row = min(
                remaining,
                key=lambda item: (
                    work_counts[str(item["work_key"])],
                    author_counts[str(item["author_key"])],
                    hashlib.sha256(
                        f"{SELECTION_SEED}|{item['unit_id']}".encode("utf-8")
                    ).hexdigest(),
                ),
            )
            remaining.remove(row)
            selected.append(row)
            author_counts[str(row["author_key"])] += 1
            work_counts[str(row["work_key"])] += 1
    return sorted(selected, key=lambda row: (str(row["epoch_key"]), str(row["unit_id"])))


def _read_int32(path: Path, count: int) -> array.array[int]:
    if sys.byteorder != "little" or array.array("i").itemsize != 4:
        raise RuntimeError("validation prompt build requires little-endian int32")
    values = array.array("i")
    with path.open("rb") as handle:
        values.fromfile(handle, count)
        if handle.read(1):
            raise ValueError("validation shard has trailing bytes")
    if len(values) != count:
        raise ValueError("validation shard token count mismatch")
    return values


def _offset(row: Mapping[str, Any], key: str) -> int:
    if int(row[key]["shard_index"]) != 0:
        raise ValueError("unexpected validation shard index")
    return int(row[key]["token_offset"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
