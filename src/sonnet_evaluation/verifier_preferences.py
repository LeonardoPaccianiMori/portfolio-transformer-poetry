"""Verifier-labelled form preferences from the existing DPO candidates.

The builder scores the frozen candidate set with the reviewed prosody checker,
removes degenerate text, and pairs the strongest and weakest candidate of each
opening. It produces a frozen preference dataset with its own provenance
version; it does not reuse the AI-judge vote schema.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from sonnet_evaluation.sonnet_prosody_sealed import score_output

PREFERENCE_VERSION = "minerva_7b_v7_verifier_form_preferences_v1"
SCOPE = "exploratory_verifier_labelled_form_dpo"
PAIR_TYPE = "verifier_form_contrast"
DEGENERACY_ALPHA_RATIO = 0.65
DEGENERACY_MEAN_WORD_LENGTH = 3.3
DEFAULT_MIN_GAP = 2.0
DEFAULT_MIN_CHOSEN_SCORE = 8.0
WORD_PATTERN = re.compile(r"[A-Za-zÀ-ÿ']+")


def load_candidate_records(candidates_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(candidates_dir.glob("candidate_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("source_split") != "sonnets_train":
            raise ValueError(f"candidate {path.name} is not from the train split")
        if payload.get("v7_test_accessed") is not False:
            raise ValueError(f"candidate {path.name} reports test access")
        text = str(payload.get("text", ""))
        if not text:
            raise ValueError(f"candidate {path.name} has no text")
        records.append(
            {
                "candidate_id": str(payload["candidate_id"]),
                "prompt_id": str(payload["prompt_id"]),
                "opening_line": str(payload["opening_line"]),
                "recipe_id": str(payload["recipe_id"]),
                "seed": int(payload["seed"]),
                "text": text,
            }
        )
    if not records:
        raise ValueError("candidate directory is empty")
    if len({row["candidate_id"] for row in records}) != len(records):
        raise ValueError("candidate ids are duplicated")
    return records


def text_degeneracy(text: str) -> dict[str, Any]:
    alphabetic = sum(character.isalpha() for character in text)
    alpha_ratio = alphabetic / max(1, len(text))
    words = WORD_PATTERN.findall(text)
    mean_word_length = (
        sum(len(word) for word in words) / len(words) if words else 0.0
    )
    return {
        "alpha_ratio": alpha_ratio,
        "mean_word_length": mean_word_length,
        "degenerate": (
            alpha_ratio < DEGENERACY_ALPHA_RATIO
            or mean_word_length < DEGENERACY_MEAN_WORD_LENGTH
        ),
    }


def form_score(metrics: Mapping[str, Any]) -> float:
    return (
        float(metrics["hendecasyllable_lines"])
        + 2.0 * float(metrics["quatrain_ok"])
        + 2.0 * float(metrics["tercet_ok"])
        + 2.0 * float(metrics["all_lines_valid"])
        + float(metrics["rhyme_score"])
    )


def score_candidates(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    scored = []
    for record in records:
        metrics = score_output(str(record["text"]))
        degeneracy = text_degeneracy(str(record["text"]))
        scored.append(
            {
                **record,
                "metrics": metrics,
                "degeneracy": degeneracy,
                "form_score": None if degeneracy["degenerate"] else form_score(metrics),
            }
        )
    return scored


def build_verifier_pairs(
    scored: Iterable[Mapping[str, Any]],
    *,
    min_gap: float = DEFAULT_MIN_GAP,
    min_chosen_score: float = DEFAULT_MIN_CHOSEN_SCORE,
) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in scored:
        if row["form_score"] is not None:
            groups[str(row["prompt_id"])].append(row)
    pairs = []
    for prompt_id in sorted(groups):
        rows = sorted(groups[prompt_id], key=lambda row: str(row["candidate_id"]))
        if len(rows) < 2:
            continue
        ranked = sorted(
            rows,
            key=lambda row: (float(row["form_score"]), str(row["candidate_id"])),
        )
        rejected, chosen = ranked[0], ranked[-1]
        if float(chosen["form_score"]) < min_chosen_score:
            continue
        if float(chosen["form_score"]) - float(rejected["form_score"]) < min_gap:
            continue
        pair_key = (
            f"{prompt_id}|{chosen['candidate_id']}|{rejected['candidate_id']}"
        )
        pairs.append(
            {
                "pair_id": "pair_"
                + hashlib.sha256(pair_key.encode("utf-8")).hexdigest()[:16],
                "pair_type": PAIR_TYPE,
                "prompt_id": prompt_id,
                "opening_line": str(chosen["opening_line"]),
                "chosen": str(chosen["text"]),
                "rejected": str(rejected["text"]),
                "chosen_candidate_id": str(chosen["candidate_id"]),
                "rejected_candidate_id": str(rejected["candidate_id"]),
                "chosen_recipe_id": str(chosen["recipe_id"]),
                "rejected_recipe_id": str(rejected["recipe_id"]),
                "chosen_score": float(chosen["form_score"]),
                "rejected_score": float(rejected["form_score"]),
            }
        )
    if not pairs:
        raise ValueError("no verifier preference pairs were built")
    if len({pair["pair_id"] for pair in pairs}) != len(pairs):
        raise ValueError("verifier pair ids are duplicated")
    return pairs


def write_verifier_preferences(
    path: Path,
    *,
    scored: list[Mapping[str, Any]],
    pairs: list[Mapping[str, Any]],
    candidates_dir: Path,
    min_gap: float,
    min_chosen_score: float,
) -> dict[str, Any]:
    degenerate = sum(1 for row in scored if row["degeneracy"]["degenerate"])
    payload = {
        "preference_version": PREFERENCE_VERSION,
        "scope": SCOPE,
        "source_split": "sonnets_train",
        "candidate_dir": str(candidates_dir),
        "candidate_count": len(scored),
        "degenerate_candidate_count": degenerate,
        "pair_count": len(pairs),
        "pair_type": PAIR_TYPE,
        "form_score_definition": (
            "hendecasyllable_lines + 2*quatrain_ok + 2*tercet_ok "
            "+ 2*all_lines_valid + rhyme_score"
        ),
        "degeneracy_rule": (
            f"alpha_ratio < {DEGENERACY_ALPHA_RATIO} or mean_word_length < "
            f"{DEGENERACY_MEAN_WORD_LENGTH}"
        ),
        "min_gap": min_gap,
        "min_chosen_score": min_chosen_score,
        "v7_test_accessed": False,
        "pairs": list(pairs),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def load_verifier_preferences(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("preference_version") != PREFERENCE_VERSION
        or payload.get("scope") != SCOPE
        or payload.get("source_split") != "sonnets_train"
        or payload.get("pair_type") != PAIR_TYPE
        or payload.get("v7_test_accessed") is not False
    ):
        raise ValueError("verifier preference dataset lineage mismatch")
    pairs = payload.get("pairs", [])
    if not pairs or len({pair["pair_id"] for pair in pairs}) != len(pairs):
        raise ValueError("verifier preference pairs are empty or duplicated")
    return payload
