import json
import importlib.util
from pathlib import Path

import pytest

from sonnet_analysis.minerva_v7_final_evaluation import (
    EXPECTED_DOCUMENTS, EXPECTED_INDEX_SHA256, EXPECTED_SEEDS,
    EXPECTED_SHARD_SHA256, EXPECTED_TOKENIZER_SHA256,
    FINAL_PROTOCOL_VERSION, load_frozen_final_protocol,
)


ANALYZER_PATH = Path(__file__).parents[1] / "scripts/analyze_minerva_v7_dpo_validation.py"
ANALYZER_SPEC = importlib.util.spec_from_file_location("dpo_validation_analyzer", ANALYZER_PATH)
assert ANALYZER_SPEC and ANALYZER_SPEC.loader
ANALYZER = importlib.util.module_from_spec(ANALYZER_SPEC)
ANALYZER_SPEC.loader.exec_module(ANALYZER)


def _protocol():
    return {
        "protocol_version": FINAL_PROTOCOL_VERSION,
        "protocol_status": "frozen_before_first_test_access",
        "selected_final_system": "dpo", "comparator_system": "stage_3",
        "test_document_count": EXPECTED_DOCUMENTS, "test_token_count": 217364,
        "test_document_index_sha256": EXPECTED_INDEX_SHA256,
        "test_token_shard_sha256": EXPECTED_SHARD_SHA256,
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        "test_opening_selection": "all_documents_first_nonempty_line",
        "seeds": list(EXPECTED_SEEDS),
        "planned_output_count": EXPECTED_DOCUMENTS * 4,
        "systems": ["stage_3", "dpo"],
        "v7_test_access_authorized": True, "retuning_after_test_forbidden": True,
        "recipe": {
            "recipe_id": "no_labels_creative", "temperature": 0.85,
            "top_p": 0.95, "top_k": None, "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 4, "max_new_tokens": 512,
            "continuation_line_target": 13,
        },
        **{key: "a" * 64 for key in (
            "stage_3_state_identity_sha256", "dpo_adapter_sha256",
            "validation_analysis_sha256", "blinded_summary_sha256",
            "preservation_evaluation_sha256", "selection_record_sha256",
        )},
    }


def test_frozen_final_protocol_accepts_exact_bounded_grid(tmp_path):
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(_protocol()))
    assert load_frozen_final_protocol(path)["planned_output_count"] == 4976


def test_frozen_final_protocol_fails_closed_on_changed_seed(tmp_path):
    protocol = _protocol(); protocol["seeds"] = [1, 2]
    path = tmp_path / "protocol.json"; path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="seeds"):
        load_frozen_final_protocol(path)


def test_final_analysis_declares_test_access_and_no_retuning():
    script = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "scripts/analyze_minerva_v7_one_time_final.py"
    ).read_text(encoding="utf-8")
    assert '"v7_test_accessed": True' in script
    assert '"retuning_after_test_forbidden": True' in script


def test_final_blind_sample_uses_the_explicit_final_seed_set():
    rows = [
        {
            "system_id": system,
            "prompt_id": f"prompt-{prompt}",
            "seed": seed,
            "text": f"{system}-{prompt}-{seed}",
        }
        for system in ("stage_3", "dpo")
        for prompt in range(3)
        for seed in EXPECTED_SEEDS
    ]
    blind = ANALYZER._blind_sample(
        rows, count=2, seed=1, version="final", seeds=EXPECTED_SEEDS
    )
    assert blind["eligible_seeds"] == list(EXPECTED_SEEDS)
    assert len(blind["mapping"]) == 4
    assert {row["seed"] for row in blind["mapping"]} <= set(EXPECTED_SEEDS)
