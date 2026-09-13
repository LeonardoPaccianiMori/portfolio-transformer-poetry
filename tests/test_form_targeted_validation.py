import json
from pathlib import Path

from sonnet_analysis.form_targeted_validation import (
    GENERATION_VERSION,
    load_generation_records,
    output_name,
)
from sonnet_evaluation.sonnet_prosody_sealed import pair_records


VALID_LINE = "Nel mezzo del cammin di nostra vita"


def write_generation_dir(tmp_path: Path) -> Path:
    generation_dir = tmp_path / "generation"
    generation_dir.mkdir()
    outputs = []
    for system_id in ("baseline", "candidate"):
        path = generation_dir / output_name(system_id, "p1", 6200)
        payload = {
            "generation_version": GENERATION_VERSION,
            "analysis_role": "form_targeted_lora_pilot_validation",
            "system_id": system_id,
            "prompt": {"id": "p1", "opening_line": VALID_LINE},
            "opening_line": VALID_LINE,
            "seed": 6200,
            "text": "\n".join([VALID_LINE] * 14),
            "v7_test_accessed": False,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        outputs.append(
            {
                "path": path.name,
                "sha256": "0" * 64,
                "system_id": system_id,
                "prompt_id": "p1",
                "seed": 6200,
            }
        )
    complete = {
        "generation_version": GENERATION_VERSION,
        "completed_output_count": len(outputs),
        "outputs": outputs,
    }
    (generation_dir / "complete.json").write_text(
        json.dumps(complete), encoding="utf-8"
    )
    return generation_dir


def test_output_name_is_deterministic_and_system_specific():
    first = output_name("baseline", "p1", 6200)
    assert first == output_name("baseline", "p1", 6200)
    assert first != output_name("candidate", "p1", 6200)


def test_load_generation_records_reads_text_and_systems(tmp_path):
    generation_dir = write_generation_dir(tmp_path)
    records = load_generation_records(generation_dir)
    assert len(records) == 2
    assert {record["system_id"] for record in records} == {"baseline", "candidate"}
    assert all("\n" in record["text"] for record in records)


def test_pair_records_pairs_named_systems():
    scored = [
        {"system_id": "baseline", "prompt_id": "p1", "seed": 6200, "value": 1.0},
        {"system_id": "candidate", "prompt_id": "p1", "seed": 6200, "value": 2.0},
        {"system_id": "baseline", "prompt_id": "p2", "seed": 6200, "value": 3.0},
    ]
    pairs = pair_records(scored, systems=("baseline", "candidate"))
    assert len(pairs) == 1
    assert pairs[0][0]["value"] == 1.0
    assert pairs[0][1]["value"] == 2.0
