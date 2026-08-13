import hashlib
import json
from pathlib import Path

import pytest

from sonnet_demo.server import (
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    SelectedSonnetGenerator,
    V7_DEPLOYMENT_MODE,
    parse_generation_request,
    validate_v7_demo_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]


def test_parse_generation_request_applies_defaults():
    request = parse_generation_request({"opening_line": "  Amor mi guida  "})
    assert request.opening_line == "Amor mi guida"
    assert request.temperature == DEFAULT_TEMPERATURE
    assert request.seed == DEFAULT_SEED


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"opening_line": ""}, "must not be empty"),
        ({"opening_line": "first\nsecond"}, "exactly one line"),
        ({"opening_line": "Amor", "temperature": 0.1}, "between"),
        ({"opening_line": "Amor", "seed": -1}, "between"),
        ({"opening_line": "Amor", "seed": True}, "integer"),
    ],
)
def test_parse_generation_request_rejects_invalid_values(payload, message):
    with pytest.raises(ValueError, match=message):
        parse_generation_request(payload)


def test_demo_assets_are_present_and_complete():
    for filename in ("index.html", "styles.css", "app.js"):
        text = (ROOT / "demo" / filename).read_text(encoding="utf-8")
        assert text.strip()
        assert "TODO" not in text
    html = (ROOT / "demo/index.html").read_text(encoding="utf-8")
    assert 'id="generationForm"' in html
    assert 'id="sonnetLines"' in html


def test_v7_demo_artifact_validation_requires_frozen_selection_and_adapter_hash(
    tmp_path,
):
    adapter = tmp_path / "adapter.pt"
    adapter.write_bytes(b"frozen adapter")
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({
        "status": "frozen_before_v7_test_access",
        "selected_final_system": "dpo",
        "retuning_after_test_forbidden": True,
        "dpo_adapter_sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    validated = validate_v7_demo_artifacts(
        adapter_path=adapter, selection_path=selection
    )
    assert validated["selected_final_system"] == "dpo"
    adapter.write_bytes(b"changed")
    with pytest.raises(ValueError, match="adapter hash mismatch"):
        validate_v7_demo_artifacts(adapter_path=adapter, selection_path=selection)


def test_v7_generator_metadata_labels_quantization_as_deployment_only():
    generator = SelectedSonnetGenerator(
        model=object(), tokenizer=object(), device="cpu", generation_mode="v7_dpo"
    )
    assert generator.deployment_mode == V7_DEPLOYMENT_MODE
    assert "approximation" in generator.deployment_mode
