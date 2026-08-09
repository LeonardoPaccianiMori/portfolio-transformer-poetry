from pathlib import Path

import pytest

from sonnet_demo.server import (
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    parse_generation_request,
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
