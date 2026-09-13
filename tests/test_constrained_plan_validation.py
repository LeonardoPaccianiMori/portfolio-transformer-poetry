import pytest

from sonnet_analysis.constrained_plan_validation import (
    plan_from_anchored,
    repair_endings,
)
from sonnet_evaluation.sonnet_prosody import rhyme_key


def make_lexicon() -> dict:
    keys = [
        "ita",
        "ore",
        "are",
        "ente",
        "ando",
        "ura",
        "ezza",
        "aggio",
        "ore2",
        "are2",
    ]
    entries = []
    for index, key in enumerate(keys):
        words = [f"w{key}{n}" for n in range(6)]
        if key == "ita":
            words = ["vita"] + [f"w{key}{n}" for n in range(1, 6)]
        entries.append(
            {
                "key": key,
                "soft_key": key,
                "count": 100 - index,
                "words": words,
            }
        )
    return {"lexicon_version": "sonnet_rhyme_lexicon_v1", "entries": entries}


def test_plan_from_anchored_anchors_line_one():
    plan = plan_from_anchored(
        "Nel mezzo del cammin di nostra vita", make_lexicon(), seed=1
    )
    assert plan["words"][0] == "vita"
    assert len(set(plan["words"])) == 14
    by_class = {}
    for row in plan["lines"]:
        by_class.setdefault(row["class"], set()).add(row["key"])
    assert all(len(values) == 1 for values in by_class.values())
    a_keys = {row["key"] for row in plan["lines"] if row["class"] == "A"}
    assert a_keys == {rhyme_key("vita")}


def test_plan_from_anchored_rejects_absent_key():
    with pytest.raises(ValueError):
        plan_from_anchored(
            "verso finale sconosciutissimo", make_lexicon(), seed=1
        )


def test_repair_endings_replaces_only_wrong_endings():
    planned = ["vita", "core"] * 7
    lines = [
        f"verso numero {index} " + ("vita" if index % 2 == 0 else "cosa")
        for index in range(14)
    ]
    text = "\n".join(lines)
    repaired, stats = repair_endings(text, planned)
    assert stats["repaired_lines"] == 7
    repaired_lines = repaired.splitlines()
    assert repaired_lines[0].endswith("vita")
    assert repaired_lines[1].endswith("core")
    assert repaired_lines[13].endswith("core")


def test_repair_endings_preserves_trailing_punctuation():
    repaired, stats = repair_endings("verso finale cosa,", ["vita"])
    assert repaired == "verso finale vita,"
    assert stats["repaired_lines"] == 1
    repaired, stats = repair_endings("verso finale vita", ["vita"])
    assert repaired == "verso finale vita"
    assert stats["repaired_lines"] == 0
