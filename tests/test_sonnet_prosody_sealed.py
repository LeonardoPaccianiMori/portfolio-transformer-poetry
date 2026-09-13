from sonnet_evaluation.sonnet_prosody_sealed import (
    aggregate,
    canonical_scheme,
    mcnemar,
    paired_comparison,
    pair_records,
    quatrain_scheme,
    score_output,
    tercet_scheme,
)

VALID_LINE = "Nel mezzo del cammin di nostra vita"


def record(system: str, prompt: str, seed: int, **overrides) -> dict:
    base = {
        "system_id": system,
        "prompt_id": prompt,
        "seed": seed,
        "structure_ok": True,
        "stanza_pattern_ok": True,
        "all_lines_valid": True,
        "quatrain_ok": True,
        "tercet_ok": True,
        "hendecasyllable_lines": 14,
        "failed_lines": 0,
        "uncertain_lines": 0,
        "perfect_rhyme_pairs": 4,
        "soft_rhyme_pairs": 1,
        "rhyme_score": 0.8,
        "tercet_scheme": "ABCABC",
    }
    base.update(overrides)
    return base


def test_canonical_scheme_relabels_by_first_appearance():
    assert canonical_scheme("CDECDE") == "ABCABC"
    assert canonical_scheme("CDCCDC") == "ABAABA"


def test_quatrain_and_tercet_extraction():
    scheme = "ABBAABBACDECDE"
    assert quatrain_scheme(scheme) == "ABBAABBA"
    assert tercet_scheme(scheme) == "ABCABC"


def test_score_output_accepts_a_clean_sonnet():
    text = "\n".join([VALID_LINE] * 14)
    metrics = score_output(text)
    assert metrics["hendecasyllable_lines"] == 14
    assert metrics["all_lines_valid"] is True
    assert metrics["structure_ok"] is True
    assert metrics["rhyme_score"] == 1.0


def test_score_output_records_stanza_pattern():
    blocks = [[VALID_LINE] * 4, [VALID_LINE] * 4, [VALID_LINE] * 3, [VALID_LINE] * 3]
    text = "\n\n".join("\n".join(block) for block in blocks)
    metrics = score_output(text)
    assert metrics["stanza_pattern"] == [4, 4, 3, 3]
    assert metrics["stanza_pattern_ok"] is True


def test_pair_records_groups_by_prompt_and_seed():
    records = [
        record("stage_3", "p1", 6200),
        record("dpo", "p1", 6200),
        record("stage_3", "p2", 6200),
    ]
    pairs = pair_records(records)
    assert len(pairs) == 1
    assert pairs[0][0]["prompt_id"] == "p1"


def test_paired_comparison_computes_mean_and_counts():
    pairs = [
        (record("stage_3", "p1", 1, rhyme_score=0.8), record("dpo", "p1", 1, rhyme_score=0.4)),
        (record("stage_3", "p2", 1, rhyme_score=0.5), record("dpo", "p2", 1, rhyme_score=0.5)),
        (record("stage_3", "p3", 1, rhyme_score=0.2), record("dpo", "p3", 1, rhyme_score=0.6)),
    ]
    result = paired_comparison(pairs, "rhyme_score")
    assert result["pair_count"] == 3
    assert abs(result["mean_difference_stage_3_minus_dpo"] - (0.4 + 0.0 - 0.4) / 3) < 1e-9
    assert result["positive_pairs"] == 1
    assert result["negative_pairs"] == 1
    assert result["equal_pairs"] == 1


def test_mcnemar_counts_discordant_pairs():
    pairs = [
        (record("stage_3", "p1", 1, all_lines_valid=True), record("dpo", "p1", 1, all_lines_valid=False)),
        (record("stage_3", "p2", 1, all_lines_valid=False), record("dpo", "p2", 1, all_lines_valid=True)),
        (record("stage_3", "p3", 1, all_lines_valid=True), record("dpo", "p3", 1, all_lines_valid=True)),
        (record("stage_3", "p4", 1, all_lines_valid=False), record("dpo", "p4", 1, all_lines_valid=False)),
    ]
    result = mcnemar(pairs, "all_lines_valid")
    assert result["stage_3_only"] == 1
    assert result["dpo_only"] == 1
    assert result["both"] == 1
    assert result["neither"] == 1
    assert result["discordant"] == 2


def test_aggregate_returns_mean_rates():
    records = [
        record("stage_3", "p1", 1, all_lines_valid=True, hendecasyllable_lines=14),
        record("stage_3", "p2", 1, all_lines_valid=False, hendecasyllable_lines=10),
    ]
    metrics = aggregate(records, "stage_3")
    assert metrics["output_count"] == 2
    assert metrics["all_lines_valid_rate"] == 0.5
    assert metrics["mean_hendecasyllable_lines"] == 12.0
