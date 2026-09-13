from sonnet_evaluation.sonnet_prosody_validation import (
    labeled_scheme,
    metre_metrics,
    pairwise_agreement,
    rhyme_metrics,
    score_ground_truth,
    select_review_lines,
)

VALID_LINE = "Nel mezzo del cammin di nostra vita"


def make_poem(poem_id: str, scheme: str, lines: list[str] | None = None) -> dict:
    texts = lines or [VALID_LINE] * 14
    if len(texts) != 14 or len(scheme) != 14:
        raise ValueError("test poems need 14 lines and 14 class letters")
    return {
        "poem_id": poem_id,
        "author": "Test",
        "expected_scheme": scheme,
        "lines": [
            {
                "line_number": index + 1,
                "text": text,
                "expected_metre": "hendecasyllable",
                "rhyme_class": scheme[index],
            }
            for index, text in enumerate(texts)
        ],
    }


def test_labeled_scheme_assigns_letters_by_first_appearance():
    assert labeled_scheme(["a", "b", "a", "a", "b"]) == "ABAAB"


def test_pairwise_agreement_is_one_for_identical_labels():
    assert pairwise_agreement("ABBA", "ABBA") == 1.0


def test_pairwise_agreement_counts_relation_matches():
    assert pairwise_agreement("AB", "AA") == 0.0
    assert pairwise_agreement("AB", "AB") == 1.0
    assert pairwise_agreement("AA", "AB") == 0.0


def test_metre_metrics_pass_a_valid_poem():
    ground_truth = {"schema_version": 1, "poems": [make_poem("p1", "A" * 14)]}
    results = score_ground_truth(ground_truth)
    metrics = metre_metrics(results)
    assert metrics["correct"] == 14
    assert metrics["incorrect"] == 0
    assert metrics["uncertain"] == 0
    assert metrics["accuracy"] == 1.0
    assert metrics["coverage"] == 1.0
    assert metrics["gate_met"] is True
    assert metrics["coverage_warning"] is False


def test_metre_metrics_separate_uncertain_from_incorrect():
    lines = [VALID_LINE] * 14
    lines[5] = "Amor"
    lines[9] = "che la diritta via era smarrita"
    lines[10] = "che la diritta via era smarrita"
    ground_truth = {"schema_version": 1, "poems": [make_poem("p1", "A" * 14, lines)]}
    metrics = metre_metrics(score_ground_truth(ground_truth))
    assert metrics["correct"] == 11
    assert metrics["incorrect"] == 1
    assert metrics["uncertain"] == 2
    assert metrics["definite"] == 12
    assert metrics["accuracy"] == 11 / 12
    assert metrics["coverage"] == 12 / 14
    assert metrics["coverage_warning"] is True


def test_score_ground_truth_observes_identical_lines_as_one_rhyme_class():
    ground_truth = {"schema_version": 1, "poems": [make_poem("p1", "A" * 14)]}
    poems = score_ground_truth(ground_truth)
    assert poems[0]["observed_scheme"] == "A" * 14
    assert poems[0]["scheme_match"] is True


def test_rhyme_metrics_detect_a_scheme_mismatch():
    ground_truth = {"schema_version": 1, "poems": [make_poem("p1", "AB" * 7)]}
    metrics = rhyme_metrics(score_ground_truth(ground_truth))
    assert metrics["exact_scheme_matches"] == 0
    assert metrics["scheme_accuracy"] == 0.0
    assert metrics["mean_pairwise_agreement"] < 1.0
    assert metrics["gate_met"] is False


def test_select_review_lines_prioritizes_flagged_lines():
    lines = [VALID_LINE] * 14
    lines[3] = "dolce vita"
    ground_truth = {"schema_version": 1, "poems": [make_poem("p1", "A" * 14, lines)]}
    review = select_review_lines(score_ground_truth(ground_truth), limit=5)
    assert len(review) == 5
    assert review[0]["category"] == "metre_flagged"
    assert review[0]["line_number"] == 4
    assert {line["category"] for line in review[1:]} == {"sample"}
