# Sonnet Prosody Checker Validation v1

Date: 2026-09-13

This report validates the rule-based `sonnet_prosody` checker against the
frozen ground truth of 30 sonnets (420 lines).
The checker flags were reviewed on 2026-09-13; see
`reports/sonnet_prosody_review_outcome_v1.md`.

## Ground truth

- Selection rule: Per author, keep the train sonnets in manifest order with exactly 14 clean lines, no square brackets, and no entry in excluded_poems; take the 15 items at positions round(i*(n-1)/14) for i=0..14.
- Annotation method: The rhyme classes were read from the line-final words of each selected poem and checked against the frozen text. They are a draft that the owner reviews with the flagged-line packet before the checker is used as a metric or reward.

## Metre

- Lines scored: 420
- Correct: 366
- Incorrect: 0
- Uncertain: 54
- Accuracy among definite lines: 100.00%
- Definite coverage: 87.14%
- Coverage warning (below 90%): YES
- Gate (accuracy >= 95% on definite lines): PASS

Ambiguous lines are excluded from the definite accuracy and reported
separately in the uncertainty reasons and the review packet.

## Rhyme

- Poems: 30
- Exact scheme matches: 29 (96.67%)
- Soft (Sicilian) scheme matches: 30 (100.00%)
- Mean pairwise agreement: 99.96%
- Gate (both >= 90%): PASS

### Mismatched poems

- dante_xvi_amore_e_l_cor_gentil_sono_una_cosa: expected ABABABABCDECDE, observed ABABABABCDEFDE (pairwise 98.90%)

## Full-corpus statistics (V6)

- Poems: 1868
- Lines: 26152
- Correct: 22781
- Incorrect: 230
- Uncertain: 3141
- Poems with all 14 lines valid: 490
- Verse types: {'piano': 21972, 'tronco': 808, 'sdrucciolo': 1}
- Uncertainty reasons: {'synaeresis_possible': 671, 'sdrucciolo_possible': 1470, 'hiatus_possible': 2097, 'dialefe_possible': 579, 'no_vowel_final_word': 4}

## Manual review

- Packet: `reports/sonnet_prosody_review_packet_v1.csv`
- Selection rule: all metre-flagged lines, then scheme-mismatch lines,
  then a deterministic sample, up to 100 lines.
- Status: reviewed on 2026-09-13 at the owner's request. Outcome:
  `reports/sonnet_prosody_review_outcome_v1.md`.

## Verification

- Rebuild: `python3 scripts/build_sonnet_prosody_ground_truth.py`
- Validate: `python3 scripts/validate_sonnet_prosody.py`
