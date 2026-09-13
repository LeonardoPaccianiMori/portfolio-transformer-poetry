# Retroactive Prosody Scoring of the Sealed Stage-3 and DPO Outputs v1

Date: 2026-09-13

This report scores the frozen one-time final-test outputs with the
rule-based `sonnet_prosody` checker. It only reads the sealed
artifacts. The checker flags were reviewed on 2026-09-13; see
`reports/sonnet_prosody_review_outcome_v1.md`. The results below
measure form only, not literary quality. Selection effects are not
implied.

- Sealed outputs: 4976
- Paired outputs: 2488
- Final protocol SHA-256: `8284101f80c739f366f5e46322bf5d83a61b7fe7b0f1a54e32154d6adc682c11`
- DPO adapter SHA-256: `72aa174b2ef87e021a367b0f7e786fce8c3437bb5ca1f8c7f9c5b13588620822`

## Per-system form metrics

| Metric | Stage 3 | DPO |
|---|---:|---:|
| Outputs | 2488 | 2488 |
| Structure OK rate | 0.9996 | 0.9992 |
| Stanza pattern OK rate | 0.0326 | 0.0374 |
| All 14 lines valid rate | 0.0000 | 0.0000 |
| No definite metre failure rate | 0.2267 | 0.2291 |
| Quatrain scheme OK rate | 0.0000 | 0.0000 |
| Tercet scheme OK rate | 0.0000 | 0.0000 |
| Mean hendecasyllable lines | 5.473 | 5.702 |
| Mean failed lines | 2.047 | 1.989 |
| Mean uncertain lines | 6.480 | 6.308 |
| Mean perfect rhyme pairs | 1.191 | 1.197 |
| Mean soft rhyme pairs | 0.244 | 0.223 |
| Mean rhyme score | 0.5885 | 0.5916 |

## Paired comparisons (Stage 3 minus DPO)

| Field | Mean difference | 95% CI | + pairs | - pairs | = pairs |
|---|---:|---|---:|---:|---:|
| hendecasyllable_lines | -0.2291 | [-0.3343, -0.1239] | 950 | 1055 | 483 |
| failed_lines | 0.0583 | [-0.0232, 0.1398] | 923 | 866 | 699 |
| uncertain_lines | 0.1712 | [0.0594, 0.2830] | 1099 | 954 | 435 |
| perfect_rhyme_pairs | -0.0060 | [-0.0784, 0.0663] | 806 | 807 | 875 |
| soft_rhyme_pairs | 0.0201 | [-0.0082, 0.0484] | 352 | 314 | 1822 |
| rhyme_score | -0.0031 | [-0.0271, 0.0209] | 619 | 652 | 1217 |

## Discordant binary outcomes (McNemar normal approximation)

| Field | Stage 3 only | DPO only | Both | Neither | p |
|---|---:|---:|---:|---:|---:|
| structure_ok | 2 | 1 | 2485 | 0 | 0.5637 |
| stanza_pattern_ok | 68 | 80 | 13 | 2327 | 0.3239 |
| all_lines_valid | 0 | 0 | 0 | 2488 | 1 |
| quatrain_ok | 0 | 0 | 0 | 2488 | 1 |
| tercet_ok | 0 | 0 | 0 | 2488 | 1 |

## Tercet pattern distribution (top patterns)

- Stage 3: {'ABCDEF': 1994, 'ABCBDE': 33, 'ABCDEA': 30, 'AABCDE': 30, 'ABBCDE': 30, 'ABCDEB': 29, 'ABCCDE': 29, 'ABCDDE': 29, 'ABCDAE': 27, 'ABACDE': 26, 'ABCDED': 26, 'ABCADE': 26, 'ABCDEE': 25, 'ABCDBE': 25, 'ABCDEC': 23, 'ABCDCE': 21, 'ABCDE?': 13, 'ABCD?E': 12, 'ABCD??': 5, 'ABCDCC': 4, 'AAABCD': 3, 'ABBCBD': 3, 'AABCDA': 2, 'AB?CDE': 2, 'ABBCDC': 2, 'ABBCDB': 2, 'ABBBCD': 2, 'AB????': 2, 'ABCCDC': 2, 'ABBBBC': 2, 'ABCDBD': 1, 'ABCA??': 1, 'ABCDDB': 1, 'ABCBDD': 1, 'ABAACD': 1, 'AABCCD': 1, 'ABCDAD': 1, 'ABAC??': 1, 'ABCDDD': 1, 'A?????': 1, 'A?BBCD': 1, 'ABACBD': 1, '?': 1, 'ABCCCD': 1, 'ABCBAB': 1, 'ABCD?A': 1, 'ABCDBA': 1, 'AABACD': 1, 'ABC???': 1, 'ABACCD': 1, 'ABACDC': 1, 'ABACDA': 1, 'ABCADA': 1, 'ABBCDD': 1, 'ABACDD': 1, 'ABBCCD': 1, 'ABCACD': 1, 'ABCDAA': 1, 'ABC?DE': 1}
- DPO: {'ABCDEF': 2031, 'ABCBDE': 34, 'ABCDEC': 31, 'ABCADE': 31, 'AABCDE': 31, 'ABACDE': 30, 'ABCDCE': 27, 'ABCCDE': 25, 'ABCDDE': 25, 'ABBCDE': 24, 'ABCDAE': 23, 'ABCDBE': 23, 'ABCDEA': 21, 'ABCDEE': 21, 'ABCDEB': 19, 'ABCDED': 15, 'ABCDE?': 11, 'ABCD?E': 9, 'ABCD??': 4, 'ABCCDD': 4, 'ABCAAD': 3, 'ABCADB': 2, 'ABC?DE': 2, 'ABC???': 2, 'ABCCAD': 2, 'ABCDDD': 2, '??????': 2, 'ABCBBD': 2, '?': 2, 'ABACAD': 2, 'ABBCBD': 2, 'AAABCD': 1, 'ABBCDD': 1, 'ABCDAA': 1, 'ABCADA': 1, 'ABCDCA': 1, 'A?????': 1, 'ABBCDB': 1, 'ABCBCD': 1, 'ABACDA': 1, 'AB?CDE': 1, 'AABBBC': 1, 'ABBBCB': 1, 'ABACCD': 1, 'ABCD?D': 1, 'ABBCBB': 1, 'AABCBD': 1, 'ABCDAB': 1, 'ABBACD': 1, 'ABCDBC': 1, 'ABBCDC': 1, 'AABCDD': 1, 'ABBCAD': 1, 'ABBBCC': 1, 'AABCAB': 1, 'ABCCDB': 1, 'ABCBDB': 1}

## Verification

- Score: `python3 scripts/score_sealed_sonnet_prosody.py`
- Checker validation: `reports/sonnet_prosody_validation_v1.md`
