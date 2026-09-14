# Reasoning-Trace SFT v1

Date: 2026-09-14

The model writes a rhyme plan (scheme plus one ending per line) and
then the sonnet. Traces are derived from corpus poems; the evaluation
arms use the same openings, seeds, and recipe. Metrics exclude repair.

## Trace and form

| Metric | Baseline | Trace SFT |
|---|---:|---:|
| Outputs with a parsed trace | 0.0000 | 0.9917 |
| Valid traces (scheme and true rhymes) | 0.0000 | 0.0458 |
| Scheme valid without repair | 0.0000 | 0.0292 |
| Poem scheme equals traced scheme | 0.0000 | 0.0292 |
| Accepted lines | 8.817 | 8.292 |
| Failed lines | 8.883 | 1.496 |
| Uncertain lines | 12.817 | 4.354 |
| 4+4+3+3 stanza pattern | 0.0000 | 0.0000 |
| Rhyme score | 0.8553 | 0.9634 |

## Coherence proxies

| Metric | Baseline | Trace SFT |
|---|---:|---:|
| type_token_ratio | 0.4410 | 0.7292 |
| repeated_line_ratio | 0.0001 | 0.0003 |
| repeated_bigram_ratio | 0.1496 | 0.0240 |
| final_word_repetition | 0.4049 | 0.0660 |

## Memorization screen

Verbatim overlap with 16,298 training sonnets (223,892 lines, 5-word shingles). Poem text only.

| Metric | Baseline | Trace SFT |
|---|---:|---:|
| Outputs with a copied line | 0 | 0 |
| Max copied lines in one output | 0 | 0 |
| Mean 5-gram overlap | 0.0021 | 0.0027 |

## Pre-registered reading

- Result: SELF_PLAY_NULL
- Autonomous scheme validity 0.0292 against baseline 0.0000 and gate 0.30.
- Accepted-line gap -0.525 (minimum -0.5).
- Judge gain None (minimum -0.3).

## Caveats

- Traces come from corpus poems, so the poems are learned imitations
  of real texts, not new compositions.
- The gate uses the poem scheme without repair; trace validity is
  reported separately.

Verification: `python3 scripts/score_reasoning_trace_sft.py`
