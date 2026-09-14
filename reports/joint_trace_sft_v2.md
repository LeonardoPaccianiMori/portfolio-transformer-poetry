# Reasoning-Trace SFT v1

Date: 2026-09-15

The model writes a rhyme plan (scheme plus one ending per line) and
then the sonnet. Traces are derived from corpus poems; the evaluation
arms use the same openings, seeds, and recipe. Metrics exclude repair.

## Trace and form

| Metric | Baseline | Trace SFT |
|---|---:|---:|
| Outputs with a parsed trace | 0.0000 | 0.9875 |
| Valid traces (scheme and true rhymes) | 0.0000 | 0.8542 |
| Scheme valid without repair | 0.0000 | 0.5719 |
| Poem scheme equals traced scheme | 0.0000 | 0.5687 |
| Accepted lines | 0.000 | 7.700 |
| Failed lines | 0.000 | 1.475 |
| Uncertain lines | 0.000 | 5.025 |
| 4+4+3+3 stanza pattern | 0.0000 | 0.0000 |
| Rhyme score | 0.0000 | 0.9924 |

## Coherence proxies

| Metric | Baseline | Trace SFT |
|---|---:|---:|
| type_token_ratio | 0.0000 | 0.7042 |
| repeated_line_ratio | 0.0000 | 0.0005 |
| repeated_bigram_ratio | 0.0000 | 0.0350 |
| final_word_repetition | 0.0000 | 0.0287 |

## Memorization screen

Verbatim overlap with 16,298 training sonnets (223,892 lines, 5-word shingles). Poem text only.

| Metric | Baseline | Trace SFT |
|---|---:|---:|
| Outputs with a copied line | 0 | 0 |
| Max copied lines in one output | 0 | 0 |
| Mean 5-gram overlap | 0.0000 | 0.0034 |

## Pre-registered reading

- Result: SELF_PLAY_SIGNAL
- Autonomous scheme validity 0.5719 against baseline 0.0000 and gate 0.30.
- Accepted-line gap +7.700 (minimum -0.5).
- Judge gain None (minimum -0.3).

## Caveats

- Traces come from corpus poems, so the poems are learned imitations
  of real texts, not new compositions.
- The gate uses the poem scheme without repair; trace validity is
  reported separately.

Verification: `python3 scripts/score_reasoning_trace_sft.py`
