# Minerva 7B V6 Validation Candidate Selection

Selection frozen: 2026-08-09. V6 final-test poems and prompts had not been used.

## Candidates

| Epoch | Validation loss | Modern loss | Instruction loss | Controlled form | Mean repetition | Memorization |
|---:|---:|---:|---:|---:|---:|---|
| 5 | 3.164941 | 2.197172 | 1.628403 | 8/8 | 0.2339 | 8/8 low risk |
| 4 | 3.171254 | 2.191147 | 1.442023 | 8/8 | 0.2208 | 8/8 low risk |
| 3 | 3.181642 | 2.186604 | 1.293135 | 8/8 | 0.2423 | 8/8 low risk |

All candidates preserved the supplied first line, produced exactly fourteen
non-empty lines, passed the modern-Italian and instruction gates, and showed no
shared 40-character n-gram with any V6 training poem.

## Qualitative Comparison

Epoch 4 has the most consistent balance across the eight frozen prompts. Most
outputs sustain one topic and recognizable sentence progression. Its main
defects are imperfect historical grammar, occasional semantic discontinuity,
and one repeated two-line construction. Epoch 5 has a marginally lower
validation loss but more conspicuous phrase loops and two stronger collapse
cases. Epoch 3 has more repeated syntactic frames, one leaked numeric-looking
artifact, and the highest validation loss and repetition score.

This is a relative model-selection decision, not a claim that every epoch-4
output is a correct classical sonnet. Rhyme and metre remain unevaluated.

## Frozen Selection

- Selected epoch: **4**
- Adapter SHA-256: `aff3f2c4d193ce880ec9c7a6df6373f433001662c3ca78d7f915890733cb0df3`
- Candidate-summary SHA-256: `451f83f94c52c1dc5554d9c1a329330e2b80ff772f1c9e991bac94845641ce20`
- Selection record: `configs/minerva_7b_v6_selected_adapter.json`

The selected epoch may now be evaluated once on the frozen V6 final test.
