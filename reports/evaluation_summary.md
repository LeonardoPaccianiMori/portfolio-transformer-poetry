# First Transformer Evaluation Summary

Evaluated run: `runs/transformer_context768_scaled_001`

Generation directory: `outputs/generations/transformer_context768_scaled_001`

Reports used:

- `reports/generation_metrics.md`
- `reports/memorization_checks.md`
- `reports/qualitative_review.md`

## Summary

The scaled full-context character transformer has learned some surface properties of the corpus, but it does not yet generate coherent sonnets.

It reliably preserves the provided prompt, produces poem-like line breaks, uses punctuation and accented characters, and sometimes emits the correct `<|poem_end|>` separator. However, the outputs are mostly malformed at the word level, do not stay near the 14-line sonnet form, and do not maintain readable semantic coherence.

The current result is useful as a first from-scratch transformer baseline, not as a satisfactory poetry model.

## Automatic Findings

The generated samples are about 904-915 characters long and contain 26-28 non-empty lines. This is much longer than a 14-line sonnet, so the model is not yet controlling poem structure.

All five prompts are preserved according to the automatic prompt check. This is a useful baseline behavior: generation begins from the requested prompt instead of ignoring it.

The `<|poem_end|>` separator appears in 4 of 5 samples, but the qualitative review also shows corrupted separator-like strings. This means the model has learned that separator-looking text exists, but not how to use it cleanly as a stopping/control signal.

Repeated 4-gram ratios are low to moderate, between 0.0384 and 0.0732. There is no obvious runaway repetition loop in these samples, but local malformed repetition remains visible in the generated text.

## Memorization Findings

The nearest-neighbor memorization proxy reports low risk for all five samples.

Each sample has 0.0000 40-character containment against the training split. Longest common substrings are short, between 9 and 15 characters. This does not prove that the model cannot memorize, but it gives no evidence of direct surface copying in this fixed-prompt batch.

This is an important distinction: the model is weak because it is incoherent, not because these samples are copied from training poems.

## Qualitative Findings

All five samples are rated low for sonnet-like structure and coherence.

Language/style plausibility is low for most samples and medium only for `solo_et_pensoso`, where the real Petrarch prompt and surface character patterns make the output look closer to the corpus style. Even there, the continuation is not semantically coherent.

Repetition problems are rated medium across the samples. The issue is not a single repeated phrase taking over the output. The issue is local character-level degeneration: malformed words, doubled letters, broken apostrophe patterns, and corrupted special-token fragments.

Memorization concern is rated low across the samples, matching the automatic nearest-neighbor report.

## Main Failure Modes

- The model produces many malformed words rather than stable Italian or archaic Italian lexical units.
- The model over-generates lines and does not stop at sonnet length.
- The model sometimes emits corrupted versions of `<|poem_end|>`.
- The model has weak semantic coherence across lines.
- The model imitates surface punctuation and accent patterns better than it models meaning.

## Interpretation

The character-level transformer is learning local character statistics and some corpus formatting, but the current training setup is not enough for good sonnet generation.

The most likely causes are:

- character-level tokenization makes long-range word and phrase structure harder;
- 5000 training steps is still a limited training budget;
- generation currently has no sonnet-aware stopping rule;
- the model is not explicitly trained or decoded to stop after 14 lines;
- the architecture is still small for full-sonnet context modeling.

## Recommended Next Steps

1. Add line-count-aware generation stopping so samples can stop after a target number of non-empty lines or at `<|poem_end|>`.
2. Add sampling controls such as temperature and top-k sampling so generation quality can be tested under fixed decoding settings.
3. Run another evaluation after decoding controls are implemented.
4. After the character baseline is better understood, start the planned BPE tokenizer track to reduce sequence length and improve word-level stability.

The next implementation checkpoint should be generation controls, not more training. The current evaluation shows that uncontrolled decoding is a major part of the visible failure.
