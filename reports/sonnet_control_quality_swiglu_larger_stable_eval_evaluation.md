# Selected SwiGLU Sonnet Fine-Tuning Evaluation

This evaluation covers the exact best-validation checkpoint from
`sonnet_control_quality_swiglu_larger_stable_eval_20k_001`: step 1,000 with
deterministic validation loss 2.9771. It was initialized from the selected
33.7M-parameter broader-Italian SwiGLU parent at pretraining step 110,000.

## Generation Protocol

Five fixed prompts used seeds 1337 through 1341, temperature 1.0, no top-k
restriction, and a 900-token safety limit. The decoder suppressed
`<|endoftext|>` until each generation reached 14 non-empty lines.

## Automatic Results

| Measurement | Result |
| --- | --- |
| Prompt preservation | 5 / 5 |
| Decoder stop reason | 5 / 5 target-line limit |
| Average characters per sample | 521.0 |
| Average repeated character-4-gram ratio | 0.0846 |
| Memorization-risk label | 5 / 5 low |
| Maximum normalized longest common training span | 17 characters |

The 14-line result verifies the generation controller. It does not verify that
the model learned sonnet metre, rhyme, octave/sestet structure, or coherent
poetic development.

## Qualitative Assessment

All five samples have low sonnet-like structure and low coherence. They retain
some surface signals of archaic Italian, including apostrophes, poetic lexical
fragments, and line breaks, but most lines contain malformed words, broken
syntax, or incompatible semantic relations. The main failure is not local
looping or direct copying; it is that the model does not reliably form coherent
Italian sentences or a sustained poetic thought.

The `solo_et_pensoso` prompt is especially useful evidence. Although it begins
with a famous Petrarch opening, its continuation becomes unrelated malformed
text. The memorization checker finds a low-risk 16-character longest common
span, so this is not evidence of direct continuation copying.

## Conclusion

Fine-tuning transferred the requested line-oriented output format only through
the decoder control. It did not make the selected parent a realistic sonnet
generator. The exact step-1,000 checkpoint remains the correct artifact for
evaluation because later training overfits under the deterministic validation
protocol.

The detailed automatic and qualitative evidence is in
`reports/generation_metrics_sonnet_control_quality_swiglu_larger_stable_eval_best.md`,
`reports/memorization_sonnet_control_quality_swiglu_larger_stable_eval_best.md`,
and
`reports/qualitative_review_sonnet_control_quality_swiglu_larger_stable_eval_best.md`.
