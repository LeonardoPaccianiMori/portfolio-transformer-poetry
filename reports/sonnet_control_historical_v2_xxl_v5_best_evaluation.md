# Selected Historical XXL V5 Sonnet Fine-Tuning Evaluation

This evaluation covers the exact stable-protocol checkpoint selected from
`sonnet_control_historical_v2_xxl_v5_stable_eval_20k_001`: step 2,250 with
deterministic validation loss 3.072862. It was initialized from the selected
234.8M-parameter historical-Italian XXL parent at pretraining step 18,000.

## Generation Protocol

Five fixed prompts used seeds 1337 through 1341, temperature 1.0, no top-k
restriction, and a 900-token safety limit. The decoder suppressed
`<|endoftext|>` until each generation reached 14 non-empty lines.

## Automatic Results

| Measurement | Result |
| --- | --- |
| Prompt preservation | 5 / 5 |
| Decoder stop reason | 5 / 5 target-line limit |
| Average characters per sample | 539.4 |
| Average repeated character-4-gram ratio | 0.0985 |
| Memorization-risk label | 5 / 5 low |
| Maximum normalized longest common training span | 19 characters |

The five 14-line outputs verify that the generation controller works. They do
not demonstrate that the model learned sonnet metre, rhyme, octave/sestet
structure, or a semantic volta.

## Qualitative Assessment

All five samples are rated low for sonnet-like structure, language/style
plausibility, and coherence. They preserve occasional surface properties of
the target distribution: apostrophes, accented characters, poetic vocabulary,
and line breaks. However, each sample contains malformed or invented lexical
items, broken syntax, or abrupt semantic shifts that prevent a coherent poetic
statement.

Repetition is secondary rather than the principal failure. The automatic
repeated-4-gram ratios range from 0.0368 to 0.1303, and the qualitative review
rates two samples low and three medium for repetition problems. The stronger
evidence is lexical and syntactic degeneration, such as `cominlo`,
`creaturaggia`, and `Io s'ha Tovar`.

The nearest-neighbor check finds zero 40-character n-gram containment for all
five samples. The maximum normalized contiguous overlap is 19 characters, so
these outputs provide no evidence of direct surface copying under the recorded
heuristic. The `solo_et_pensoso` sample begins with its supplied Petrarch
prompt but does not continue it coherently.

## Conclusion

The selected XXL parent and V5 fine-tuning stage improve neither grammatical
Italian nor sustained poetic coherence enough for this project to call the
model an acceptable sonnet generator. Line count is currently controlled by
the decoder, not reliably learned form.

This does not yet trigger the one permitted PAISÀ rescue run. The approved
from-scratch exit policy requires poem-aware task-format post-training and the
fixed 20-output acceptance evaluation before judging the current attempt.

The detailed evidence is in
`reports/generation_metrics_sonnet_control_historical_v2_xxl_v5_best.md`,
`reports/memorization_checks_sonnet_control_historical_v2_xxl_v5_best.md`, and
`reports/qualitative_review_sonnet_control_historical_v2_xxl_v5_best.md`.
