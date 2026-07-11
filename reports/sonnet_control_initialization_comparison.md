# Controlled Initialization Comparison

This report compares two sonnet-corpus training runs with matching architecture, tokenizer, data, optimizer initialization, seed, and training schedule. The intended difference is model initialization: broader-pretrained weights versus random weights.

## Comparability

All declared shared settings match.

## Results

| Initialization | Best Step | Best Val | Final Val | Avg Chars | Avg Lines | Avg Repeated 4-gram Ratio | Avg Unique-Character Ratio | Low Memorization Risk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| pretrained | 2,500 | 2.6720 | 4.2649 | 522.6 | 14.0 | 0.1115 | 0.0722 | 5/5 |
| random | 13,000 | 3.2047 | 3.4472 | 529.2 | 14.0 | 0.0895 | 0.0685 | 5/5 |

## Interpretation

The pretrained arm achieved a lower best validation loss by 0.5327 nats per BPE token. Both arms use the same tokenizer, so this loss difference is directly comparable. The pretrained arm reached its best value earlier, while the random arm required substantially more sonnet-only updates and still had a worse best validation result.

The 14-line generation target is decoder enforced in both arms. Automatic format metrics therefore validate the control procedure, not learned sonnet structure. Memorization labels are heuristic surface-copying checks. Qualitative reviews must be read alongside this report.

## Qualitative Findings

The assistant-authored first-pass reviews rate every sample low for sonnet-like
structure and coherence. The pretrained arm is consistently more readable at a
local level: it has more stable word boundaries, more recognizable Italian-like
phrases, and fewer fused or invented-looking word forms. The random arm often
produces malformed strings such as `Donnaa`, `credevalor`, and
`celestiranite`.

This is a local language-quality advantage, not successful sonnet generation.
The pretrained samples still fail to sustain syntax, semantics, rhyme, or a
coherent poetic argument across fourteen lines. See
`reports/qualitative_review_sonnet_control_pretrained_fresh_best.md` and
`reports/qualitative_review_sonnet_control_random_best.md` for every prompt and
the complete generated text.

## Limits

This is one random seed with validation estimated from five sampled batches at each evaluation. It supports the value of broader pretraining under this setup, but it does not establish a general result without additional seeds and more stable validation estimates.
