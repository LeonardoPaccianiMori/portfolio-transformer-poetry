# Controlled Experiment Comparison: Gradient Clipping

This report compares two sonnet-corpus training runs with matching architecture, tokenizer, data, optimizer initialization, seed, and training schedule except for the declared experimental factor. The intended difference is global gradient clipping with max norm 1.0.

## Comparability

All declared shared settings match.

## Results

| Arm | Best Step | Best Val | Final Val | Avg Chars | Avg Lines | Avg Repeated 4-gram Ratio | Avg Unique-Character Ratio | Low Memorization Risk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| unclipped | 2,500 | 2.6720 | 4.2649 | 522.6 | 14.0 | 0.1115 | 0.0722 | 5/5 |
| clip_1_0 | 2,500 | 2.6742 | 4.2753 | 527.2 | 14.0 | 0.1051 | 0.0722 | 5/5 |

## Interpretation

The unclipped arm achieved a lower best validation loss by 0.0022 nats per BPE token. Both arms use the same tokenizer, so this loss difference is directly comparable.

The 14-line generation target is decoder enforced in both arms. Automatic format metrics therefore validate the control procedure, not learned sonnet structure. Memorization labels are heuristic surface-copying checks. Qualitative reviews must be read alongside this report.

## Qualitative Findings

The clipping run reached the same decoder-enforced format and retained low
copying-risk labels for all five prompts. Its mean repeated 4-gram ratio is
slightly lower (0.1051 versus 0.1115), but this is not a clear quality gain:
the `amor` output repeats `perdono` heavily, and the `line_start` output is
byte-identical to the unclipped baseline with the same prompt and seed. Both
batches have medium local language/style plausibility, low coherence, and low
sonnet-like structure. See
`reports/qualitative_review_sonnet_control_pretrained_fresh_best.md` and
`reports/qualitative_review_sonnet_control_pretrained_clip1_best.md`.

## Decision

Keep the unclipped pretrained-fresh configuration as the baseline. Clipping at
global norm 1.0 was active during training, but its selected validation loss is
0.0022 nats per BPE token worse and the qualitative review identifies no
corresponding benefit. Retain clipping as an optional, logged stability control
for future experiments rather than enabling it by default.

## Limits

This is one random seed with validation estimated from five sampled batches at each evaluation. It supports the value of broader pretraining under this setup, but it does not establish a general result without additional seeds and more stable validation estimates.
