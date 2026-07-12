# Controlled Experiment Comparison: Learning-Rate Schedule

This report compares two sonnet-corpus training runs with matching architecture, tokenizer, data, optimizer initialization, seed, and training schedule except for the declared experimental factor. The intended difference is the learning-rate schedule: constant 3e-5 versus 250-step warmup plus cosine decay to 3e-6.

## Comparability

All declared shared settings match.

## Results

| Initialization | Best Step | Best Val | Final Val | Avg Chars | Avg Lines | Avg Repeated 4-gram Ratio | Avg Unique-Character Ratio | Low Memorization Risk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| constant | 2,500 | 2.6720 | 4.2649 | 522.6 | 14.0 | 0.1115 | 0.0722 | 5/5 |
| warmup_cosine | 2,500 | 2.6666 | 3.6671 | 605.2 | 14.0 | 0.1091 | 0.0644 | 5/5 |

## Interpretation

The warmup_cosine arm achieved a lower best validation loss by 0.0054 nats per BPE token. Both arms use the same tokenizer, so this loss difference is directly comparable.

The 14-line generation target is decoder enforced in both arms. Automatic format metrics therefore validate the control procedure, not learned sonnet structure. Memorization labels are heuristic surface-copying checks. Qualitative reviews must be read alongside this report.

## Qualitative Findings

The first-pass review finds no clear quality improvement from the schedule. Both
batches have medium local language/style plausibility, low coherence, and low
sonnet-like structure. The warmup-cosine batch includes one unusually long,
repetitive `line_start` sample, while its remaining samples are broadly similar
to the constant-rate batch. See
`reports/qualitative_review_sonnet_control_pretrained_fresh_best.md` and
`reports/qualitative_review_sonnet_control_pretrained_warmup_cosine_best.md`.

## Decision

Keep the constant-rate pretrained-fresh configuration as the baseline. The
schedule-only change does not produce a meaningful selected-checkpoint loss or
qualitative gain under this one-seed evaluation. The next isolated stability
experiment is gradient clipping.

## Limits

This is one random seed with validation estimated from five sampled batches at each evaluation. It supports the value of broader pretraining under this setup, but it does not establish a general result without additional seeds and more stable validation estimates.
