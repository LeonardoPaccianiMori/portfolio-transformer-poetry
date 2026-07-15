# Controlled Experiment Comparison: RoPE Pretraining

This report compares two sonnet-corpus training runs with matching architecture, tokenizer, data, optimizer initialization, seed, and training schedule except for the declared experimental factor. The intended difference is learned absolute position embeddings versus RoPE throughout broader-corpus pretraining, followed by the same fresh-optimizer sonnet fine-tuning protocol.

## Comparability

All declared shared settings match.

## Results

| Arm | Best Step | Best Val | Final Val | Avg Chars | Avg Lines | Avg Repeated 4-gram Ratio | Avg Unique-Character Ratio | Low Memorization Risk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| learned_absolute_pretraining | 2,500 | 2.6720 | 4.2649 | 522.6 | 14.0 | 0.1115 | 0.0722 | 5/5 |
| rope_pretraining | 1,500 | 2.7329 | 4.4846 | 555.4 | 14.0 | 0.1003 | 0.0651 | 5/5 |

## Interpretation

The learned_absolute_pretraining arm achieved a lower best validation loss by 0.0609 nats per BPE token. Both arms use the same tokenizer, so this loss difference is directly comparable.

The 14-line generation target is decoder enforced in both arms. Automatic format metrics therefore validate the control procedure, not learned sonnet structure. Memorization labels are heuristic surface-copying checks. Qualitative reviews must be read alongside this report.

## Limits

This is one random seed with validation estimated from five sampled batches at each evaluation. It supports the value of broader pretraining under this setup, but it does not establish a general result without additional seeds and more stable validation estimates.

## Qualitative Findings

Both arms produce locally Italian-like, archaizing fragments but have low
coherence and low learned sonnet-like structure across the fixed prompts. The
RoPE arm has no 40-character training-text matches but shows no qualitative
advantage: its samples include malformed or invented-looking forms such as
`anese`, `insemeraa`, `facondiente`, `dagit`, and `direschera`, together with
unstable grammatical relations. The learned-absolute-position samples have
comparable limitations. The shared 14-line output length is decoder enforced,
not evidence that either model learned sonnet form.

## Decision

Retain the learned-absolute-position LayerNorm parent as the baseline. RoPE is
worse by 0.0609 nats per BPE token at best validation, and the fixed samples do
not show a compensating qualitative improvement. This is a single-seed result,
not a general claim that RoPE is unsuitable for language models. The public
evidence is the metrics, memorization, qualitative-review, and comparison
reports named above.
