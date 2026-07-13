# Controlled Experiment Comparison: RMSNorm Pretraining

This report compares two sonnet-corpus training runs with matching architecture, tokenizer, data, optimizer initialization, seed, and training schedule except for the declared experimental factor. The intended difference is LayerNorm versus RMSNorm throughout broader-corpus pretraining, followed by the same fresh-optimizer sonnet fine-tuning protocol.

## Comparability

All declared shared settings match.

## Results

| Arm | Best Step | Best Val | Final Val | Avg Chars | Avg Lines | Avg Repeated 4-gram Ratio | Avg Unique-Character Ratio | Low Memorization Risk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| layer_norm_pretraining | 2,500 | 2.6720 | 4.2649 | 522.6 | 14.0 | 0.1115 | 0.0722 | 5/5 |
| rmsnorm_pretraining | 2,000 | 2.6740 | 4.2960 | 534.0 | 14.0 | 0.0997 | 0.0672 | 5/5 |

## Interpretation

The layer_norm_pretraining arm achieved a lower best validation loss by 0.0020 nats per BPE token. Both arms use the same tokenizer, so this loss difference is directly comparable.

The 14-line generation target is decoder enforced in both arms. Automatic format metrics therefore validate the control procedure, not learned sonnet structure. Memorization labels are heuristic surface-copying checks. Qualitative reviews must be read alongside this report.

## Limits

This is one random seed with validation estimated from five sampled batches at each evaluation. It supports the value of broader pretraining under this setup, but it does not establish a general result without additional seeds and more stable validation estimates.

## Qualitative Findings

Both arms produce locally Italian-like, archaizing fragments but receive low
ratings for coherence and learned sonnet-like structure across the five fixed
prompts. The RMSNorm arm has no 40-character training-text matches and does not
show a qualitative improvement over LayerNorm: samples include malformed forms
such as `plo`, `lassorimanga`, and `tremarzo`, plus unstable syntax and speaker
relations. The LayerNorm samples have comparable failures. The 14-line length
in both arms is imposed by the decoder and is not evidence that either model
learned the sonnet form.

## Decision

Retain the LayerNorm-pretrained fresh fine-tuning run as the baseline. The
observed validation-loss difference is negligible at this evaluation precision,
and the fixed samples show no clear RMSNorm quality advantage. This is a
single-seed result, not a general claim about RMSNorm. The public evidence is
the metrics, memorization, qualitative-review, and comparison reports named
above.
