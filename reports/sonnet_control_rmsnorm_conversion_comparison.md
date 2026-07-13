# Controlled Experiment Comparison: LayerNorm-to-RMSNorm Conversion

This report compares two sonnet-corpus training runs with matching architecture, tokenizer, data, optimizer initialization, seed, and training schedule except for the declared experimental factor. The intended difference is LayerNorm-to-RMSNorm conversion before sonnet fine-tuning, with a fresh optimizer state.

## Comparability

All declared shared settings match.

## Results

| Arm | Best Step | Best Val | Final Val | Avg Chars | Avg Lines | Avg Repeated 4-gram Ratio | Avg Unique-Character Ratio | Low Memorization Risk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| layer_norm_baseline | 2,500 | 2.6720 | 4.2649 | 522.6 | 14.0 | 0.1115 | 0.0722 | 5/5 |
| rmsnorm_conversion | 2,000 | 2.6739 | 4.2986 | 542.8 | 14.0 | 0.0948 | 0.0691 | 5/5 |

## Interpretation

The layer_norm_baseline arm achieved a lower best validation loss by 0.0019 nats per BPE token. Both arms use the same tokenizer, so this loss difference is directly comparable.

The 14-line generation target is decoder enforced in both arms. Automatic format metrics therefore validate the control procedure, not learned sonnet structure. Memorization labels are heuristic surface-copying checks. Qualitative reviews must be read alongside this report.

## Qualitative Findings

The conversion batch preserved the fixed 14-line decoder procedure and low
copying-risk labels. Its average repeated 4-gram ratio is lower (0.0948 versus
0.1115), but this is not a clear quality gain: both batches have medium local
Italian-like texture, low coherence, and low sonnet-like structure. The RMSNorm
conversion outputs still contain malformed words and unstable syntax, including
`anaalta`, `partorìdila`, `penetroso`, and `convendetta`. See
`reports/qualitative_review_sonnet_control_pretrained_fresh_best.md` and
`reports/qualitative_review_sonnet_control_layernorm_to_rmsnorm_best.md`.

## Decision

Keep the LayerNorm pretrained-fresh model as the baseline. The conversion
fine-tune gives no meaningful selected-loss or qualitative improvement, but it
does not test RMSNorm during broader-corpus pretraining. The scheduled true
RMSNorm experiment must pretrain a fresh RMSNorm parent before applying the
same sonnet fine-tuning and evaluation protocol.

## Limits

This is one random seed with validation estimated from five sampled batches at each evaluation. It supports the value of broader pretraining under this setup, but it does not establish a general result without additional seeds and more stable validation estimates.
