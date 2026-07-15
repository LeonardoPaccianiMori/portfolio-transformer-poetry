# Controlled Experiment Comparison: SwiGLU Pretraining

This report compares two sonnet-corpus training runs with matching architecture, tokenizer, data, optimizer initialization, seed, and training schedule except for the declared experimental factor. The intended difference is ReLU versus parameter-matched SwiGLU feed-forward blocks throughout broader-corpus pretraining, followed by the same fresh-optimizer sonnet fine-tuning protocol.

## Comparability

All declared shared settings match.

## Results

| Arm | Best Step | Best Val | Final Val | Avg Chars | Avg Lines | Avg Repeated 4-gram Ratio | Avg Unique-Character Ratio | Low Memorization Risk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| relu_pretraining | 2,500 | 2.6720 | 4.2649 | 522.6 | 14.0 | 0.1115 | 0.0722 | 5/5 |
| swiglu_pretraining | 1,000 | 2.5935 | 4.3082 | 537.8 | 14.0 | 0.0863 | 0.0718 | 5/5 |

## Interpretation

The swiglu_pretraining arm achieved a lower best validation loss by 0.0785 nats per BPE token. Both arms use the same tokenizer, so this loss difference is directly comparable.

The 14-line generation target is decoder enforced in both arms. Automatic format metrics therefore validate the control procedure, not learned sonnet structure. Memorization labels are heuristic surface-copying checks. Qualitative reviews must be read alongside this report.

## Limits

This is one random seed with validation estimated from five sampled batches at each evaluation. It supports the value of broader pretraining under this setup, but it does not establish a general result without additional seeds and more stable validation estimates.

## Qualitative Findings

SwiGLU has the lower selected validation loss and lower repeated 4-gram ratio,
while both arms have low surface-copying risk for all fixed prompts. However,
the SwiGLU samples still receive low ratings for coherence and learned
sonnet-like structure. They contain malformed or invented-looking forms such as
`ggioli`, `sondere`, `graa`, `prebili`, `unidre`, and `orraglio`, alongside
unstable syntax and speaker relations. The ReLU samples have comparable
limitations. The shared 14-line length is decoder enforced rather than evidence
that either model learned sonnet form.

## Decision

Record SwiGLU as a promising numerical result, not a proven output-quality
improvement. It lowers best validation loss by 0.0785 nats per BPE token without
raising this surface-copying metric, but the five fixed samples do not provide a
clear qualitative advantage. Retain the ReLU learned-absolute-position parent as
the baseline until a follow-up experiment gives stronger quality evidence. This
is one seed with five sampled validation batches per evaluation, not a general
claim about SwiGLU.
