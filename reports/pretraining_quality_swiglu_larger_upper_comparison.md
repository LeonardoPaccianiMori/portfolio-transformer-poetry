# Controlled Comparison: Larger vs Upper SwiGLU Parents

This comparison uses the same corrected 33-source Italian pretraining corpus,
8,000-token BPE vocabulary, 512-token context, warmup-cosine schedule,
deterministic 337-window validation holdout, fixed prompts, seeds, temperature,
and 300-token generation limit. Both runs processed 204.8M training-token
exposures. The material difference is model capacity and its fitting batch size.

| Measurement | Larger | Upper |
| --- | ---: | ---: |
| Parameters | 33,671,312 | 59,807,900 |
| Batch size | 2 | 1 |
| Training steps | 200,000 | 400,000 |
| Best-validation step | 110,000 | 180,000 |
| Best deterministic validation loss | 2.5001 | 2.4983 |
| Average generated characters | 742 | 761 |
| Average repeated character-4-gram ratio | 0.1503 | 0.1790 |

## Interpretation

The upper parent improves the best deterministic validation loss by 0.0018
nats per token, about 0.07 percent. This is a real numerical difference on the
same holdout, but it is very small. The five matched generations do not provide
corresponding visible evidence of improved coherence; their average repetition
is higher, and they show more malformed word sequences and semantic drift.

This does not prove that the upper architecture is inferior. Generation quality
is assessed from only five prompts at one temperature and seed sequence, while
the models also differ in batch size. It does show that the modest validation
gain is insufficient evidence to choose the larger model for sonnet fine-tuning
or to assume that increasing capacity will improve output quality on the
17.1M-token corpus.

## Decision

Run the pre-registered 97.7M-parameter max capacity check under the same
corpus-exposure and evaluation protocol. After that run, select a parent model
using validation loss, fixed-prompt behavior, repetition metrics, and the
qualitative evidence together. Do not scale beyond the max model unless the
completed three-model comparison supplies clear evidence that capacity, rather
than corpus size or training policy, is the limiting factor.
