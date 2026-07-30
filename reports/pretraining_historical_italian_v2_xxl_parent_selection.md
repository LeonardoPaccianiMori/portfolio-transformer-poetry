# Historical Italian V2 XXL Parent Selection

## Decision

Select
`runs/pretraining_historical_italian_v2_xxl_accum8_100k_001/best_validation.pt`
from optimizer step 18,000 as the parent checkpoint for the V5 sonnet
fine-tuning stage. Its deterministic sequential-window validation loss was
2.6694.

## Evidence

The completed run used a 234,839,008-parameter causal transformer with a
16,000-token BPE vocabulary, 512-token context, 16 layers, 1,024 embedding
dimensions, 16 attention heads, SwiGLU feed-forward blocks, LayerNorm, and
learned absolute positions. It trained on 17,891,995 unique broader-Italian
training tokens.

| Checkpoint | Role | Step | Validation loss | Fixed prompts | Avg. repeated character-4-gram ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| `best_validation.pt` | selected parent | 18,000 | 2.6694 | 5 | 0.1487 |
| `model.pt` | final overfitting contrast | 100,000 | 5.0911 | 5 | 0.1843 |

All ten outputs preserved their prompts and used the same five prompts, seeds,
temperature 1.0, and 300-token limit. The selected checkpoint's outputs show
some historical Italian prose-like local texture, but they frequently change
topic, combine incompatible entities, and do not sustain coherent arguments.
The final model sometimes appears locally more fluent, but it has materially
worse validation loss, more repeated character 4-grams, and obvious semantic
drift. It is therefore evidence of overfitting, not a competing parent.

The run used `latest_only` checkpoint retention. Its actual near-best interval
checkpoints were discarded, so the final model is explicitly a contrast rather
than a validation-loss neighbor. No result here is a sonnet-quality claim or a
memorization evaluation; those require the downstream sonnet fine-tuning and
fixed acceptance protocol.

## Scope And Next Experiment

This decision selects the parent for the current from-scratch attempt. The next
stage is V5 sonnet fine-tuning with the established deterministic
sequential-window validation and early-stopping protocol, starting only from
the selected step-18,000 checkpoint. The current run's final `model.pt` must
not be used for fine-tuning or generation comparison.

The automatic diagnostics are in
`reports/pretraining_historical_italian_v2_xxl_checkpoint_comparison.md`, and
the full run record is in
`reports/pretraining_historical_italian_v2_xxl_accum8_100k_001.md`.
