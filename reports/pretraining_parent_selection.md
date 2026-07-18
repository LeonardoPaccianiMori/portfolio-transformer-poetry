# Pretraining Parent Selection

## Decision

Select `pretraining_quality_swiglu_larger_200k_001/best_validation.pt` as the
parent checkpoint for the next sonnet fine-tuning experiment. It is the
33,671,312-parameter SwiGLU model selected at step 110,000 by its deterministic
full-holdout validation loss of 2.5001.

## Evidence

All three candidates used the same corrected 33-source Italian corpus,
8,000-token BPE tokenizer, 512-token context, warmup-cosine training schedule,
approximately 204.8M training-token exposures, deterministic validation policy,
and fixed-prompt decoding protocol.

| Parent | Parameters | Selected Step | Best Deterministic Validation Loss | Selected-Batch Repeated Character-4-gram Ratio |
| --- | ---: | ---: | ---: | ---: |
| Larger SwiGLU | 33,671,312 | 110,000 | 2.5001 | 0.1503 |
| Upper SwiGLU | 59,807,900 | 180,000 | 2.4983 | 0.1790 |
| Max SwiGLU | 97,729,856 | 180,000 | 2.5164 | 0.1440 |

The upper model's 0.0018 lower validation loss than the larger model is too
small to outweigh its worse fixed-prompt repetition and the qualitative evidence
of more malformed fragments and semantic drift. The max model is worse on
validation loss and also shows degraded qualitative output. The larger model is
therefore the most defensible tradeoff between validation performance,
generation behavior, parameter efficiency, and training cost.

## Neighborhood Robustness

Each parent was also generated from its planned before/best/after checkpoint
neighborhood. All nine batches preserved all five prompts. Within a run,
repetition ratios change noticeably even where validation losses are close. For
example, the larger model's before, selected, and after batches have ratios
0.1651, 0.1503, and 0.1364 respectively. This confirms that sampled text is
not stable enough for an alternative checkpoint to be chosen from one favorable
batch. The validation-selected checkpoint remains the selection checkpoint.

The full automatic table and interpretation rules are in
`reports/pretraining_checkpoint_neighborhoods.md`.

## Scope And Next Experiment

This decision selects a parent model, not a finished prose model. Its generated
text has plausible historical-Italian surface texture but remains semantically
incoherent. No further capacity increase is justified by these experiments.

The next scheduled experiment is sonnet fine-tuning from this exact checkpoint
with the existing deterministic sequential-window validation and early-stopping
protocol. It must use the parent BPE tokenizer and architecture unchanged, then
be compared with the earlier sonnet-control baseline under the same generation
prompts and decoding settings.
