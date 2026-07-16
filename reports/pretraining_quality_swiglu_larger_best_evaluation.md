# Pretraining Quality Evaluation: Larger SwiGLU Parent

This evaluation covers fixed-prompt generations from
`runs/pretraining_quality_swiglu_larger_200k_001/best_validation.pt`, selected
at step 110,000 by deterministic full-holdout validation loss (2.5001).

## Generation Setup

| Setting | Value |
| --- | --- |
| Prompts | 5 fixed project prompts |
| Base seed | 1337 |
| Temperature | 1.0 |
| Maximum new tokens | 300 |
| Checkpoint | Best validation, step 110,000 |
| Model role | Historical-Italian prose parent; not sonnet-specialized |

The generated files remain local under
`outputs/generations/pretraining_quality_swiglu_larger_best/`. The public
automatic-metrics and review-template artifacts are
`reports/generation_metrics_pretraining_quality_swiglu_larger_best.md` and
`reports/qualitative_review_pretraining_quality_swiglu_larger_best.md`.

## Automatic Results

All five prompts retained their prompt text and reached the configured token
limit. The generations averaged 742 characters. Their repeated character
4-gram ratio averaged 0.1503, ranging from 0.1126 to 0.1813. This indicates
noticeable local repetition, but not simple collapse into one repeated phrase.

## Qualitative Assessment

The model consistently produces extended text with plausible early-modern
Italian lexical and syntactic texture. It also produces named people, places,
institutions, and historical or artistic discourse that resemble the broader
prose corpus.

However, coherence remains low. Sentences often begin plausibly but combine
incompatible referents, incomplete grammatical relations, or abrupt topic
changes. Local repetitions such as repeated words and constructions occur, and
the samples do not maintain a sustained argument or narrative. This is a real
improvement over the earlier incoherent parent-model samples, but it is not yet
realistic literary prose.

## Memorization Scope

No memorization conclusion is recorded for this run. The repository's current
memorization checker compares generations against the sonnet fine-tuning
corpus, while this parent was trained on the separate broader Italian corpus.
Using that checker here would be misleading. A future broader-corpus
memorization check must compare against the 33 actual pretraining sources.

## Decision

Retain this run as the 33.7M-parameter quality-parent reference. Its selected
checkpoint and evaluation settings will be reused for the upper and max
capacity comparisons. The next scheduled run is the 59.8M-parameter upper
SwiGLU parent with the same corrected corpus, optimizer schedule, deterministic
validation policy, and approximately equal token exposure.
