# Pretraining Quality Evaluation: Upper SwiGLU Parent

This evaluation covers fixed-prompt generations from
`runs/pretraining_quality_swiglu_upper_400k_001/best_validation.pt`, selected
at step 180,000 by deterministic full-holdout validation loss (2.4983).

## Generation Setup

| Setting | Value |
| --- | --- |
| Prompts | 5 fixed project prompts |
| Base seed | 1337 |
| Temperature | 1.0 |
| Maximum new tokens | 300 |
| Checkpoint | Best validation, step 180,000 |
| Model role | Historical-Italian prose parent; not sonnet-specialized |

The generated files remain local under
`outputs/generations/pretraining_quality_swiglu_upper_best/`. The public
automatic-metrics and review-template artifacts are
`reports/generation_metrics_pretraining_quality_swiglu_upper_best.md` and
`reports/qualitative_review_pretraining_quality_swiglu_upper_best.md`.

## Automatic Results

All five prompts retained their prompt text and reached the configured token
limit. The generations averaged 761 characters. Their repeated character
4-gram ratio averaged 0.1790, ranging from 0.1447 to 0.2339. This is higher
than the 0.1503 average for the smaller SwiGLU parent under the same prompts,
seeds, and decoding settings.

## Qualitative Assessment

The upper model continues to produce historical Italian lexical texture and
occasionally learns document-like signals, such as a chapter heading and a
chronicle-style date. However, the fixed samples do not show a clear
improvement in prose quality over the larger parent. They contain malformed
word sequences, incompatible grammatical roles, abrupt semantic drift, and
repeated constructions. The model's slightly better deterministic validation
loss therefore does not by itself establish a better visible generation result.

## Memorization Scope

No memorization conclusion is recorded for this run. The repository's current
memorization checker compares generations against the sonnet fine-tuning
corpus, while this parent was trained on the separate broader Italian corpus.
Using that checker here would be misleading. A future broader-corpus
memorization check must compare against the 33 actual pretraining sources.

## Decision

Retain this run as the 59.8M-parameter capacity comparison. The next planned
experiment is the 97.7M-parameter max parent with matching corpus exposure and
evaluation. The two completed runs show that increased parameter count is not
yet a demonstrated quality gain on this dataset; the max run is the final
planned capacity check, not a reason to continue increasing model size.
