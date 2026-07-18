# Pretraining Checkpoint-Neighborhood Evaluation

Each parent run is evaluated at the checkpoint selected by its lowest deterministic validation loss and at the nearest planned checkpoints before and after it. Every batch uses the same five prompts, fixed seeds, temperature 1.0, and a 300-token limit.

## Automatic Diagnostics

| Parent Run | Checkpoint | Validation Selected | Step | Validation Loss | Prompts | Avg Chars | Avg Non-empty Lines | Avg Repeated Character-4-gram Ratio | Avg Unique-Character Ratio | Prompts Preserved |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| larger_swiglu | before | no | 100,000 | 2.5035 | 5 | 773.2 | 1.4 | 0.1651 | 0.0483 | yes |
| larger_swiglu | best | yes | 110,000 | 2.5001 | 5 | 742.0 | 1.2 | 0.1503 | 0.0488 | yes |
| larger_swiglu | after | no | 125,000 | 2.5029 | 5 | 757.6 | 4.2 | 0.1364 | 0.0518 | yes |
| upper_swiglu | before | no | 175,000 | 2.5074 | 5 | 768.6 | 2.2 | 0.1642 | 0.0503 | yes |
| upper_swiglu | best | yes | 180,000 | 2.4983 | 5 | 760.6 | 1.0 | 0.1790 | 0.0460 | yes |
| upper_swiglu | after | no | 200,000 | 2.5115 | 5 | 754.0 | 2.4 | 0.1439 | 0.0500 | yes |
| max_swiglu | before | no | 175,000 | 2.5206 | 5 | 761.6 | 1.4 | 0.1722 | 0.0450 | yes |
| max_swiglu | best | yes | 180,000 | 2.5164 | 5 | 743.0 | 2.2 | 0.1440 | 0.0520 | yes |
| max_swiglu | after | no | 200,000 | 2.5231 | 5 | 770.8 | 1.8 | 0.1726 | 0.0509 | yes |

## Interpretation Rules

- The validation-selected checkpoint remains the model-selection checkpoint. Neighbor outputs are a stability diagnostic, not a basis for cherry-picking a different checkpoint.
- Repetition is measured as the proportion of repeated character 4-grams within each output, then averaged across the five prompts. Lower values can indicate less local looping, but do not by themselves establish better prose.
- Prompt preservation must be `yes`; otherwise the generation procedure is invalid for that batch.
- These automatic measurements must be read together with qualitative inspection of the matched outputs. They do not measure grammaticality, historical style, factual consistency, or literary quality.

## Selection Scope

This report evaluates checkpoint stability within each already-trained parent run. It does not compare training cost, training-corpus coverage, or fine-tuned sonnet quality, which remain separate selection criteria.
