# Pretraining Checkpoint-Neighborhood Evaluation

Each parent run is evaluated at the checkpoint selected by its lowest deterministic validation loss and at any predeclared comparison checkpoints. Every batch uses the same five prompts, fixed seeds, temperature 1.0, and a 300-token limit.

## Automatic Diagnostics

| Parent Run | Checkpoint | Comparison Role | Validation Selected | Step | Validation Loss | Prompts | Avg Chars | Avg Non-empty Lines | Avg Repeated Character-4-gram Ratio | Avg Unique-Character Ratio | Prompts Preserved |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| historical_v2_xxl | best_validation | selected_parent | yes | 18,000 | 2.6694 | 5 | 783.2 | 1.0 | 0.1487 | 0.0491 | yes |
| historical_v2_xxl | final_overfit_contrast | final_overfit_contrast | no | 100,000 | 5.0911 | 5 | 784.8 | 3.0 | 0.1843 | 0.0487 | yes |

## Interpretation Rules

- The validation-selected checkpoint remains the model-selection checkpoint. Other planned outputs are diagnostics, not a basis for cherry-picking a different checkpoint.
- `Comparison Role` states whether a batch is the selected parent, a retained neighbor, or another explicitly labelled contrast such as an overfitted final model.
- Repetition is measured as the proportion of repeated character 4-grams within each output, then averaged across the five prompts. Lower values can indicate less local looping, but do not by themselves establish better prose.
- Prompt preservation must be `yes`; otherwise the generation procedure is invalid for that batch.
- These automatic measurements must be read together with qualitative inspection of the matched outputs. They do not measure grammaticality, historical style, factual consistency, or literary quality.

## Selection Scope

This report evaluates checkpoint stability within each already-trained parent run. It does not compare training cost, training-corpus coverage, or fine-tuned sonnet quality, which remain separate selection criteria.
