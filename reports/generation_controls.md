# Generation Controls Report

Checkpoint: `runs/transformer_context768_scaled_001`

Uncontrolled generation directory: `outputs/generations/transformer_context768_scaled_001`

Controlled generation directory: `outputs/generations/transformer_context768_scaled_001_controlled`

## Controls Added

The generation code now supports:

- `temperature`, which scales logits before sampling;
- `top_k`, which limits sampling to the most likely next tokens;
- `stop_text`, which stops generation when decoded text contains a target string;
- `target_lines`, which stops generation after the target number of non-empty lines has been completed.

For the controlled batch, settings were:

- `temperature=1.0`
- `top_k=None`
- `stop_text="<|poem_end|>"`
- `target_lines=14`
- `max_new_tokens=900`
- `seed=1337`

## Automatic Comparison

| Prompt | Uncontrolled Lines | Controlled Lines | Uncontrolled Chars | Controlled Chars | Controlled Stop Reason |
| --- | --- | --- | --- | --- | --- |
| amor | 27 | 14 | 904 | 476 | target_lines |
| donna | 26 | 14 | 905 | 459 | target_lines |
| io_son | 28 | 14 | 906 | 469 | target_lines |
| solo_et_pensoso | 26 | 14 | 915 | 539 | target_lines |
| line_start | 27 | 14 | 906 | 487 | target_lines |

The controlled generation batch fixes the most visible structural problem from the first evaluation: every sample now stops at 14 non-empty lines.

The controlled batch does not improve language quality by itself. The generated words are still mostly malformed, because decoding controls constrain output shape but do not change what the trained model has learned.

## Memorization Comparison

The controlled batch remains low risk under the nearest-neighbor memorization proxy.

All five controlled samples have:

- `0.0000` 40-character containment against the training split;
- longest common substrings between 9 and 15 characters;
- `low` risk labels.

## Interpretation

Generation controls are useful for evaluation because they make outputs more comparable. The original uncontrolled batch mixed model quality with output-length problems. The controlled batch isolates the next failure more clearly: the model can be forced into a 14-line shape, but it still lacks stable word formation and semantic coherence.

The next modeling decision should focus on quality, not line count. The most useful next implementation step is to add decoding experiments with `temperature` and `top_k`, then compare whether lower-randomness sampling improves readability. If decoding alone is not enough, the project should proceed to the planned BPE tokenizer track.
