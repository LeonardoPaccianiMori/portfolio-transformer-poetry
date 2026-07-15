# Generation Metrics

Generation directory: `outputs/generations/sonnet_control_swiglu_pretrained_fresh_best`

| Prompt | Chars | Lines | Boundary Markers | Unique Chars | Repeat Ratio | Prompt Kept | Seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| amor | 562 | 14 | 0 | 0.0730 | 0.1127 | yes | 1337 |
| donna | 567 | 14 | 0 | 0.0705 | 0.0993 | yes | 1338 |
| io_son | 580 | 14 | 0 | 0.0586 | 0.0589 | yes | 1339 |
| solo_et_pensoso | 481 | 14 | 0 | 0.0728 | 0.0858 | yes | 1340 |
| line_start | 499 | 14 | 0 | 0.0842 | 0.0746 | yes | 1341 |

## Notes

- `Lines` counts non-empty lines.

- `Boundary Markers` counts `<|endoftext|>` occurrences.

- `Repeat Ratio` is based on repeated character 4-grams by default.

- These are basic automatic checks, not a full quality evaluation.
