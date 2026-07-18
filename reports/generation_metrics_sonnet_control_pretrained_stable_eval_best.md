# Generation Metrics

Generation directory: `outputs/generations/sonnet_control_pretrained_stable_eval_best`

| Prompt | Chars | Lines | Boundary Markers | Unique Chars | Repeat Ratio | Prompt Kept | Seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| amor | 532 | 14 | 0 | 0.0733 | 0.0870 | yes | 1337 |
| donna | 594 | 14 | 0 | 0.0589 | 0.1117 | yes | 1338 |
| io_son | 581 | 14 | 0 | 0.0671 | 0.0952 | yes | 1339 |
| solo_et_pensoso | 482 | 14 | 0 | 0.0788 | 0.0772 | yes | 1340 |
| line_start | 563 | 14 | 0 | 0.0693 | 0.1143 | yes | 1341 |

## Notes

- `Lines` counts non-empty lines.

- `Boundary Markers` counts `<|endoftext|>` occurrences.

- `Repeat Ratio` is based on repeated character 4-grams by default.

- These are basic automatic checks, not a full quality evaluation.
