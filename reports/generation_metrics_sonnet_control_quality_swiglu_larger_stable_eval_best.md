# Generation Metrics

Generation directory: `outputs/generations/sonnet_control_quality_swiglu_larger_stable_eval_best`

| Prompt | Chars | Lines | Boundary Markers | Unique Chars | Repeat Ratio | Prompt Kept | Seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| amor | 533 | 14 | 0 | 0.0750 | 0.0906 | yes | 1337 |
| donna | 480 | 14 | 0 | 0.0792 | 0.0860 | yes | 1338 |
| io_son | 524 | 14 | 0 | 0.0763 | 0.0845 | yes | 1339 |
| solo_et_pensoso | 532 | 14 | 0 | 0.0602 | 0.0926 | yes | 1340 |
| line_start | 536 | 14 | 0 | 0.0690 | 0.0694 | yes | 1341 |

## Notes

- `Lines` counts non-empty lines.

- `Boundary Markers` counts `<|endoftext|>` occurrences.

- `Repeat Ratio` is based on repeated character 4-grams by default.

- These are basic automatic checks, not a full quality evaluation.
