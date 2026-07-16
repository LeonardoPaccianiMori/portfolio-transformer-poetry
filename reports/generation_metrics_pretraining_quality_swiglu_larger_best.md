# Generation Metrics

Generation directory: `outputs/generations/pretraining_quality_swiglu_larger_best`

| Prompt | Chars | Lines | Boundary Markers | Unique Chars | Repeat Ratio | Prompt Kept | Seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| amor | 731 | 1 | 0 | 0.0465 | 0.1126 | yes | 1337 |
| donna | 713 | 1 | 0 | 0.0617 | 0.1408 | yes | 1338 |
| io_son | 731 | 2 | 0 | 0.0451 | 0.1813 | yes | 1339 |
| solo_et_pensoso | 745 | 1 | 0 | 0.0564 | 0.1590 | yes | 1340 |
| line_start | 790 | 1 | 0 | 0.0342 | 0.1576 | yes | 1341 |

## Notes

- `Lines` counts non-empty lines.

- `Boundary Markers` counts `<|poem_end|>` occurrences.

- `Repeat Ratio` is based on repeated character 4-grams by default.

- These are basic automatic checks, not a full quality evaluation.
