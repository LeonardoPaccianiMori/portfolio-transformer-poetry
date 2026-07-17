# Generation Metrics

Generation directory: `outputs/generations/pretraining_quality_swiglu_upper_best`

| Prompt | Chars | Lines | Boundary Markers | Unique Chars | Repeat Ratio | Prompt Kept | Seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| amor | 747 | 1 | 0 | 0.0469 | 0.2339 | yes | 1337 |
| donna | 783 | 1 | 0 | 0.0524 | 0.1654 | yes | 1338 |
| io_son | 740 | 1 | 0 | 0.0459 | 0.1750 | yes | 1339 |
| solo_et_pensoso | 763 | 1 | 0 | 0.0393 | 0.1447 | yes | 1340 |
| line_start | 770 | 1 | 0 | 0.0455 | 0.1760 | yes | 1341 |

## Notes

- `Lines` counts non-empty lines.

- `Boundary Markers` counts `<|poem_end|>` occurrences.

- `Repeat Ratio` is based on repeated character 4-grams by default.

- These are basic automatic checks, not a full quality evaluation.
