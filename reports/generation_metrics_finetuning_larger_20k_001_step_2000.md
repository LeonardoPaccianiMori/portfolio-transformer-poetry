# Generation Metrics

Generation directory: `outputs/generations/finetuning_larger_20k_001_step_2000`

| Prompt | Chars | Lines | Boundary Markers | Unique Chars | Repeat Ratio | Prompt Kept | Seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| amor | 573 | 14 | 0 | 0.0698 | 0.1053 | yes | 1337 |
| donna | 597 | 14 | 0 | 0.0720 | 0.1010 | yes | 1338 |
| io_son | 544 | 14 | 0 | 0.0717 | 0.1128 | yes | 1339 |
| solo_et_pensoso | 536 | 14 | 0 | 0.0597 | 0.1144 | yes | 1340 |
| line_start | 521 | 14 | 0 | 0.0729 | 0.1023 | yes | 1341 |

## Notes

- `Lines` counts non-empty lines.

- `Boundary Markers` counts `<|endoftext|>` occurrences.

- `Repeat Ratio` is based on repeated character 4-grams by default.

- These are basic automatic checks, not a full quality evaluation.
