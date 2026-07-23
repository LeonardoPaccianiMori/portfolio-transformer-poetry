# Generation Metrics

Generation directory: `outputs/generations/sonnet_corpus_scaling_v5_best`

| Prompt | Chars | Lines | Boundary Markers | Unique Chars | Repeat Ratio | Prompt Kept | Seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| amor | 543 | 14 | 0 | 0.0810 | 0.0741 | yes | 1337 |
| donna | 502 | 14 | 0 | 0.0737 | 0.0741 | yes | 1338 |
| io_son | 540 | 14 | 0 | 0.0667 | 0.0708 | yes | 1339 |
| solo_et_pensoso | 483 | 14 | 0 | 0.0766 | 0.0667 | yes | 1340 |
| line_start | 517 | 14 | 0 | 0.0812 | 0.0564 | yes | 1341 |

## Notes

- `Lines` counts non-empty lines.

- `Boundary Markers` counts `<|endoftext|>` occurrences.

- `Repeat Ratio` is based on repeated character 4-grams by default.

- These are basic automatic checks, not a full quality evaluation.
