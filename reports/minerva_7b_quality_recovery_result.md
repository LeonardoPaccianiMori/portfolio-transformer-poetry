# Minerva 7B Quality-Recovery Diagnostic Result

## Condition Results

| Condition | Grammar | Topic | Collapse | Controlled form | Mean repetition |
| --- | ---: | ---: | ---: | ---: | ---: |
| `untouched_control` | 8/12 | 12/12 | 6/12 | 9/12 | 0.3670 |
| `stage_a_control` | 3/12 | 12/12 | 3/12 | 12/12 | 0.2208 |
| `stage_b_control` | 2/12 | 11/12 | 3/12 | 12/12 | 0.2718 |
| `stage_b_conservative` | 6/12 | 11/12 | 3/12 | 12/12 | 0.2628 |
| `stage_b_low_temperature` | 6/12 | 12/12 | 5/12 | 12/12 | 0.3412 |
| `stage_b_anti_repeat` | 6/12 | 12/12 | 0/12 | 12/12 | 0.1154 |
| `stage_b_nucleus` | 3/12 | 12/12 | 1/12 | 12/12 | 0.1724 |

## Predeclared Rankings

Ranking order is grammar descending, collapse ascending, topic descending, then repeated-character 4-gram ratio ascending.

Lineage: `untouched_control` > `stage_a_control` > `stage_b_control`

Stage B decoding: `stage_b_anti_repeat` > `stage_b_conservative` > `stage_b_low_temperature` > `stage_b_nucleus` > `stage_b_control`

This validation-only result does not authorize training and does not replace the completed final-test evaluation.
