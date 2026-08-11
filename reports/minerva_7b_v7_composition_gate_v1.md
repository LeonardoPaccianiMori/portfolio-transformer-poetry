# Minerva 7B V7 Token Counts And Composition Gate

Status: **PASS**.

The pinned Minerva tokenizer counted each verified logical document without
adding model wrappers, then accounted for exactly one EOS boundary per unit.
No token IDs or encoded training shards were persisted.

## Scale

| Measurement | Value |
| --- | ---: |
| Logical units | 26,934 |
| Logical characters | 644,027,809 |
| Text tokens | 178,707,493 |
| EOS boundaries | 26,934 |
| Training-accounting tokens | 178,734,427 |
| Modern replay tokens | 2,034,777 |

## Frozen Logical Roles

| Role | Documents | Characters | Training tokens |
| --- | ---: | ---: | ---: |
| historical_general | 1,496 | 210,873,928 | 58,105,538 |
| historical_non_sonnet_poetry | 1,602 | 58,032,412 | 18,877,139 |
| nineteenth_century_bridge | 1,446 | 363,119,974 | 97,763,895 |
| standard_sonnets | 22,390 | 12,001,495 | 3,987,855 |

## Approved Stage Mixtures

| Stage | Component | Target share |
| --- | --- | ---: |
| stage_1_historical_general | historical_general | 85% |
| stage_1_historical_general | nineteenth_century_bridge | 10% |
| stage_1_historical_general | modern_preservation_replay | 5% |
| stage_2_non_sonnet_poetry | historical_non_sonnet_poetry | 75% |
| stage_2_non_sonnet_poetry | stage_1_historical_replay | 20% |
| stage_2_non_sonnet_poetry | modern_preservation_replay | 5% |
| stage_3_sonnets | standard_sonnets_v7_train | 80% |
| stage_3_sonnets | stage_2_historical_replay | 15% |
| stage_3_sonnets | modern_preservation_replay | 5% |

## Concentration Controls

| Dimension | Raw largest share | Ceiling | Reweighting required | Feasible |
| --- | ---: | ---: | --- | --- |
| broader_work | 1.74% | 15.00% | False | True |
| broader_author | 4.00% | 20.00% | False | True |
| sonnet_author | 6.98% | 5.00% | True | True |
| sonnet_epoch | 41.68% | 30.00% | True | True |

The 5% replay component is the deterministic PAISA modern-language sample.
Instruction-following preservation remains a separate fixed evaluation gate;
this report does not mislabel PAISA as instruction-tuning data.

## Safety Boundary

V7 validation/test and protected V6 sonnets are counted only for audit and
remain unavailable to training. Conditioned material remains absent. Corpus
roles stay inactive; no encoded mixture, GPU job, or cache deletion occurs.
