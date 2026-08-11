# Minerva 7B V7 Encoded Data And Stage Plan

Status: **ACTIVE_VERIFIED**.

Two independent local builds produced the same content identity. The
canonical broader roles and V7 training sonnets are active for local
training-data use; this does not authorize a GPU benchmark or training run.

## Encoded Scale

| Measurement | Value |
| --- | ---: |
| Documents | 26,935 |
| Tokens | 180,769,204 |
| EOS boundaries | 26,935 |
| Shards | 30 |
| Encoded bytes | 723,076,816 |

## Broader Validation

| Role | Validation tokens | Fraction | Pass |
| --- | ---: | ---: | --- |
| historical_general | 581,062 | 1.00% | True |
| historical_non_sonnet_poetry | 188,772 | 1.00% | True |
| nineteenth_century_bridge | 977,942 | 1.00% | True |

## Exact Initial Stage Budgets

| Stage | Budget tokens | 2,048-token windows | Unused primary tail |
| --- | ---: | ---: | ---: |
| stage_1_historical_general | 67,665,920 | 33,040 | 8,444 |
| stage_2_non_sonnet_poetry | 24,903,680 | 12,160 | 10,607 |
| stage_3_sonnets | 4,423,680 | 2,160 | 12,077 |

Training windows pack documents only within the same role and split,
with one EOS boundary. Each 2,049-token source span yields 2,048
next-token targets. Validation windows are fixed and non-overlapping.
Document indexes preserve author/work groups and harmonized epoch
labels so the later sampler can enforce the frozen concentration caps.
This checkpoint materializes token pools and exact whole-window budgets,
not sampled window assignments. The later deterministic sampler must
enforce the frozen broader work/author and sonnet author/epoch ceilings.

## Safety Boundary

V7 validation/test, broader validation, and protected V6 sonnets remain
outside every training pool. Conditioned material is absent. Token shards,
document indexes, and the PAISÀ replay remain local and ignored. No GPU
work starts and no reusable cache is deleted.
