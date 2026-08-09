# Minerva 7B Parent-Decoding Confirmation Protocol

Decision approved: 2026-08-09. Configuration frozen before confirmation GPU
generation or qualitative scoring.

## Purpose

The validation-only quality-recovery diagnostic found complementary strengths:
untouched Minerva 7B produced the most grammatical outputs, while the Stage B
anti-repeat condition eliminated collapse and preserved fourteen-line control.
This confirmation tests whether anti-repeat decoding can retain the untouched
parent's grammar while repairing its instability. It occurs before any
full-weight continued-pretraining proposal.

No optimizer update occurs. V6 final-test poems and openings remain unavailable.

## Prompt Lock

The confirmation uses 24 previously unused V6 validation poems from 12 authors.
Each author contributes two openings. The period distribution is 6/6/8/2/2
across the thirteenth through eighteenth centuries. The set is disjoint from
the eight-prompt Minerva sanity set, the twelve-prompt recovery diagnostic, and
the ten-prompt final acceptance set. Runtime validation requires each opening
to match its processed validation poem exactly.

## Conditions

The same pinned Minerva 7B Instruct revision is loaded once in 4-bit NF4.
Prompts, seed 4099, 512-token ceiling, chat instruction, and thirteen generated
continuation lines remain fixed.

| Condition | Model state | Temperature | Top-k | Top-p | Repetition penalty |
| --- | --- | ---: | ---: | ---: | ---: |
| `untouched_default` | untouched parent | 0.80 | 50 | 1.00 | 1.00 |
| `untouched_anti_repeat` | untouched parent | 0.70 | 50 | 0.92 | 1.10 |
| `stage_b_anti_repeat` | sonnet epoch 4 | 0.70 | 50 | 0.92 | 1.10 |

The repetition penalty applies only to tokens already generated in the
continuation. The model instruction and supplied opening are not penalized.
Condition-level resume is allowed only when all frozen lineage and decoding
metadata match. The complete set contains 72 outputs.

## Review And Gates

Outputs are condition-blinded before review. Every output receives the existing
binary labels for generally grammatical Italian, seven-line topic continuity,
and severe collapse. Historical spelling and poetic inversion alone are not
grammar failures. Automatic form and repetition measurements are diagnostic,
and all outputs are checked against V6 training poems for high-risk overlap.

A condition passes acceptable quality only if every threshold passes:

| Requirement | Threshold |
| --- | ---: |
| Exact prompt and fourteen-line form | at least 22/24 |
| Generally grammatical Italian | at least 15/24 |
| Seven-line topic continuity | at least 12/24 |
| Severe collapse | at most 2/24 |
| High-risk memorization | exactly 0/24 |

Conditions rank by gate pass, grammar descending, collapse ascending, topic
descending, controlled form descending, then repeated-character 4-gram ratio
ascending. A pass can select a validation candidate for a separately frozen
final confirmation. Failure of all three conditions permits design of a
mixed-corpus full-weight calibration, but does not authorize training by itself.

## Completion Criteria

This checkpoint is complete when all frozen hashes pass, 72 outputs and
metadata are present, the blind review is committed before unblinding, the
automatic and memorization checks are complete, and the predeclared condition
gates and next decision are public.
