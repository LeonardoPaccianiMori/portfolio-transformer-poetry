# Minerva 7B Instruction-Judge Preflight

- Model: `sapienzanlp/Minerva-7B-instruct-v1.0`
- Revision: `d1fc0f0e589ae879c5ac763e0e4206a4d14a3f6d`
- Weights: untouched 4-bit NF4; no adapter and no updates.
- Human-labelled validation controls: 56.
- Parsed responses: 56/56.
- Final-test material used: no.

## Gate Checks

| Check | Value | Required | Status |
| --- | ---: | ---: | --- |
| Parseable responses | 1.0000 | >= 0.9800 | pass |
| Grammar AUROC | 0.3901 | >= 0.7500 | fail |
| Topic AUROC | 0.3148 | >= 0.7000 | fail |
| Non-collapse AUROC | 0.4031 | >= 0.7500 | fail |
| Human ordinal concordance | 0.4676 | >= 0.6500 | fail |

## Decisions

- Complete judge gate: **fail**.
- Remote FP16 confirmation: **not authorized**.
- DPO, GRPO, and additional training: **not authorized by this preflight**.

A passing preflight would establish only agreement with this bounded human-labelled control set. Metre, rhyme, literary quality, and broader generalization remain separate requirements.
