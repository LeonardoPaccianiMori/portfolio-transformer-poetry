# Minerva 3B Judge Gate Result

- Model: `sapienzanlp/Minerva-3B-base-v1.0`
- Revision: `129ae5366bae3611a1c9f8c68606c38b7de8b055`
- Weights: untouched FP16; no adapter and no parameter updates.
- Score: negative mean continuation NLL conditioned on the opening line.
- Validation triplets: 8.
- Blinded human-labelled controls: 56.
- Final-test material used: no.
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU with FP16 CPU layer offload.
- Peak CUDA reservation: 4484.0 MiB.

## Gate Checks

| Check | Value | Required | Status |
| --- | ---: | ---: | --- |
| Genuine above corrupted | 1.0000 | >= 0.8750 | pass |
| Genuine above generated | 0.0000 | >= 0.7500 | fail |
| Generated above corrupted | 1.0000 | >= 0.6250 | pass |
| Grammar AUROC | 0.8558 | >= 0.7000 | pass |
| Non-collapse AUROC | 0.1241 | >= 0.6500 | fail |
| Human ordinal concordance | 0.4003 | >= 0.6500 | fail |

## Diagnostic Mean NLL

Lower NLL means Minerva assigns higher likelihood.

| Group | Mean NLL |
| --- | ---: |
| Genuine validation sonnets | 3.8121 |
| From-scratch generated controls | 3.4274 |
| Word-order corruptions | 5.1244 |
| Human grammar: yes | 1.9808 |
| Human grammar: no | 2.6856 |
| Human collapse: yes | 2.0091 |
| Human collapse: no | 2.7601 |

## Decision

**Fail.** At least one predeclared check fails, so DPO and GRPO remain unauthorized under the recorded exit policy.

Authentic sonnets may overlap Minerva's external pretraining data. The human-labelled generated controls are therefore mandatory evidence, not an optional diagnostic.

The judge score measures model likelihood, not metre, rhyme, historical authenticity, or complete human preference. Those controls remain separate in any authorized post-training recipe.
