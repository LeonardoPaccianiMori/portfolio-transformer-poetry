---
license: cc-by-nc-4.0
language:
  - it
library_name: peft
pipeline_tag: text-generation
base_model: LPM93/teaching-transformers-classical-italian-sonnets
tags:
  - minerva
  - italian-sonnets
  - peft
  - lora
  - research
---

# Single-Model Plan-and-Poem Adapter

This rank-16 PEFT LoRA adapter is part of the 2026-09 follow-up that
made the form objective measurable without repair. Merge order:
`stage3` -> `dpo_adapter` -> `plan_following_adapter` -> `plan_generator` -> `joint_single_model`. The base weights are in the `stage3` subfolder of this
repository and the intermediate adapters are in the sibling subfolders named in
`lineage.json`. The subfolder contains no base weights.

## Training

- 17,337 training examples, 353 validation
  examples, 1084 optimizer updates, batch 4 x 4 gradient accumulation,
  seed 11418.
- Adapter tensors SHA-256: `8430ae0856fb5932b92b1edf7de5b8dbd9935fa03fcb8e1d3a0e2f64b290110d`.

## Measured outcome

| Metric | Value |
| --- | --- |
| valid_plans | 0.8542 |
| composed_scheme_validity | 0.5719 |
| accepted_lines | 7.7/14 |

## Intended purpose and limits

Intended primarily for non-commercial research transparency, inspection,
reproducibility, evaluation, and verification. This is an intended-purpose
statement, not an additional license restriction.

The checker behind these numbers measures metre, rhyme, rhyme scheme, and
fourteen-line structure. It does not measure grammar, meaning, or literary
quality. Coherence remains the project's documented limit.

Full evidence: <https://github.com/LeonardoPaccianiMori/portfolio-transformer-poetry>
