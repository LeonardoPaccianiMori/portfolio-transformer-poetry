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

# Plan-Follower v2 Adapter

This rank-16 PEFT LoRA adapter is part of the 2026-09 follow-up that
made the form objective measurable without repair. Merge order:
`stage3` -> `dpo_adapter` -> `plan_following_adapter` -> `plan_follower_v2`. The base weights are in the `stage3` subfolder of this
repository and the intermediate adapters are in the sibling subfolders named in
`lineage.json`. The subfolder contains no base weights.

## Training

- 324 training examples, 16 validation
  examples, 160 optimizer updates, batch 4 x 4 gradient accumulation,
  seed 11417.
- Adapter tensors SHA-256: `790f2a021f27e5371f0089ca69404313dd9637963b50e23cbeb35e4cae4a8eba`.

## Measured outcome

| Metric | Value |
| --- | --- |
| poem_valid_given_plan | 0.7929 |
| previous_poem_valid_given_plan | 0.5178 |
| composed_rate | 0.5583 |

## Intended purpose and limits

Intended primarily for non-commercial research transparency, inspection,
reproducibility, evaluation, and verification. This is an intended-purpose
statement, not an additional license restriction.

The checker behind these numbers measures metre, rhyme, rhyme scheme, and
fourteen-line structure. It does not measure grammar, meaning, or literary
quality. Coherence remains the project's documented limit.

Full evidence: <https://github.com/LeonardoPaccianiMori/portfolio-transformer-poetry>
