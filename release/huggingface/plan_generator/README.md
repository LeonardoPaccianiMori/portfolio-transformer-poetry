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

# Rhyme-Plan Generator Adapter

This rank-16 PEFT LoRA adapter is part of the 2026-09 follow-up that
made the form objective measurable without repair. Merge order:
`stage3` -> `dpo_adapter` -> `plan_following_adapter` -> `plan_generator`. The base weights are in the `stage3` subfolder of this
repository and the intermediate adapters are in the sibling subfolders named in
`lineage.json`. The subfolder contains no base weights.

## Training

- 22,072 training examples, 450 validation
  examples, 1380 optimizer updates, batch 8 x 2 gradient accumulation,
  seed 11416.
- Adapter tensors SHA-256: `6b6bd9f143959ae2ed45ed7409dc0ca21a4fb17af638268af6fb6c621eac150f`.

## Measured outcome

| Metric | Value |
| --- | --- |
| valid_plans_240 | 221/240 (0.921) |
| valid_plans_960_temp04 | 891/960 (0.928) |
| parsed_rate | 0.9917 |

## Intended purpose and limits

Intended primarily for non-commercial research transparency, inspection,
reproducibility, evaluation, and verification. This is an intended-purpose
statement, not an additional license restriction.

The checker behind these numbers measures metre, rhyme, rhyme scheme, and
fourteen-line structure. It does not measure grammar, meaning, or literary
quality. Coherence remains the project's documented limit.

Full evidence: <https://github.com/LeonardoPaccianiMori/portfolio-transformer-poetry>
