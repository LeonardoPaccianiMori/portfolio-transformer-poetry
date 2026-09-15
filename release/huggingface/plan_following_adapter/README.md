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

# Plan-Following Adapter

This rank-16 PEFT LoRA adapter is part of the 2026-09 follow-up that
made the form objective measurable without repair. Merge order:
`stage3` -> `dpo_adapter` -> `plan_following_adapter`. The base weights are in the `stage3` subfolder of this
repository and the intermediate adapters are in the sibling subfolders named in
`lineage.json`. The subfolder contains no base weights.

## Training

- 15,476 training examples, 814 validation
  examples, 968 optimizer updates, batch 8 x 2 gradient accumulation,
  seed 11413.
- Adapter tensors SHA-256: `bfffd57e6e2b92ab47adffd7acd33fe674b506489b7f86570be259f8a564970a`.

## Measured outcome

| Metric | Value |
| --- | --- |
| key_match_probe | 0.729 |
| planned_key_match | 0.711 |
| mismatched_key_match | 0.734 |
| control_key_match | 0.858 |
| planned_scheme_compliance | 0.0042 |

## Intended purpose and limits

Intended primarily for non-commercial research transparency, inspection,
reproducibility, evaluation, and verification. This is an intended-purpose
statement, not an additional license restriction.

The checker behind these numbers measures metre, rhyme, rhyme scheme, and
fourteen-line structure. It does not measure grammar, meaning, or literary
quality. Coherence remains the project's documented limit.

Full evidence: <https://github.com/LeonardoPaccianiMori/portfolio-transformer-poetry>
