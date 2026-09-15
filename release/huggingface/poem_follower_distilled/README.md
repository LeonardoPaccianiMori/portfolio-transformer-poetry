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

# Distilled Poem-Follower Adapter

This rank-16 PEFT LoRA adapter is part of the 2026-09 follow-up that
made the form objective measurable without repair. Merge order:
`stage3` -> `dpo_adapter` -> `plan_following_adapter` -> `plan_follower_v2` -> `poem_follower_distilled`. The base weights are in the `stage3` subfolder of this
repository and the intermediate adapters are in the sibling subfolders named in
`lineage.json`. The subfolder contains no base weights.

## Training

- 1,059 training examples, 44 validation
  examples, 268 optimizer updates, batch 4 x 4 gradient accumulation,
  seed 11420.
- Adapter tensors SHA-256: `502c0d2cd778e0cbb4ccf838c3d6e41d8b376addf1c801e50ebf724ccb33d9b5`.

## Measured outcome

| Metric | Value |
| --- | --- |
| composed_scheme_validity | 0.6917 |
| stage2_poem_validity | 0.7374 |
| accepted_lines | 7.3/14 |
| failed_lines | 0.93 |
| judge_glm_5_2 | 2.917 |
| judge_qwen_3_6_plus | 2.725 |
| baseline_judge_glm_5_2 | 2.875 |
| baseline_judge_qwen_3_6_plus | 2.75 |

## Intended purpose and limits

Intended primarily for non-commercial research transparency, inspection,
reproducibility, evaluation, and verification. This is an intended-purpose
statement, not an additional license restriction.

The checker behind these numbers measures metre, rhyme, rhyme scheme, and
fourteen-line structure. It does not measure grammar, meaning, or literary
quality. Coherence remains the project's documented limit.

Full evidence: <https://github.com/LeonardoPaccianiMori/portfolio-transformer-poetry>
