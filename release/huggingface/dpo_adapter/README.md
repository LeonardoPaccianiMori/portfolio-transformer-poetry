---
license: cc-by-nc-4.0
language:
  - it
library_name: peft
pipeline_tag: text-generation
base_model: LPM93/minerva-7b-classical-italian-sonnets-stage3
tags:
  - minerva
  - italian-sonnets
  - dpo
  - peft
  - research
---

# Minerva 7B Classical Italian Sonnets — AI-Judged DPO Adapter

> **Preparation status:** candidate package only. This artifact has not been
> uploaded or approved for public distribution.

This rank-8 PEFT LoRA adapter attaches only to the exact selected Stage-3 model
at state identity
`478d5979e25a78375d7af0434db6a5432678762fac2d142af2d4798dda53a474`.
The base model is a separate candidate package and was **not trained from
scratch**. This adapter repository would contain no base weights.

The adapter was trained for 61 optimizer updates from 482 training and 52
validation preference pairs. The pairs came from 4,096 Stage-3 candidates and
were selected by three AI judges. Human/AI calibration was only **12/20** and
failed its gate. The result is **AI-judged** preference optimization, not RLHF,
human-calibrated DPO, or human alignment.

## Intended purpose

If later approved and published, the adapter is intended primarily for
non-commercial research transparency, inspection, reproducibility, evaluation,
and independent verification. This is an intended-purpose statement, not an
additional license restriction. CC BY-NC 4.0 permits other uses that satisfy
its own NonCommercial definition.

## Evaluation

On the sealed automatic test, DPO increased the surface-screen rate from
15.07% to 17.60%, a paired change of +2.53 percentage points with a 95%
interval from +0.52 to +4.50 points. Fourteen-line output was decoder-controlled.
In the final blind literary review, Stage 3 and DPO both produced **0/100**
strict-good outputs; only the historical-register interval excluded zero.

The evidence supports a narrow surface/completion improvement, not broad
literary quality. The adapter can reproduce historical-looking but broken or
incoherent text and may preserve unknown overlap from the parent.

## AI contribution

Leonardo Pacciani-Mori conceived and directed the project, made executive
decisions, approved the research plan, reviewed outputs, and sometimes ran GPU
work. Codex 5.5 and later Codex 5.6 Sol substantially assisted research design,
implementation, tests, execution, and analysis. The study was not independently
designed or independently implemented by Leonardo.

## Rights and provenance

Read `RIGHTS_SCOPE.md`, `NOTICE.md`, `TRAINING_CONTENT_SUMMARY.md`, and
`lineage.json` before reuse. The CC BY-NC tag applies only to Leonardo-controlled
rights, if any, in the adapter tensors. It does not license or relicense the
Stage-3 base or any training corpus.

Project source and evidence:
<https://github.com/LeonardoPaccianiMori/portfolio-transformer-poetry>
