---
license: cc-by-nc-4.0
language:
  - it
library_name: transformers
pipeline_tag: text-generation
base_model: LPM93/teaching-transformers-classical-italian-sonnets
tags:
  - minerva
  - italian-sonnets
  - research
---

# Minerva 7B Classical Italian Sonnets — Stage 3

This is the selected third full-weight BF16 adaptation stage from *Teaching
Transformers to Write Classical Italian Sonnets*. It continues from the exact
selected Stage-2 state in the `stage2` subfolder of this repository and
ultimately from the existing Minerva 7B parent. It was **not trained from
scratch**.

Stage 3 planned 135 updates; validation selected update **120**, after 1,920
windows and 3,932,160 target tokens. Its incremental mixture was 80% V7
training sonnets, 15% Stage-2 historical replay, and 5% PAISÀ
modern-preservation replay. The selected state identity is
`478d5979e25a78375d7af0434db6a5432678762fac2d142af2d4798dda53a474`.

## Intended purpose

The weights are intended primarily for
non-commercial research transparency, inspection, reproducibility, evaluation,
and independent verification. This is an intended-purpose statement, not an
additional license restriction. CC BY-NC 4.0 permits other uses that satisfy
its own NonCommercial definition.

## Evaluation

The decoder controlled fourteen-line stopping; that does not establish learned
rhyme, metre, stanza structure, grammar, or literary quality. In the final
blind review this model produced **0/100** strict-good outputs. The later
AI-judged DPO branch also produced **0/100**. Human/AI calibration was only
**12/20**, so the preference work remains labelled **AI-judged**, not
human-aligned.

The sealed automatic comparison found a 15.07% surface-screen rate for Stage 3
and 17.60% for DPO. Only the historical-register interval excluded zero in the
blind literary comparison. This model is research evidence, not a reliable
poet or a solved-sonnet system.

## AI contribution

Leonardo Pacciani-Mori conceived and directed the project, made executive
decisions, approved the research plan, reviewed outputs, and sometimes ran GPU
work. Codex 5.5 and later Codex 5.6 Sol substantially assisted research design,
implementation, tests, execution, and analysis. The study was not independently
designed or independently implemented by Leonardo.

## Rights and provenance

Read `RIGHTS_SCOPE.md`, `NOTICE.md`, `TRAINING_CONTENT_SUMMARY.md`, and
`lineage.json` before reuse. The CC BY-NC tag applies only to Leonardo-controlled
rights, if any, in his modifications. It does not replace independently
received Apache-2.0 rights in Minerva or grant rights in training corpora.

Project source and evidence:
<https://github.com/LeonardoPaccianiMori/portfolio-transformer-poetry>
