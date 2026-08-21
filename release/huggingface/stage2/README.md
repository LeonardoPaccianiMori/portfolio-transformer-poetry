---
license: cc-by-nc-4.0
language:
  - it
library_name: transformers
pipeline_tag: text-generation
base_model: LPM93/teaching-transformers-classical-italian-sonnets
tags:
  - minerva
  - italian-poetry
  - research
---

# Minerva 7B Classical Italian Poetry — Stage 2

This is the selected second full-weight BF16 adaptation stage from *Teaching
Transformers to Write Classical Italian Sonnets*. It continues from the exact
selected Stage-1 state in the `stage1` subfolder of this repository, which
itself begins with the existing Minerva 7B parent. The model was **not trained
from scratch**.

Stage 2 added 760 optimizer updates and 24,903,680 target tokens. Its
incremental mixture was 75% historical non-sonnet poetry, 20% Stage-1
historical replay, and 5% PAISÀ modern-preservation replay. The published
lineage is cumulative through Stage 1 and Stage 2. The selected research state
identity is
`75817039c2392daac314d9f3365b4c0e1a7b6a5bdab33cf5f95d39ec1ee8397d`.

## Intended purpose

The weights are intended primarily for
non-commercial research transparency, inspection, reproducibility, evaluation,
and independent verification. This is an intended-purpose statement, not an
additional license restriction. CC BY-NC 4.0 permits other uses that satisfy
its own NonCommercial definition.

## Evaluation boundary

This intermediate stage passed the project's declared adaptation and
preservation gates; it is not presented as a good sonnet generator. The later
AI-judged preference branch failed human/AI calibration at **12/20** and is not
human-aligned. In the final blind review, both the later Stage-3 and DPO systems
produced **0/100** strict-good outputs. Those downstream results bound the
project claim; they are not evaluations of Stage 2 itself.

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
