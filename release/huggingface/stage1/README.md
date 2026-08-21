---
license: cc-by-nc-4.0
language:
  - it
library_name: transformers
pipeline_tag: text-generation
base_model: sapienzanlp/Minerva-7B-instruct-v1.0
tags:
  - minerva
  - historical-italian
  - research
---

# Minerva 7B Classical Italian — Stage 1

This is the selected first full-weight BF16 adaptation stage from *Teaching
Transformers to Write Classical Italian Sonnets*. It begins with
`sapienzanlp/Minerva-7B-instruct-v1.0` at revision
`d1fc0f0e589ae879c5ac763e0e4206a4d14a3f6d`; it was **not trained from
scratch**.

Stage 1 used 2,065 optimizer updates and 67,665,920 target tokens. The mixture
was 85% historical/general Italian, 10% nineteenth-century bridge material,
and 5% PAISÀ modern-preservation replay. These are target-token exposure
shares, not document, example, or corpus-size shares. The selected research
state identity is
`c3aba5b672e8634028477885ad96d7c25c48d2c60cc5597b6da730089212ac39`.

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
project claim; they are not evaluations of Stage 1 itself.

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
