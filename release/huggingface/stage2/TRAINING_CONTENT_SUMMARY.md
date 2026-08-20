# Aggregate Project-Adaptation Training-Content Summary — Stage 2

This summary is prepared for transparency and possible EU AI Act use. It is
not a claim that a particular regulatory classification applies.

It covers only this project's adaptation stages. It does not reproduce or
independently verify the Minerva parent's upstream pretraining, SFT, or
preference-data summary; see the pinned parent disclosure at
<https://huggingface.co/sapienzanlp/Minerva-7B-instruct-v1.0/tree/d1fc0f0e589ae879c5ac763e0e4206a4d14a3f6d>.

The model cumulatively contains the Minerva parent, selected Stage 1, and
selected Stage 2. Stage 1 consumed 67,665,920 target tokens. Stage 2 added
24,903,680 target tokens across 12,160 windows: 75% historical non-sonnet
poetry, 20% Stage-1 historical replay, and 5% PAISÀ modern-Italian preservation
replay.

Source families were Biblioteca Italiana, Italian Wikisource, Liber Liber,
Project Gutenberg, Oxford Text Archive / ILC, and PAISÀ. The exact incremental
and cumulative source-family target-token totals are in `lineage.json`.

Text was cleaned and encoded under the public data policies, with validation,
test, conditioned-language, and protected evaluation material excluded from
training. No corpus text, document identifiers, token IDs, or private index is
included in this repository.
