# Aggregate Project-Adaptation Training-Content Summary — Stage 1

This summary is prepared for transparency and possible EU AI Act use. It is
not a claim that a particular regulatory classification applies.

It covers only this project's adaptation stage. It does not reproduce or
independently verify the Minerva parent's upstream pretraining, SFT, or
preference-data summary; see the pinned parent disclosure at
<https://huggingface.co/sapienzanlp/Minerva-7B-instruct-v1.0/tree/d1fc0f0e589ae879c5ac763e0e4206a4d14a3f6d>.

The model starts from the separately released Minerva 7B Instruct parent. The
project then consumed 67,665,920 target tokens across 33,040 deterministic
2,048-token windows: 85% historical/general Italian, 10% nineteenth-century
bridge material, and 5% PAISÀ modern-Italian preservation replay.

Source families were Biblioteca Italiana, Italian Wikisource, Liber Liber,
Project Gutenberg, Oxford Text Archive / ILC, and PAISÀ. Material includes
historical prose, literary prose, poetry encountered within broader sources,
and modern Italian web text used only for preservation replay. The exact
source-family target-token totals are in `lineage.json`.

Text was cleaned and encoded under the public data policies, with validation,
test, conditioned-language, and protected evaluation material excluded from
training. No corpus text, document identifiers, token IDs, or private index is
included in this repository.
