# Aggregate Project-Adaptation Training-Content Summary — DPO Adapter

This summary is prepared for transparency and possible EU AI Act use. It is
not a claim that a particular regulatory classification applies.

It covers only this project's adaptation and preference-optimization work. It
does not reproduce or independently verify the Minerva parent's upstream
pretraining, SFT, or preference-data summary; see the pinned parent disclosure
at <https://huggingface.co/sapienzanlp/Minerva-7B-instruct-v1.0/tree/d1fc0f0e589ae879c5ac763e0e4206a4d14a3f6d>.

The adapter depends on the exact selected Stage-3 base and therefore inherits
its cumulative source-family lineage: 67,665,920 Stage-1 target tokens,
24,903,680 Stage-2 target tokens, and 3,932,160 selected Stage-3 target tokens.
Exact family totals are in `lineage.json`.

Preference construction generated 4,096 candidates from 512 V7 training-only
openings. Three AI judges produced decisive majority labels for 534 pairs;
482 prompt-disjoint pairs trained the adapter and 52 were retained for
validation. Human/AI calibration was 12/20 and failed its gate. The experiment
is AI-judged, not human-aligned.

No opening, candidate, preference, vote, annotation, generation, mapping,
optimizer state, RNG state, or corpus text is included in this repository. The
adapter package contains only LoRA tensors, configuration, and aggregate
documentation.
