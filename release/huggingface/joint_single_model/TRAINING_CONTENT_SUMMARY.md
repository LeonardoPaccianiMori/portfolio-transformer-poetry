# Aggregate Project-Adaptation Training-Content Summary — Single-Model Plan-and-Poem Adapter

This summary is prepared for transparency and possible EU AI Act use. It is
not a claim that a particular regulatory classification applies.

It covers only this project's adaptation work. It does not reproduce or
independently verify the Minerva parent's upstream pretraining, SFT, or
preference-data summary; see the pinned parent disclosure at
<https://huggingface.co/sapienzanlp/Minerva-7B-instruct-v1.0/tree/d1fc0f0e589ae879c5ac763e0e4206a4d14a3f6d>.

Parent chain: `stage3` -> `dpo_adapter` -> `plan_following_adapter` -> `plan_generator` -> `joint_single_model`.

The adapter writes the plan and then the poem in one generation. It was
trained on 17,690 cards: 11,264 traces derived from corpus sonnets and 6,426
pairs from the project's own pipeline runs, repeated six times to balance the
mix. Training ran for 1,084 optimizer updates (1 or 2 epochs depending on the
mix), 7,671 seconds.

On the frozen 960-plan grid the single model produced valid plans in 0.8542 of
cases and a scheme-valid poem without repair in 0.5719, above the 0.50 bar set
for a one-model system. It is slightly below the two-model pipeline (0.6250
with the clean follower, 0.6917 distilled) and is published as a compact
alternative.
