# Aggregate Project-Adaptation Training-Content Summary — Plan-Follower v2 Adapter

This summary is prepared for transparency and possible EU AI Act use. It is
not a claim that a particular regulatory classification applies.

It covers only this project's adaptation work. It does not reproduce or
independently verify the Minerva parent's upstream pretraining, SFT, or
preference-data summary; see the pinned parent disclosure at
<https://huggingface.co/sapienzanlp/Minerva-7B-instruct-v1.0/tree/d1fc0f0e589ae879c5ac763e0e4206a4d14a3f6d>.

Parent chain: `stage3` -> `dpo_adapter` -> `plan_following_adapter` -> `plan_follower_v2`.

The adapter writes a poem to a supplied plan. It was fine-tuned on 324
cards built from the project's own valid plan-plus-poem generations (the plan
model's output, filtered by the checker and a failed-line limit), so it learns
the distribution of model-written plans rather than only corpus plans.
Training ran for 160 optimizer updates (8 epochs).

Measured on the frozen 960-plan grid, the poem was scheme-valid without
repair in 0.7929 of cases given a valid plan, against 0.5178 for the original
merged plan-follower; the composed rate rose from 0.3646 to 0.5583.
