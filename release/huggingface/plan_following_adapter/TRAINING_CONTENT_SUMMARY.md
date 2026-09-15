# Aggregate Project-Adaptation Training-Content Summary — Plan-Following Adapter

This summary is prepared for transparency and possible EU AI Act use. It is
not a claim that a particular regulatory classification applies.

It covers only this project's adaptation work. It does not reproduce or
independently verify the Minerva parent's upstream pretraining, SFT, or
preference-data summary; see the pinned parent disclosure at
<https://huggingface.co/sapienzanlp/Minerva-7B-instruct-v1.0/tree/d1fc0f0e589ae879c5ac763e0e4206a4d14a3f6d>.

Parent chain: `stage3` -> `dpo_adapter` -> `plan_following_adapter`.

The adapter was trained on 15,476 response-masked examples built from the
V8 training sonnets. Each example pairs the frozen instruction prompt, the
sonnet's own opening line, and the list of ending words of all fourteen lines,
with the poem continuation as the target. Training ran for 968 optimizer
updates (one epoch, 1,601 seconds).

The measured outcome is a diagnostic, not a success claim. A 20-prompt
adherence probe reached 0.729 key match. The three-condition evaluation gave
planned 0.711, mismatched-plan 0.734, and format-control 0.858 key match, so
the model copies whichever ending list is in context; planned scheme
compliance was 0.0042. Line 1 adherence was 0.017 because the opening prefill
fixes that line. The artifact is published because it is the parent of the
successful plan generator, not as a good sonnet writer.
