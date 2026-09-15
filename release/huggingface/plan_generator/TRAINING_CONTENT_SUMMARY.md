# Aggregate Project-Adaptation Training-Content Summary — Rhyme-Plan Generator Adapter

This summary is prepared for transparency and possible EU AI Act use. It is
not a claim that a particular regulatory classification applies.

It covers only this project's adaptation work. It does not reproduce or
independently verify the Minerva parent's upstream pretraining, SFT, or
preference-data summary; see the pinned parent disclosure at
<https://huggingface.co/sapienzanlp/Minerva-7B-instruct-v1.0/tree/d1fc0f0e589ae879c5ac763e0e4206a4d14a3f6d>.

Parent chain: `stage3` -> `dpo_adapter` -> `plan_following_adapter` -> `plan_generator`.

The adapter writes only a rhyme plan: a fourteen-letter scheme plus one
ending word for each of lines 2-14. It was trained on 22,522 cards: 11,264
traces derived from real corpus sonnets (scheme and endings taken from the
poem itself) and 11,258 plans generated deterministically from the rhyme
lexicon for the same openings. The target stops after the plan. Training ran
for 1,380 optimizer updates (one epoch, 5,382 seconds).

On 240 held-out evaluation openings the adapter produced 239 parsed plans
(0.9917), all with valid scheme patterns, of which 221 were fully valid
(0.921) against 0/240 for the unadapted baseline. At sampling temperature 0.4
the same model reached 891 valid plans of 960 (0.928) on the frozen grid.
