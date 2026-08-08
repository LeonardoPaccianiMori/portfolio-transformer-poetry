# Minerva 3B Base Versus QLoRA Evaluation

## Evaluated Systems

- Base: untouched `sapienzanlp/Minerva-3B-base-v1.0`, revision
  `129ae5366bae3611a1c9f8c68606c38b7de8b055`.
- QLoRA: the same frozen 4-bit base with the selected rank-16 adapter from
  epoch 3/update 558 of `minerva_3b_qlora_v5_001`.
- Task: continue one exact held-out opening line with 13 generated lines.
- Fixed set: ten V5 test openings, seeds 1337 and 1338, 20 outputs per system.
- Decoding: `temperature=0.8`, `top_k=50`, 900-token ceiling, special-token
  suppression, and decoder-enforced stopping after 13 completed continuation
  lines.

## Automatic Comparison

| Measurement | Untouched Base | Selected QLoRA |
| --- | ---: | ---: |
| Outputs | 20 | 20 |
| Exact prompt and controlled 14-line form | 15 | 20 |
| Outputs reaching 900-token ceiling | 6 | 0 |
| Mean generated characters | 2,055.8 | 568.9 |
| Median generated characters | 1,773 | 573 |
| Mean repeated character 4-gram ratio | 0.4396 | 0.2340 |
| Median repeated character 4-gram ratio | 0.4474 | 0.2177 |
| High-risk V5 training-copy outcomes | 0 | 0 |

The form result is decoder-enforced and is not evidence of learned metre or
rhyme. The comparison nevertheless shows a real task-adaptation effect: the
adapter makes newline production reliable, strongly reduces uncontrolled
length, and roughly halves the repetition diagnostic.

The untouched base frequently abandons the opening for generic modern prose,
web-page furniture, fabricated reference material, or long repetitive essays.
Its automatic form gate fails at 15/20. Some individual passages are fluent
modern Italian, but they do not perform the requested historical-sonnet task.

## QLoRA Qualitative Review

A continuation counts as plausibly grammatical only when it remains generally
grammatical, not when it merely contains isolated well-formed phrases. Topic
continuity requires one recognizable topic or argument through at least seven
generated lines. Severe collapse includes list loops, phrase loops, or syntax
that degenerates enough to prevent continuation-level interpretation.

| Output | Grammatical Italian | Seven-Line Topic | Severe Collapse | Main Evidence |
| --- | --- | --- | --- | --- |
| giacomo_madonna__seed_1337 | no | yes | yes | The lady's virtue remains the topic, but repeated `far suo`, `vertute`, and broken clause joins dominate the ending. |
| giacomo_madonna__seed_1338 | yes | yes | no | A mostly interpretable praise of the lady and Amor survives throughout, despite awkward constructions. |
| dante_di_cio__seed_1337 | no | yes | no | Will, reason, and the requested answer remain linked, but several clauses are malformed. |
| dante_di_cio__seed_1338 | no | yes | no | Speech and the counterfactual answer remain recognizable, with persistent agreement and reference errors. |
| cino_roma_superba__seed_1337 | no | yes | no | Rome, law, violence, and fortune form a sustained political address, but syntax repeatedly breaks. |
| cino_roma_superba__seed_1338 | no | no | yes | Repeated `a che` and `Mondo` phrases replace a stable proposition. |
| cecco_chi_vol__seed_1337 | no | no | yes | Malformed relations and the repeated `in voi` ending prevent a recoverable argument. |
| cecco_chi_vol__seed_1338 | no | yes | no | Memory, the heart, fault, and divine pardon remain connected, though the Italian is unreliable. |
| cavalcanti_amore_lagia__seed_1337 | no | no | yes | Repeated time phrases give way to incompatible elemental abstractions. |
| cavalcanti_amore_lagia__seed_1338 | no | no | yes | A long inventory of names, clothing, food, and household objects becomes list collapse. |
| petrarca_successor__seed_1337 | no | yes | no | A public figure's glory and celestial light persist, but lexical and syntactic corruption remains substantial. |
| petrarca_successor__seed_1338 | no | no | no | The successor gives way to an unclear Amor narrative and an unfinished temporal conclusion. |
| stampa_meste_rime__seed_1337 | yes | yes | no | The lament addressed to `Madre mia` remains broadly grammatical and thematically stable. |
| stampa_meste_rime__seed_1338 | no | yes | no | Grief, death, poetry, and lost speech remain coherent as a topic, but later clauses and the ending are incomplete. |
| colonna_gravosi_pensier__seed_1337 | no | yes | no | Painful thoughts, Amor, and flight remain linked, with frequent malformed transitions. |
| colonna_gravosi_pensier__seed_1338 | no | yes | no | Burden, moral fault, and human suffering persist, but sentence structure is repeatedly defective. |
| andreini_alta_sorte__seed_1337 | no | yes | no | A religious address to the mother and divine light persists until pronoun/reference failure at the end. |
| andreini_alta_sorte__seed_1338 | no | no | yes | `che fia` and `suo valore` loops overwhelm the opening premise. |
| alfieri_cessar_mai__seed_1337 | no | yes | no | Love, light, desire, and torment remain recognizable, but grammar and addressee references drift. |
| alfieri_cessar_mai__seed_1338 | no | no | yes | `mille` and `fiamme ardenti` repeat into overt lexical collapse. |

Counts against the predeclared acceptance criteria:

- Exact prompt and controlled form: **20/20**; required at least 18/20.
- Plausibly grammatical Italian: **2/20**; required at least 12/20.
- Topic or argument sustained for seven lines: **13/20**; required at least
  10/20.
- Severe repetition or generation collapse: **7/20**; required no more than
  2/20.
- High-risk memorization: **0/20**; required 0/20.

## Decision

**Untouched Base: fail. Selected QLoRA: fail the predeclared acceptable-quality
gate, with substantial improvement.**

QLoRA changes the model from an uncontrolled general-prose continuation system
into a recognizably sonnet-oriented generator. It passes form, topic, and
surface-copying requirements, but it does not yet produce generally reliable
Italian and still collapses in seven outputs. It is therefore the strongest
system in the project so far, but it is not an acceptable realistic-sonnet
generator under the recorded definition.

The next approved checkpoint is the untouched-Minerva judge-validation gate
for the independent DPO and GRPO branches. Judge development must use training
and validation prompts only; none of the final-test outputs assessed here may
be used to select or calibrate the judge protocol.
