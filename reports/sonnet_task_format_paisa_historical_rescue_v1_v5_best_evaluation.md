# PAISA Historical Rescue Task-Format Sonnet Evaluation

## Evaluated Model

- Run: `sonnet_task_format_paisa_historical_rescue_v1_v5_12k_001`
- Selected checkpoint: update 250, `best_validation.pt`
- Parent: PAISA-to-historical V5 sonnet-control checkpoint selected at update 3,250
- Task: given one exact opening line, generate the remaining 13 lines
- Fixed acceptance set: 10 V5 held-out opening lines, seeds 1337 and 1338, 20 outputs total
- Decoding: `temperature=0.8`, `top_k=50`, target 13 continuation lines

## Automatic Controls

- Exact opening-line preservation and controlled 14-line form: **20/20**, exceeding the required 18/20.
- Internal task and end-of-text markers displayed: **0/20**.
- High-risk memorization outcomes: **0/20**. No output shares a 40-character n-gram with the V5 training split, so every longest copied span is strictly shorter than 40 characters.

The form result is decoder-enforced. It is not evidence that the model learned metre or rhyme. The underlying evidence is in `sonnet_task_format_paisa_historical_rescue_v1_v5_best_acceptance.md`, `sonnet_task_format_paisa_historical_rescue_v1_v5_best_metrics.md`, and `sonnet_task_format_paisa_historical_rescue_v1_v5_best_memorization.md`.

## Qualitative Review

The 20 full outputs are retained locally and excluded from the public tree. A
continuation counts as plausibly grammatical only when it remains generally
grammatical rather than merely containing isolated well-formed phrases. Topic
continuity requires a recognizable topic or argument through at least seven
generated lines.

| Output | Grammatical Italian | Seven-Line Topic | Severe Collapse | Main Evidence |
| --- | --- | --- | --- | --- |
| giacomo_madonna__seed_1337 | no | no | yes | Repeated invocations of `mia donna` do not form coherent clause relations. |
| giacomo_madonna__seed_1338 | no | no | yes | Poetic fragments and dialogue markers do not develop a consistent argument. |
| dante_di_cio__seed_1337 | no | no | yes | Malformed lexical and syntactic material prevents sentence-level continuity. |
| dante_di_cio__seed_1338 | no | no | yes | Repeated complaints and broken references never answer the opening premise. |
| cino_roma_superba__seed_1337 | no | no | yes | The political opening turns into incompatible love-language fragments. |
| cino_roma_superba__seed_1338 | no | no | yes | `per voi` and `per me` repetition becomes an anaphoric collapse. |
| cecco_chi_vol__seed_1337 | no | no | yes | Locally poetic images lack stable subject, predicate, and argumentative links. |
| cecco_chi_vol__seed_1338 | no | no | yes | Some grammatical phrases occur, but they do not sustain the opening claim. |
| cavalcanti_amore_lagia__seed_1337 | no | no | yes | Repeated `pianto`, `morte`, and `vita` dominate the continuation. |
| cavalcanti_amore_lagia__seed_1338 | no | no | yes | Corrupted words and unfinished clauses prevent a coherent scene. |
| petrarca_successor__seed_1337 | no | no | yes | The political opening is replaced by disjointed abstract love-language. |
| petrarca_successor__seed_1338 | no | no | yes | Corrupted forms and shifting references prevent semantic continuity. |
| stampa_meste_rime__seed_1337 | no | no | yes | Fragmented images of wind, air, and sorrow never form a sustained thought. |
| stampa_meste_rime__seed_1338 | no | no | yes | Malformed phrases and pronoun drift overwhelm the apparent lament. |
| colonna_gravosi_pensier__seed_1337 | no | no | yes | Repeated `dolore` and `sdegni` replace a developed argument. |
| colonna_gravosi_pensier__seed_1338 | no | no | yes | Familiar poetic vocabulary is assembled into incompatible propositions. |
| andreini_alta_sorte__seed_1337 | no | no | yes | Semantic corruption such as `secchiello de le piume` breaks the opening's theme. |
| andreini_alta_sorte__seed_1338 | no | no | yes | Repeated starter phrases and incomplete clauses block coherence. |
| alfieri_cessar_mai__seed_1337 | no | no | yes | Inconsistent grammar and reference drift follow the opening love address. |
| alfieri_cessar_mai__seed_1338 | no | no | yes | Repeated `tua mercede` produces overt lexical collapse. |

Counts against the predeclared criteria:

- Plausibly grammatical Italian: **0/20**; required at least 12/20.
- Topic or argument sustained for at least seven lines: **0/20**; required at least 10/20.
- Severe repetition or generation collapse: **20/20**; required no more than 2/20.
- High-risk memorization: **0/20**; required 0/20.

## Decision

**Fail.** The PAISA and historical-prose curriculum improved the training data scale and the model reliably follows the task interface, but it did not produce an acceptable classical-Italian sonnet generator under the fixed quality rubric. The automatic form and memorization controls pass; the language and coherence gates fail decisively.

This completes the one permitted data-balanced rescue. Under the recorded exit policy, from-scratch model development ends here. The next model-development phase is the approved Minerva QLoRA comparison; this result does not authorize another from-scratch architecture, optimizer, tokenizer, corpus-mixture, or decoding search.
