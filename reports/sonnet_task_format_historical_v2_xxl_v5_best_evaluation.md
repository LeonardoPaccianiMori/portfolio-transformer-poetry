# Task-Format Sonnet Evaluation

## Evaluated Model

- Run: `sonnet_task_format_historical_v2_xxl_v5_12k_001`
- Selected checkpoint: update 250, `best_validation.pt`
- Parent: V5 historical-Italian XXL sonnet-control checkpoint selected at update 2,250
- Task: given one exact opening line, generate the remaining 13 lines
- Fixed acceptance set: 10 V5 held-out opening lines, seeds 1337 and 1338, 20 outputs total
- Decoding: `temperature=0.8`, `top_k=50`, target 13 continuation lines

## Automatic Controls

- Exact opening-line preservation and controlled 14-line form: **20/20**, exceeding the required 18/20.
- Internal task and end-of-text markers displayed: **0/20**.
- High-risk memorization outcomes: **0/20**. No output shares a 40-character n-gram with the V5 training split, so every longest copied span is strictly shorter than 40 characters.

The first result is a decoder-enforced control, not evidence that the model has learned metre or rhyme. The underlying evidence is in `sonnet_task_format_historical_v2_xxl_v5_best_acceptance_controls.md`, `generation_metrics_sonnet_task_format_historical_v2_xxl_v5_best_acceptance.md`, and `memorization_checks_sonnet_task_format_historical_v2_xxl_v5_best_acceptance.md`.

## Qualitative Review

The 20 full outputs are retained locally and excluded from the public tree. The
ratings below use the fixed acceptance rubric. A result counts as plausibly
grammatical only if its continuation remains generally grammatical rather than
merely containing isolated well-formed phrases. Topic continuity requires a
recognizable topic or argument through at least seven generated lines.

| Output | Grammatical Italian | Seven-Line Topic | Severe Collapse | Main Evidence |
| --- | --- | --- | --- | --- |
| giacomo_madonna__seed_1337 | no | no | yes | Invented forms and subject/reference drift after the first local phrases. |
| giacomo_madonna__seed_1338 | no | no | yes | Dialogue-like fragments do not combine into a coherent argument. |
| dante_di_cio__seed_1337 | no | no | yes | Broken constructions and malformed lexical material prevent sentence-level coherence. |
| dante_di_cio__seed_1338 | no | no | yes | Repeated woman/subject references and incompatible clauses. |
| cino_roma_superba__seed_1337 | no | no | yes | Invented or corrupted words and no sustained response to the opening question. |
| cino_roma_superba__seed_1338 | no | no | yes | Anaphoric lists begin coherently but dissolve into unrelated love-language fragments. |
| cecco_chi_vol__seed_1337 | no | no | yes | Pronoun and predicate relations repeatedly break across lines. |
| cecco_chi_vol__seed_1338 | no | no | yes | Locally plausible love vocabulary lacks a stable claim or development. |
| cavalcanti_amore_lagia__seed_1337 | no | no | yes | Repeated self-reference and unfinished syntax dominate the continuation. |
| cavalcanti_amore_lagia__seed_1338 | no | no | yes | Fragmented clauses and lexical corruption prevent a sustained scene. |
| petrarca_successor__seed_1337 | no | no | yes | Repetition and malformed questions displace the political opening's topic. |
| petrarca_successor__seed_1338 | no | no | yes | Repeated phrases and invented vocabulary prevent semantic continuity. |
| stampa_meste_rime__seed_1337 | no | no | yes | Several incomplete or malformed lines interrupt the apparent first-person theme. |
| stampa_meste_rime__seed_1338 | no | no | yes | Pronoun drift and repeated function-word patterns overwhelm local syntax. |
| colonna_gravosi_pensier__seed_1337 | no | no | yes | Familiar poetic words are assembled into incompatible clause relations. |
| colonna_gravosi_pensier__seed_1338 | no | no | yes | Repetition and syntactic fragments do not develop the opening's thought. |
| andreini_alta_sorte__seed_1337 | no | no | yes | Extreme repetition of `viso` and malformed statements show collapse. |
| andreini_alta_sorte__seed_1338 | no | no | yes | Isolated images do not form a grammatical or continuous argument. |
| alfieri_cessar_mai__seed_1337 | no | no | yes | The opening love address is followed by inconsistent grammar and references. |
| alfieri_cessar_mai__seed_1338 | no | no | yes | Some local poetic phrasing survives, but it does not sustain coherent syntax or topic. |

Counts against the predeclared criteria:

- Plausibly grammatical Italian: **0/20**; required at least 12/20.
- Topic or argument sustained for at least seven lines: **0/20**; required at least 10/20.
- Severe repetition or generation collapse: **20/20**; required no more than 2/20.
- High-risk memorization: **0/20**; required 0/20.

## Decision

**Fail.** Task-format post-training solved the interface and stopping behavior but did not make the from-scratch model an acceptable classical-Italian sonnet generator. The fixed automatic controls pass; the language and coherence gates fail decisively.

Under the recorded exit policy, this failure activates exactly one final from-scratch rescue: the documented PAISÀ data-balanced pretraining route, followed by the same task-format training and fixed acceptance evaluation. No additional architecture, optimizer, tokenizer, or decoding sweep is authorized before that comparison. After the rescue evaluation, the from-scratch track ends and the project moves to the Minerva QLoRA comparison regardless of the result.
