# Task-Format Acceptance Controls

Generation directory: `outputs/generations/minerva_3b_v5_fixed_comparison/base`

## Automatic Result

- Controlled prompt/form outputs: **15/20** (required: at least 18/20).

- Automatic control gate: **fail**.

## Per-Output Controls

| Output | Author | Lines | Prompt Exact | Markers Hidden | 14-Line Form | Automatic Pass | Stop |
| --- | --- | --- | --- | --- | --- | --- | --- |
| giacomo_madonna__seed_1337 | Giacomo da Lentini | 14 | yes | yes | yes | yes | target_lines |
| giacomo_madonna__seed_1338 | Giacomo da Lentini | 14 | yes | yes | yes | yes | target_lines |
| dante_di_cio__seed_1337 | Dante Alighieri | 14 | yes | yes | yes | yes | target_lines |
| dante_di_cio__seed_1338 | Dante Alighieri | 14 | yes | yes | yes | yes | target_lines |
| cino_roma_superba__seed_1337 | Cino da Pistoia | 14 | yes | yes | yes | yes | target_lines |
| cino_roma_superba__seed_1338 | Cino da Pistoia | 14 | yes | yes | yes | yes | target_lines |
| cecco_chi_vol__seed_1337 | Cecco Angiolieri | 7 | yes | yes | no | no | max_new_tokens |
| cecco_chi_vol__seed_1338 | Cecco Angiolieri | 14 | yes | yes | yes | yes | max_new_tokens |
| cavalcanti_amore_lagia__seed_1337 | Guido Cavalcanti | 14 | yes | yes | yes | yes | target_lines |
| cavalcanti_amore_lagia__seed_1338 | Guido Cavalcanti | 14 | yes | yes | yes | yes | target_lines |
| petrarca_successor__seed_1337 | Francesco Petrarca | 14 | yes | yes | yes | yes | target_lines |
| petrarca_successor__seed_1338 | Francesco Petrarca | 3 | yes | yes | no | no | max_new_tokens |
| stampa_meste_rime__seed_1337 | Gaspara Stampa | 5 | yes | yes | no | no | max_new_tokens |
| stampa_meste_rime__seed_1338 | Gaspara Stampa | 14 | yes | yes | yes | yes | target_lines |
| colonna_gravosi_pensier__seed_1337 | Vittoria Colonna | 8 | yes | yes | no | no | max_new_tokens |
| colonna_gravosi_pensier__seed_1338 | Vittoria Colonna | 14 | yes | yes | yes | yes | target_lines |
| andreini_alta_sorte__seed_1337 | Isabella Andreini | 14 | yes | yes | yes | yes | target_lines |
| andreini_alta_sorte__seed_1338 | Isabella Andreini | 14 | yes | yes | yes | yes | target_lines |
| alfieri_cessar_mai__seed_1337 | Vittorio Alfieri | 14 | yes | yes | yes | yes | target_lines |
| alfieri_cessar_mai__seed_1338 | Vittorio Alfieri | 6 | yes | yes | no | no | max_new_tokens |

## Remaining Acceptance Evidence

- The qualitative review must assess grammatical Italian, seven-line topic/argument continuity, and severe repetition/collapse.

- The memorization report must show zero high-risk outputs.

- Automatic form control is decoder-enforced and must not be claimed as learned metre or rhyme.
