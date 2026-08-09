# Minerva 7B V6 Final Qualitative Review

Selected adapter: epoch 4. Selection was frozen on validation evidence before
these final-test outputs were generated.

`Grammar` requires generally grammatical Italian across the continuation, not
isolated correct phrases. `Topic` requires a recognizable topic or argument for
at least seven lines. `Collapse` means severe repetition or degeneration rather
than ordinary poetic recurrence.

| Output | Grammar | Topic | Collapse | Main observation |
|---|---|---|---|---|
| `giacomo_madonna__seed_1337` | no | yes | yes | Praise remains recognizable, but malformed clauses end in a repeated frame. |
| `giacomo_madonna__seed_1338` | yes | yes | no | Sustained praise of the lady with minor lexical and agreement defects. |
| `dante_di_cio__seed_1337` | no | yes | no | Maintains an answer to the addressee, but dependencies remain malformed. |
| `dante_di_cio__seed_1338` | no | yes | yes | Love and grief persist, but repeated fragments overwhelm sentence structure. |
| `cino_roma_superba__seed_1337` | yes | yes | no | Coherent rebuke of Rome with several historical-style agreement defects. |
| `cino_roma_superba__seed_1338` | no | yes | yes | The Roman-law topic remains, but nearly every line repeats the same frame. |
| `cecco_chi_vol__seed_1337` | no | yes | no | Honor and advantage persist through a long, syntactically unstable sentence. |
| `cecco_chi_vol__seed_1338` | yes | yes | no | A broadly coherent appeal for service and goodwill. |
| `cavalcanti_amore_lagia__seed_1337` | yes | yes | no | Sustained journey and courtly-love narrative with imperfect transitions. |
| `cavalcanti_amore_lagia__seed_1338` | no | yes | no | Narrative continuity survives, but malformed words and clauses prevent a pass. |
| `petrarca_successor__seed_1337` | no | yes | no | Historical narration is recognizable but remains an unresolved sentence. |
| `petrarca_successor__seed_1338` | yes | yes | no | Sustained devotional-love meditation with mostly coherent syntax. |
| `stampa_meste_rime__seed_1337` | no | yes | no | Mourning and poetic address persist, but clauses do not resolve cleanly. |
| `stampa_meste_rime__seed_1338` | yes | yes | no | Coherent meditation on sorrow, song, death, and consolation. |
| `colonna_gravosi_pensier__seed_1337` | yes | yes | no | Sustained conflict between grief, love, and spiritual rest. |
| `colonna_gravosi_pensier__seed_1338` | yes | yes | no | Consistent introspection with minor historical-language irregularities. |
| `andreini_alta_sorte__seed_1337` | no | yes | no | Fortune, virtue, and art remain linked, but the final construction is incomplete. |
| `andreini_alta_sorte__seed_1338` | no | yes | yes | Repeated lists of beauty and goodness dominate the continuation. |
| `alfieri_cessar_mai__seed_1337` | no | yes | no | Love and lament remain coherent, but agreement and sentence closure fail. |
| `alfieri_cessar_mai__seed_1338` | no | yes | yes | Counting-like repeated love formulas replace semantic development. |

## Totals

- Generally grammatical Italian: **8/20**; required at least **12/20**.
- Topic or argument sustained for at least seven lines: **20/20**; required at least **10/20**.
- Severe repetition or generation collapse: **5/20**; required at most **2/20**.
- Exact opening line and controlled 14-line form: **20/20**; required at least **18/20**.
- High-risk training-poem memorization: **0/20**; required **0/20**.

The strict acceptable-quality gate therefore fails on grammar and collapse. The
result is nevertheless materially stronger than the from-scratch branches:
every output follows the requested form and topic, many passages are coherent,
and none copies a long training span. Rhyme and metre were not evaluated and
must not be inferred from the enforced line count.
