# Minerva 7B Instruct Validation Baseline: Automatic Evidence

Generation directory: `outputs/generations/minerva_7b_instruct_validation_v1/instruct`

- Exact-opening controlled forms: **6/8**.
- Outputs reaching the 512-token ceiling: **2/8**.
- Mean repeated character 4-gram ratio: **0.2941**.
- Line count is decoder-enforced and is not evidence of metre or rhyme.
- The parent-quality decision requires the separately recorded human judgments.

| Output | Author | Lines | Exact opening | Controlled form | Stop | Repetition |
| --- | --- | ---: | --- | --- | --- | ---: |
| giacomo_or_come__seed_4242 | Giacomo da Lentini | 14 | yes | yes | target_lines | 0.1714 |
| cavalcanti_anima_mia__seed_4242 | Guido Cavalcanti | 14 | yes | yes | target_lines | 0.3183 |
| dante_due_donne__seed_4242 | Dante Alighieri | 14 | yes | yes | target_lines | 0.1705 |
| petrarca_laura__seed_4242 | Francesco Petrarca | 5 | yes | no | max_new_tokens | 0.4790 |
| stampa_piangete__seed_4242 | Gaspara Stampa | 14 | yes | yes | target_lines | 0.2007 |
| colonna_quando__seed_4242 | Vittoria Colonna | 14 | yes | yes | target_lines | 0.1399 |
| andreini_gia_non_possio__seed_4242 | Isabella Andreini | 13 | yes | no | max_new_tokens | 0.6206 |
| alfieri_improvvisatrice__seed_4242 | Vittorio Alfieri | 14 | yes | yes | target_lines | 0.2527 |
