# A2 Packet Review Outcome v1

Date: 2026-09-13

Reviewer: primary agent (`deepseek-v4.1-flash`), read-only diagnostic review.
Leonardo delegated the owner review of the A2 packet on 2026-09-13.

Scope: the 54 metre-flagged lines and the 11 scheme-mismatch lines in
`reports/sonnet_prosody_review_packet_v1.csv`. The 35 sample lines are
unflagged and were not the subject of the review.

Method: for every flagged line, the checker's word-level syllable counts,
boundary merges, and rescue sites were recomputed with the checker internals.
Each line was compared with the standard scansion of its canonical Dante or
Petrarch text. Every flagged line comes from a poem known to be
hendecasyllabic, so a flag is an error only when no documented alternative
reading can scan the line.

## Metre flags

Result: 54 of 54 flags are conservative (the checker reports uncertainty, never
a definite failure). No definite checker error was found. Every line admits at
least one standard alternative reading. The reason frequencies are not
mutually exclusive:

| Reason | Lines |
|---|---:|
| `hiatus_possible` | 41 |
| `sdrucciolo_possible` | 24 |
| `dialefe_possible` | 13 |
| `synaeresis_possible` | 9 |

Confirmed reading families:

- Dialefe after a stressed final syllable, for example "cui | essenza",
  "Già | eran", "Chi | udisse", "Dì | al", "tu | ardi", "noi | et".
- Hiatus in vowel sequences such as `-ia`, `-io`, `-ua`, `-ea`, for example
  "via", "sgradia", "sentia", "partia", "costui", "paura", "desio", "mia",
  "tua".
- Synaeresis of archaic sequences, for example "avea", "vedea", "parea",
  "devea", "movea", "meo", "pregh'eo", "l'aere".

Residual soft cases where the rescue reading is documented but less common:
`dante_lv_dante_un_sospiro_messaggier_del_core` lines 3 and 5,
`dante_lxxxiv_parole_mie_che_per_lo_mondo_siete` line 2, and
`petrarca_che_fai_che_pensi_che_pur_dietro_guardi` lines 2 and 6. The flags
remain acceptable and conservative; no definite error is demonstrated.

## Reason-set limitation

The checker's ambiguity model counts synaeresis sites only for adjacent
open-vowel pairs. It does not model a triphthong merge. Example:
`dante_lxxiii_chi_udisse_tossir_la_mal_fatata` line 8, "merzé del copertoio
c'ha cortonese". The plain count is 12; the standard rescue is a synaeresis of
"copertoio" from four to three syllables, but the checker reports only
`sdrucciolo_possible`. The uncertain verdict is unchanged, but the reason list
is incomplete. A future refinement can extend `_within_word_ambiguity`; the
definite verdicts are not affected.

## Rhyme scheme mismatch

The only exact scheme mismatch is
`dante_xvi_amore_e_l_cor_gentil_sono_una_cosa`: expected
`ABABABABCDECDE`, exact observed `ABABABABCDEFDE`. The third-tercet lines end
"poi" and "costui". This is the documented Sicilian `o`/`u` rhyme equivalence,
not an error. The soft key matches the expected scheme, and the soft scheme
matches 30 of 30 poems. No other poem has a scheme mismatch.

## Conclusion

The checker's definite verdicts and its conservative flags are confirmed. The
A2 gates and the A3 sealed-output conclusions stand. The checker status moves
from pending owner review to reviewed, with the reason-set limitation recorded
above and the measured definite coverage limit (87.1% on the ground truth)
still in force.

Verification: `python3 scripts/validate_sonnet_prosody.py` and
`python3 scripts/score_sealed_sonnet_prosody.py`.
