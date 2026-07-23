# Medieval Prose Activation Audit

## Scope And Decision

This audit activates eight Italian prose works as
`pretraining_medieval_v1`, a medieval component for the broader Italian
pretraining corpus. It is not a standalone pretraining dataset. It will be
merged only after the larger corpus composition is explicitly balanced, because
this component is concentrated in Giovanni Boccaccio and Giovanni Villani.

All activated digital editions come from Liber Liber and are licensed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The
repository retains the required attribution, source links, contributor credits,
non-commercial restriction, and share-alike obligation in
[`data/metadata/broader_prose_attribution.md`](../data/metadata/broader_prose_attribution.md)
and the canonical source manifest.

## Activated Sources

| Source | Period | Cleaned characters | Decision |
| --- | --- | ---: | --- |
| *Il Novellino* | late XIII century | 209,278 | Activate. Removed a terminal electronic-edition note, terminal contents, and terminal summary. |
| Giovanni Villani, *Nuova cronica* | 1308-1348 | 2,713,993 | Activate. Period-compatible chronicle prose. |
| Dino Compagni, *Cronica* | 1310-1312 | 221,766 | Activate. Period-compatible Florentine chronicle prose. |
| Boccaccio, *Decameron* | 1349-1353 | 1,511,873 | Activate. Removed the edition credit and publisher line before the primary text. |
| Boccaccio, *Filocolo* | 1336-1338 | 1,212,836 | Activate. Primary prose retained. |
| Boccaccio, *Elegia di Madonna Fiammetta* | 1343-1344 | 336,933 | Activate. Primary prose retained. |
| Boccaccio, *Trattatello in laude di Dante* | c. 1351-1374 | 122,698 | Activate. Primary prose retained. |
| Sacchetti, *Il Trecentonovelle* | c. 1392-1400 | 1,037,193 | Activate with an edition-completeness note. The supplied edition starts its numbered stories at `Novella II`; it must not be described as a complete 300-story collection. |

The committed build contains 7,366,570 cleaned characters and 1,276,885 words.
The exact source-level counts, archive URLs, first/last text samples, and
cleaning rules are in the committed
[`pretraining_medieval_v1` build report](../reports/pretraining_medieval_v1_build_report.json).

## Composition Check

The component's cleaned-character shares are Boccaccio 43.23%, Giovanni
Villani 36.84%, Sacchetti 14.08%, Dino Compagni 3.01%, and *Novellino* 2.84%.
It is therefore useful for early Italian literary language, but too concentrated
to train alone. The later merge must set and record source caps or sampling
weights together with the resulting token counts.

## Deferred Sources

The five Project Gutenberg *Cronica di Matteo Villani* volumes (ebooks
69898-69902) were probed but are not activated. Each 1825 Ignazio Moutier
edition includes a substantial editor preface before the primary chronicle and
a terminal table of contents. Generic Gutenberg-wrapper stripping cannot
reliably identify the primary-text boundaries. A future source-specific,
tested primary-span extractor is required before these volumes are included.

Their probe reports are committed in `data/metadata/` for volumes 1-5. The
deferral is about extraction quality, not Project Gutenberg's stated
public-domain status.

## Verification

The builder deletes its temporary raw and interim directory after success. The
final validation checked the processed texts for the audited terminal and
edition markers removed by the source-specific cleaning rules. The Python test
suite passed with 504 tests after those rules were added.
