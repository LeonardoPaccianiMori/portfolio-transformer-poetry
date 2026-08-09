# Biblioteca Italiana Historical Corpus Composition Audit

## Decision

The metadata composition gate passes. BibIt is a high-value expansion source,
but this report does **not** activate all records as training data. Every work
must still pass its role-specific TEI, edition, language-variety, and leakage audit.

- Catalog records reviewed: 1,387.
- Origins-through-Settecento records: 1,042.
- Ottocento bridge candidates: 345.
- Duplicate/edition families requiring one canonical selection: 3.
- Status: `composition_gate_passed_audit_required_before_activation`.

## Reuse Terms

BibIt states that its digital resources are freely accessible for personal or
scientific use, that public reuse must cite Biblioteca Italiana as the source,
and that commercial reuse is prohibited. This source therefore creates a
non-commercial data/model lineage. Work-level TEI headers must remain the
canonical edition, editor, publisher, revision, and digitization record.

- [Biblioteca Italiana project page](https://bibliodlcm.web.uniroma1.it/it/biblioteca-italiana)
- [Biblioteca Italiana FAQ](http://backend.bibliotecaitaliana.it/faq/)
- [BibIt catalog](http://www.bibliotecaitaliana.it/)

## Catalog Composition

| Period | Records |
| --- | ---: |
| Origini | 9 |
| Duecento | 27 |
| Trecento | 92 |
| Quattrocento | 233 |
| Cinquecento | 407 |
| Seicento | 135 |
| Settecento | 139 |
| Ottocento | 345 |

| Genre | Records |
| --- | ---: |
| Poesia | 491 |
| Trattati | 318 |
| Documenti | 151 |
| Letteratura teatrale | 146 |
| Narrativa | 93 |
| Lettere ed epistolari | 59 |
| Testi storici e storiografici | 30 |
| Memorialistica | 27 |
| Oratoria | 23 |
| Traduzioni e volgarizzamenti | 23 |
| Commenti, traduzioni e volgarizzamenti | 21 |
| Prosa scientifica, morale e d'invenzione | 3 |
| memorialistica | 1 |
| oratoria | 1 |

## Projected Roles

The projected sizes come from a deterministic period/genre sample and are
planning estimates, not final post-cleaning token counts.

| Role | Records | Estimated characters | Share of non-excluded estimate |
| --- | ---: | ---: | ---: |
| `excluded` | 14 | 4,556,235 | 0.0% |
| `historical_general` | 638 | 126,012,816 | 57.3% |
| `historical_non_sonnet_poetry` | 176 | 12,313,259 | 5.6% |
| `nineteenth_century_bridge` | 323 | 59,013,969 | 26.8% |
| `sonnet_only` | 236 | 22,562,688 | 10.3% |

## Scale Estimate

- Deterministic estimation sample: 186 records across 75 period/genre strata.
- Sample rendered literary characters: 57,309,184.
- Sample records with no rendered primary text: 2; these records are excluded.
- Projected historical non-excluded characters before TEI cleaning and deduplication: 160,888,763.
- Approximate historical token range at 3-4 characters/token: 40,222,190-53,629,588.

The Ottocento bridge is not part of that historical token estimate. The current
recommendation is to cap it at no more than 10% of a future adaptation mixture, with the exact cap frozen only after full
cleaning, deduplication, and language-variety review.

### Measured Long-Work Anchors

| Record | Work | Rendered characters |
| --- | --- | ---: |
| `bibit000019` | Alighieri, Dante / Commedia | 510,780 |
| `bibit000049` | Boiardo, Matteo Maria / Orlando innamorato | 1,301,590 |
| `bibit001135` | Ariosto, Ludovico / Orlando Furioso 1532 | 1,433,624 |
| `bibit001501` | Tasso, Torquato / Gerusalemme liberata | 597,020 |
| `bibit000099` | Tasso, Torquato / Rime | 1,166,168 |
| `bibit000760` | Petrarca, Francesco / Canzoniere | 290,949 |
| `bibit000666` | Manzoni, Alessandro / Promessi Sposi | 1,299,361 |
| `bibit001238` | Nievo, Ippolito / Confessioni di un Italiano | 2,010,770 |
| `bibit001705` | Leopardi, Giacomo / Zibaldone di pensieri | 5,795,448 |

These measured anchors prevent unusually long canonical works from being
replaced by the median size of their period/genre stratum.

## Canonical Editions

Near-duplicate editions do not enter together. The decision CSV marks one
candidate per normalized author/title family and excludes alternates. The
Ariosto family has an explicit editorial override: use *Orlando Furioso 1532*,
the final authorial edition, rather than combining the 1516, 1521, 1532, and
modern-edition records. The Manzoni family similarly selects the later standard
*Promessi Sposi* text and excludes the separate 1827 redaction from this mixture.

| Family | Selected record | Alternates | Reason |
| --- | --- | ---: | --- |
| Ariosto, Ludovico / orlando furioso | `bibit001135` | 3 | project override selects Ariosto's final 1532 authorial edition |
| Da Morrona, Alessandro / pisa illustrata nelle arti del disegno | `bibit000155` | 1 | metadata-completeness tie-breaker; verify before activation |
| Manzoni, Alessandro / promessi sposi | `bibit000666` | 1 | project override selects Manzoni's later standard text over the 1827 redaction |

## Leakage And Curriculum

Explicit sonnets and mixed collections that may contain sonnets are excluded from historical pretraining; held-out validation/test sonnets must be absent from every earlier stage.

1. historical prose/general adaptation with limited preservation replay.
2. historical non-sonnet poetry adaptation with prose and preservation replay.
3. low-learning-rate sonnet specialization with author/epoch-balanced sampling.

Long poems such as the *Commedia*, *Orlando innamorato*, *Orlando Furioso*,
and *Gerusalemme liberata* belong in the non-sonnet poetry stage. Collections
such as *Rime* and *Canzoniere* remain sonnet-only candidates until TEI-form
segmentation proves which individual units are eligible. Sonnets retained for
specialization will use author/epoch-balanced sampling, and the existing held-out
validation/test assignments remain fixed.

## Next Activation Gate

1. Download only canonical TEI records selected for a named stage.
2. Parse with external DTD/network resolution disabled and retain TEI-header provenance.
3. Route every explicit `lg type="sonetto..."` unit away from historical pretraining.
4. Audit dialect, editorial apparatus, language, empty text, and exact/near duplicates.
5. Freeze final post-cleaning token shares, the Ottocento cap, preservation replay,
   and validation splits before restarting any GPU training.
