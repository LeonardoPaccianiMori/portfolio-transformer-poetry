# Biblioteca Italiana Historical Collection Attribution

## Source And Terms

- Source: [Biblioteca Italiana / BibIt](http://www.bibliotecaitaliana.it/)
- Project description: [Biblioteca Italiana, Sapienza University of Rome](https://bibliodlcm.web.uniroma1.it/it/biblioteca-italiana)
- Reuse evidence: [Biblioteca Italiana FAQ](http://backend.bibliotecaitaliana.it/faq/)
- Catalog/API host: `http://backend.bibliotecaitaliana.it/`
- Audit/download date: 2026-08-09

Biblioteca Italiana states that its digital texts are freely accessible for
personal or scientific use, that public reuse must cite Biblioteca Italiana as
the source, and that commercial reuse is prohibited. This source therefore
creates a non-commercial data and model lineage. Attribution is required even
where the underlying historical work is public domain.

Every downloaded TEI header remains the canonical work-level record for:

- digital title, author, publisher, place, date, and identifier;
- printed source title, author, editor, publisher, place, date, and catalog ID;
- digitization, encoding, correction, and revision contributors;
- editorial and transcription practices;
- source-specific availability wording.

## Audited Scope

The committed metadata snapshot contains 1,387 Italian BibIt records from the
origins through the nineteenth century: 1,042 origins-through-Settecento works
and 345 Ottocento candidates. The public composition decision file assigns each
record to one of five roles:

- historical general text;
- historical non-sonnet poetry;
- sonnet-only/form-aware extraction;
- capped nineteenth-century bridge;
- excluded duplicate, dialect-heavy, or empty record.

The role-specific TEI audit subsequently processed all 1,373 selected canonical
records. The difference from the 1,387-record catalog snapshot is the 14
alternate or composition-excluded records, which were retained in the decision
evidence but deliberately not downloaded as canonical training candidates.

The full audit routes material into all three intended corpus stages: historical
general text, historical non-sonnet poetry, and sonnet specialization, plus a
separately capped nineteenth-century bridge. It reports 1,204 automatic record
candidates and 168 review-required records; `bibit00332` is the only parse
error because both known archive identifiers return an empty TEI document.
Review status is not activation.

## Implemented Transformations

- Parse only the TEI body with external DTD and network resolution disabled.
- Resolve standard HTML/XML entities plus a static allowlist of the 75 legacy
  Greek entities encountered in the audited snapshot. The values are pinned to
  the public BibIt numeric-entity DTD with SHA-256
  `2073624517861fd673fba09b8153e0d481e058cb091dd17e59262bb04b9c1aa5`;
  the parser never loads the DTD, and unknown named entities fail closed.
- Exclude the TEI header, generated wrappers, page markers, editorial notes,
  arguments, alternate apparatus readings, and deleted text.
- Preserve source spelling, punctuation, paragraph boundaries, verse lines,
  and stanza boundaries.
- Route every explicit `lg type="sonetto..."` unit away from historical
  pretraining and into the separately deduplicated sonnet audit.
- Keep one canonical edition per work and retain all alternate-edition
  decisions in the public metadata.
- Exclude dialect-heavy works from the unconditioned standard-Italian core.

## Full TEI Audit Results

- General historical route: 562 automatic candidates and 74,532,753
  candidate characters.
- Historical non-sonnet poetry route: 340 automatic candidates and 37,953,293
  candidate characters.
- Nineteenth-century bridge: 302 automatic candidates and 41,036,542 candidate
  characters; final mixture weighting is still pending.
- Sonnet audit: 18,742 explicit TEI candidates and 5,596 structural 14-line
  candidates. Of these, 13,037 explicit non-duplicates are automatically
  eligible for the next composition stage.
- All 1,868 active V6 sonnets were used for duplicate checks. All 411 conflicts
  with the 387 V6 validation/test identities are explicitly excluded, and no
  explicit or structurally inferred sonnet remains in an earlier-stage route.

The raw TEI cache remains machine-local under `data/local/bibit/tei`. Public
records retain each downloaded TEI SHA-256, extraction status, routed character
counts, review flags, provenance, and poem-level decisions.

## Measured Representative Works

| Work | BibIt ID | Rendered characters | Planned role |
| --- | --- | ---: | --- |
| Dante, *Commedia* | `bibit000019` | 510,780 | historical non-sonnet poetry |
| Boiardo, *Orlando innamorato* | `bibit000049` | 1,301,590 | historical non-sonnet poetry |
| Ariosto, *Orlando Furioso 1532* | `bibit001135` | 1,433,624 | historical non-sonnet poetry |
| Tasso, *Gerusalemme liberata* | `bibit001501` | 597,020 | historical non-sonnet poetry |
| Tasso, *Rime* | `bibit000099` | 1,166,168 | explicit-sonnet/mixed-form audit |
| Petrarch, *Canzoniere* | `bibit000760` | 290,949 | explicit-sonnet/mixed-form audit |
| Manzoni, *Promessi Sposi* | `bibit000666` | 1,299,361 | capped Ottocento bridge |
| Nievo, *Confessioni di un Italiano* | `bibit001238` | 2,010,770 | capped Ottocento bridge |
| Leopardi, *Zibaldone di pensieri* | `bibit001705` | 5,795,448 | capped Ottocento bridge with author concentration control |

The Ariosto decision selects the final 1532 authorial edition. The Manzoni
decision selects the later standard text and excludes the separate 1827
redaction from the same training mixture.

## Public Evidence

- [`bibit_catalog_origins_through_ottocento_v1.json`](bibit_catalog_origins_through_ottocento_v1.json)
- [`bibit_historical_composition_decisions.csv`](bibit_historical_composition_decisions.csv)
- [`../../reports/bibit_historical_composition_audit.md`](../../reports/bibit_historical_composition_audit.md)
- [`bibit_tei_audit_records.csv`](bibit_tei_audit_records.csv)
- [`bibit_sonnet_candidates_audit.csv`](bibit_sonnet_candidates_audit.csv)
- [`../../reports/bibit_tei_role_audit.md`](../../reports/bibit_tei_role_audit.md)
- [`../../reports/bibit_tei_role_audit.json`](../../reports/bibit_tei_role_audit.json)
