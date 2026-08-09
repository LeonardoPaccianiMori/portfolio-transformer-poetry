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

The composition audit does not itself activate the texts. Canonical TEI
extraction, language and editorial review, deduplication, and final mixture
approval are still required.

## Planned Transformations

- Parse only the TEI body with external DTD and network resolution disabled.
- Resolve only a controlled set of standard named character entities.
- Exclude the TEI header, generated wrappers, page markers, editorial notes,
  arguments, alternate apparatus readings, and deleted text.
- Preserve source spelling, punctuation, paragraph boundaries, verse lines,
  and stanza boundaries.
- Route every explicit `lg type="sonetto..."` unit away from historical
  pretraining and into the separately deduplicated sonnet audit.
- Keep one canonical edition per work and retain all alternate-edition
  decisions in the public metadata.
- Exclude dialect-heavy works from the unconditioned standard-Italian core.

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
