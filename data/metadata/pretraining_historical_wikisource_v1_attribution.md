# Historical Wikisource Component Attribution

`pretraining_historical_wikisource_v1` contains revision-pinned Italian
Wikisource transcriptions. The underlying works are public domain, but the
Wikisource transcription layer has its own attribution and reuse conditions.
The corpus and any derived model lineage must retain the source links, revision
records, and stated CC BY-SA/GFDL obligations.

## Istoria del Concilio tridentino

- Author: Paolo Sarpi
- Source: [Italian Wikisource](https://it.wikisource.org/wiki/Istoria_del_Concilio_tridentino)
- Source edition: Giovanni Gambarin edition, 1935, arranged across three volumes.
- Rights record: underlying 1619 work is public domain; the Wikisource
  transcription reports CC BY-SA 3.0 and GFDL terms.
- Provenance: the committed snapshot pins root revision `3809430` and 67
  primary-text page revisions. Three volume index pages and 17 alphabetical
  name-index pages are excluded.
- Changes: navigation wrappers are removed; uncertain in-text square brackets
  are retained.

## Storia di Milano

- Author: Pietro Verri
- Source: [Italian Wikisource](https://it.wikisource.org/wiki/Storia_di_Milano)
- Source edition: Societa Tipografica de' Classici Italiani, 1834.
- Rights record: underlying work is public domain; the Wikisource transcription
  reports Creative Commons Attribution-ShareAlike terms.
- Provenance: the committed snapshot pins root revision `3828025` and 35
  primary-text page revisions; the editor's `Avvertimento` page is excluded.
- Changes: navigation wrappers are removed. The exact editorial gloss `[parla
  di Arialdo]` is removed; all other spelling and punctuation are retained.

## Osservazioni sulla tortura

- Author: Pietro Verri
- Source: [Italian Wikisource](https://it.wikisource.org/wiki/Osservazioni_sulla_tortura)
- Source edition: Milan, 1843.
- Rights record: underlying work is public domain; the Wikisource transcription
  reports CC BY-SA 3.0 and GFDL terms.
- Provenance: the committed snapshot pins root revision `3821110` and 17 page
  revisions.
- Changes: navigation wrappers are removed; the audited primary prose is
  otherwise retained.

The exact immutable page revisions are committed in
`data/metadata/wikisource_snapshots/`.
