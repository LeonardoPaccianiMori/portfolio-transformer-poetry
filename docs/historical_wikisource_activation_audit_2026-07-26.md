# Historical Wikisource Activation Audit

## Decision

This audit activates three complete Italian Wikisource prose works for the
versioned `pretraining_historical_wikisource_v1` component. They are a
historical addition to the larger pretraining corpus, not a standalone
pretraining dataset.

| Source | Audited pages | Audited characters | Activation decision |
| --- | ---: | ---: | --- |
| Paolo Sarpi, *Istoria del Concilio tridentino* | 87 | 2,990,432 | Activate with 3 index pages excluded, leaving 84 revision-pinned primary pages. |
| Pietro Verri, *Storia di Milano* | 36 | 1,617,868 | Activate; remove only the exact editorial gloss `[parla di Arialdo]`. |
| Pietro Verri, *Osservazioni sulla tortura* | 17 | 155,474 | Activate without source-specific text removal. |

The three audited works total 4,763,774 characters before the approved narrow
exclusions. Their source snapshots, processed text, and final build report are
committed when the component build completes.

## Editorial-Marker Policy

Sarpi's audit found 33 bracketed items. Most are short in-text readings such as
`[si]` and `[in]`; their editorial status cannot be determined safely from the
rendered text alone. They are retained rather than guessed away. All three
Sarpi `Indice del ... volume` leaves are excluded because they are structural
tables of contents, not primary prose.

Verri's *Storia di Milano* contained one clear editorial gloss, `[parla di
Arialdo]`, which is removed exactly once. No broad bracket-removal rule is
used.

## Deferred Giannone Volumes

All five Pietro Giannone volumes remain deferred. The revision-pinned audit
found at least 36 missing or untranscribed selected pages in volume 1, at least
50 in each of volumes 2-4, and at least 46 in volume 5. The reported numbers
are lower bounds because the audit stops each volume after its first failing
revision batch. The problem is source completeness, not licensing. A complete
permitted edition is required before reconsidering these works.

## Reuse And Provenance

The underlying works are public domain. The Italian Wikisource transcription
layer requires retention of its stated attribution and CC BY-SA/GFDL metadata.
The exact source links, source editions, root revisions, page revisions, and
cleaning policy are recorded in the manifest, committed snapshots, and
[`pretraining_historical_wikisource_v1_attribution.md`](../data/metadata/pretraining_historical_wikisource_v1_attribution.md).
