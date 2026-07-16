# Broader Italian Corpus Expansion Audit

## Decision

The project will maintain two prose-only broader-pretraining corpus versions:

| Version | Period | Initial target | Purpose |
| --- | --- | ---: | --- |
| `vintage_1200_1600` | 1200-1600 | 30M BPE tokens | Preserve the historical-Italian research goal. |
| `expanded_italian_1200_1800` | 1200-1800 | 75M BPE tokens | Provide a larger general-language parent for the quality-focused comparison. |

Neither version may contain poems, drama, or the held-out sonnet corpus. The
sonnet corpus remains a separate downstream fine-tuning and evaluation dataset.

## Assembly Policy

The current local broader corpus has 18.8M training BPE tokens. It is not only
small relative to the planned targets; one Ramusio compilation contributes about
28.7 percent of its cleaned characters. The next builder revision must therefore
apply deterministic caps before tokenization:

- no one source work may contribute more than 15 percent of assembled tokens;
- no one author may contribute more than 20 percent of assembled tokens;
- the build report must show raw and retained token counts by source, author,
  archive, period bucket, and corpus version;
- capping must use a deterministic seed and be recorded in the corpus manifest.

These caps trade away some raw token volume to avoid a parent model whose
language distribution is disproportionately one compiler, editor, or genre.

## New Source Audit: Italian Wikisource

Italian Wikisource is an approved attribution-preserving source route. The
following records are candidate additions for `expanded_italian_1200_1800`.
They are entered in the source manifest with `audit_then_include`, not active
training status. Before activation, the project must implement an extractor that
uses a fixed revision or dated dump, stores attribution metadata, and removes
navigation, scan labels, and modern editorial text reproducibly.

| Source ID | Work | Author | Date | Why it adds value | Audit result |
| --- | --- | --- | ---: | --- | --- |
| `ws_galileo_saggiatore` | *Il Saggiatore* | Galileo Galilei | 1623 | Scientific argumentative prose and a new author. | The work page identifies Galileo, its 1623 date, subject, and CC BY-SA 3.0/GFDL rights metadata. [Source](https://it.wikisource.org/wiki/Il_Saggiatore) |
| `ws_galileo_dialogo` | *Dialogo sopra i due massimi sistemi del mondo* | Galileo Galilei | 1632 | Long dialogic scientific prose. | The collection page identifies the work, Galileo, a 1897 Favaro edition, CC BY-SA 3.0/GFDL metadata, and a navigable contents list. Extract only the 1632 *Dialogo* sections. [Source](https://it.wikisource.org/wiki/Dialogo_sopra_i_due_massimi_sistemi_del_mondo_tolemaico_e_copernicano_%28raccolta%29) |
| `ws_vico_scienza_nuova` | *La scienza nuova* | Giambattista Vico | 1744 | Adds eighteenth-century philosophical and historical prose. | **Deferred.** The 1911 Wikisource edition mixes variants and historical notes into the rendered prose. A 27-page revision-pinned audit found 124 bracketed markers and 23 `Si veda` references, so it is not suitable for generic cleaning. [Source](https://it.wikisource.org/wiki/La_scienza_nuova_-_Volume_I) |
| `ll_vico_principj_scienza_nuova` | *Principj di scienza nuova* | Giambattista Vico | 1744 | Candidate replacement for the deferred annotated edition. | Liber Liber offers TXT+ZIP and identifies a CC BY-NC-SA 4.0 digital edition with Paolo Rossi / Rizzoli source-edition and Claudio Paganelli digitization/revision credits. It remains audit-only until cleaned samples are inspected. [Source](https://liberliber.it/autori/autori-v/giambattista-vico/principj-di-scienza-nuova-dintorno-alla-comune-natura-delle-nazioni-in-questa-terza-impressione-dal-medesimo-autore-in-un-gran-numero-di-luoghi-corretta-schiarita-e-notabilmente-accresciuta/) |
| `ws_beccaria_delitti_pene` | *Dei delitti e delle pene* | Cesare Beccaria | 1764 | Adds compact Enlightenment legal prose. | Wikisource lists Italian editions and its work page identifies the original 1764 publication and CC BY-SA licensing. Build from the primary chapters only, excluding Voltaire material and later commentary. [Source](https://it.wikisource.org/wiki/Opera%3ADei_delitti_e_delle_pene) |
| `ws_giannone_istoria_civile_vol1` | *Istoria civile del Regno di Napoli*, vol. 1 | Pietro Giannone | 1723; 1770 source edition | Large historical prose from a new author and region. | The Wikisource index offers TXT/EPUB/PDF/RTF exports for the public-domain 1770 scan. Audit every volume for completeness before adding more than volume 1. [Source](https://it.wikisource.org/wiki/Indice%3AGiannone_-_Istoria_civile_del_regno_di_Napoli%2C_1770%2C_Vol.1.djvu) |

The Galileo, Vico, Beccaria, and Giannone works are post-1600 and therefore do
not enter `vintage_1200_1600`. They make the expanded comparison corpus larger
and more varied without changing the historical identity of the vintage corpus.

## Source-Specific Extraction Risks

- Wikisource pages and exports can include scan-derived headers, page numbers,
  editorial introductions, navigation, and incomplete transcription pages.
- The Wikisource *La scienza nuova* edition contains variants and historical
  notes embedded in the rendered text; it is deferred rather than cleaned with
  broad text-removal rules.
- The Liber Liber Vico candidate has a separate CC BY-NC-SA 4.0 license layer.
  Its downloadable text, wrapper removal, and samples require inspection before
  it can become an active source.
- The Galileo collection contains related works and fragments; title-level
  scoping is required to avoid collecting unrelated texts.
- Giannone is multi-volume. Treat each audited volume as a separate source row
  so source caps and extraction failures are visible.
- All Wikisource-derived data must retain page URL, revision or dump date,
  license label, source edition/scan note, extraction date, and cleaning method.

## Next Implementation Checkpoint

Italian-Wikisource adapter status: implemented for an audit-only probe of
*Il Saggiatore*. The root page is an index, not one self-contained text: it
lists `Dedica`, `Prefazione`, and 53 numbered primary-text subpages. The adapter
therefore records the root revision and every included subpage revision, checks
the expected first and last subpages, strips known website wrappers, and writes
only a local inspection report with provenance and short samples. It batches
revision lookups and uses a six-second request interval with visible backoff on
rate limits.

The completed local probe verified 55 unique pages, revision provenance, clean
start/end samples, and absence of the checked wrapper markers. Therefore
`ws_galileo_saggiatore` is activated as `include_probe` for
`expanded_italian_1200_1800_v1` only. Its committed snapshot is
[`data/metadata/wikisource_snapshots/ws_galileo_saggiatore.json`](../data/metadata/wikisource_snapshots/ws_galileo_saggiatore.json).
The _Dialogo_ probe then verified its six primary sections and activated
`ws_galileo_dialogo` with an `explicit_subpages` snapshot that excludes indexes,
fragments, collection material, and Latin text. The remaining three candidates
remain `audit_then_include` until they pass the same source-specific inspection
gate.

Vico audit scope on 2026-07-15: the bibliographic work title is *La scienza
nuova*, while its relevant Wikisource root index is `La scienza nuova - Volume
I`. The probe records that distinction explicitly, excludes the two editorial
branches `Dedica dell'editore` and `Introduzione dell'editore`, and excludes the
image-only `Illustrazione` frontispiece because it has no transcribed text. It
starts at `Titolo` and intentionally has no hard-coded final page, so the
complete current primary-text hierarchy can be inspected rather than silently
truncated. It remains `audit_then_include` until that local revision-pinned
probe confirms the selected page list and cleaned samples.

The successful 27-page Vico probe retained 418,315 cleaned characters but
revealed editorial material in its end sample, including bracketed variant
markers and `Si veda` reference lines. The source therefore remains
`audit_then_include`. The project provides
`scripts/audit_italian_wikisource_editorial_markers.py`, which writes a local,
revision-pinned audit report with aggregate counts and bounded page-level
contexts for candidate bracketed text and `Si veda` references. This report is
an inspection artifact only; it does not alter source text or activate Vico.

The annotated Wikisource Vico row is now `defer`. Its replacement candidate is
Liber Liber's *Principj di scienza nuova*, which is audited through
`scripts/probe_liber_liber_source.py`. That single-source probe uses the
existing TXT-ZIP/ODT discovery and boilerplate removal path, then writes only
archive provenance, size measurements, and start/end cleaned samples to a local
inspection report. It remains `audit_then_include` until that report is
reviewed.

The Liber Liber audit completed on 2026-07-16. Its TXT ZIP produced 943,208
cleaned characters and 150,586 whitespace-delimited words. The end sample is
Vico's conclusion rather than a scholarly note. It contains no `Si veda`
references. Of 64 bracketed spans, the inspected values are Vico's historical
timeline labels and must be retained; one front-matter placeholder,
`[inserisci figura1]`, remains for a separate narrow cleaning decision. The
candidate remains `audit_then_include` until that rule and activation are
explicitly approved.
