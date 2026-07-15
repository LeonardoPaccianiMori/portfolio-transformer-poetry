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
| `ws_vico_scienza_nuova` | *La scienza nuova* | Giambattista Vico | 1744 | Adds eighteenth-century philosophical and historical prose. | Wikisource hosts a 1911 edition based on the 1744 text, but it includes editorial introductions and variants. Extraction must isolate Vico's primary text. [Source](https://it.wikisource.org/wiki/La_scienza_nuova_-_Volume_I) |
| `ws_beccaria_delitti_pene` | *Dei delitti e delle pene* | Cesare Beccaria | 1764 | Adds compact Enlightenment legal prose. | Wikisource lists Italian editions and its work page identifies the original 1764 publication and CC BY-SA licensing. Build from the primary chapters only, excluding Voltaire material and later commentary. [Source](https://it.wikisource.org/wiki/Opera%3ADei_delitti_e_delle_pene) |
| `ws_giannone_istoria_civile_vol1` | *Istoria civile del Regno di Napoli*, vol. 1 | Pietro Giannone | 1723; 1770 source edition | Large historical prose from a new author and region. | The Wikisource index offers TXT/EPUB/PDF/RTF exports for the public-domain 1770 scan. Audit every volume for completeness before adding more than volume 1. [Source](https://it.wikisource.org/wiki/Indice%3AGiannone_-_Istoria_civile_del_regno_di_Napoli%2C_1770%2C_Vol.1.djvu) |

The Galileo, Vico, Beccaria, and Giannone works are post-1600 and therefore do
not enter `vintage_1200_1600`. They make the expanded comparison corpus larger
and more varied without changing the historical identity of the vintage corpus.

## Source-Specific Extraction Risks

- Wikisource pages and exports can include scan-derived headers, page numbers,
  editorial introductions, navigation, and incomplete transcription pages.
- *La scienza nuova* contains a modern editor's introduction and variant notes;
  it cannot be included through a generic page concatenation.
- The Galileo collection contains related works and fragments; title-level
  scoping is required to avoid collecting unrelated texts.
- Giannone is multi-volume. Treat each audited volume as a separate source row
  so source caps and extraction failures are visible.
- All Wikisource-derived data must retain page URL, revision or dump date,
  license label, source edition/scan note, extraction date, and cleaning method.

## Next Implementation Checkpoint

Implement a small Italian-Wikisource source adapter and tests. It must fetch a
single manifest-approved work through a documented export or dump route, record
the page revision/date, strip known wrapper text, and reject pages that do not
meet the expected title or content-boundary checks. Start with *Il Saggiatore*,
which is one self-contained work with clear provenance. Do not activate the
other four candidates until that adapter and an inspected local probe succeed.
