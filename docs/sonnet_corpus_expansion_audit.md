# Sonnet Corpus Expansion Audit

## Purpose and Decision Boundary

The active sonnet corpus contains 921 included poems from eight source
collections. Its author distribution is concentrated: Francesco Petrarca
contributes 313 poems and Guittone d'Arezzo contributes 210. This is useful
historical material, but it is too small and too concentrated to be the final
fine-tuning corpus for a model intended to generate convincing Italian
sonnets.

This audit defines a reproducible route to a larger `sonnets_expanded_v2`
corpus. It does **not** alter `data/processed/poems/`, its manifest, its
splits, or any completed experiment. A source becomes active only after it has
passed the extraction, 14-line form, duplicate, and provenance checks listed
below.

The expansion has two deliberately separate roles:

| Role | Contents | Training use |
| --- | --- | --- |
| `core_standard_italian` | Standard/literary Italian sonnets with verified source reuse terms. | Eligible for the primary fine-tuning and evaluation corpus. |
| `auxiliary_dialectal` | Sonnets whose language variety is materially dialectal, such as Romanesco. | Kept separate; never mixed into unconditioned core fine-tuning. It may support a later explicitly conditioned experiment. |

The project will target at least 500 additional verified
`core_standard_italian` sonnets before training the next data-scaling variant.
This is a concrete data-quality experiment, not a claim that corpus expansion
alone will make a 33.7M-parameter model coherent.

## Current Corpus Inventory

| Author | Included poems |
| --- | ---: |
| Francesco Petrarca | 313 |
| Guittone d'Arezzo | 210 |
| Cecco Angiolieri | 123 |
| Dante Alighieri | 84 |
| Cino da Pistoia | 81 |
| Guido Cavalcanti | 45 |
| Folgore da San Gimignano | 36 |
| Giacomo da Lentini | 24 |
| Other displayed correspondence authors | 5 |
| **Total** | **921** |

The existing corpus uses Italian Wikisource pages with source URL, edition
notes, and license notes stored for every poem in
`data/metadata/poems_manifest.csv`. The expansion must preserve this
per-poem provenance standard.

## Candidate Sources

The machine-readable counterpart is
`data/metadata/sonnet_expansion_sources_manifest.csv`.

| Source ID | Candidate | Role | Status | Evidence and decision |
| --- | --- | --- | --- | --- |
| `ws_alfieri_rime_1912` | Vittorio Alfieri, *Rime varie* (1912 edition) | Core | `activated` | The 195-page revision-pinned audit retained 45 exact 14-line sonnets and excluded 150 other-form or multi-poem pages. No retained poem is an exact duplicate of the active v1 corpus. The parallel 1903 edition is excluded to prevent duplicate poems. [Source](https://it.wikisource.org/wiki/Rime_varie_(Alfieri,_1912)) |
| `ws_foscolo_sonetti` | Ugo Foscolo, *Sonetti* | Core | `audit_then_include` | A compact, standard-Italian collection appropriate for a clean source-specific probe. Its exact eligible count is intentionally unknown until the page-level audit. [Source](https://it.wikisource.org/wiki/Sonetti_(Foscolo)) |
| `ws_varchi_infermita` | Benedetto Varchi, *Sonetti per la infermità, e guarigione di Cosimo I dei Medici* | Core | `audit_then_include` | A Renaissance candidate with an explicit sonnet title. Its former manifest title was corrected against the canonical Wikisource work page before audit. [Source](https://it.wikisource.org/wiki/Sonetti_per_la_infermit%C3%A0,_e_guarigione_di_Cosimo_I_dei_Medici) |
| `ws_belli_sonetti_romaneschi` | Giuseppe Gioachino Belli, *Sonetti romaneschi* | Auxiliary dialectal | `audit_only_auxiliary` | The author page identifies the collection but also identifies Belli as a dialect writer. Its 2,042-text author catalogue makes it potentially valuable, but Romanesco would strongly change the primary corpus's language distribution. It is excluded from core training and retained for a separate controlled experiment only. [Author page](https://it.wikisource.org/wiki/Autore:Giuseppe_Gioachino_Belli) [Collection](https://it.wikisource.org/wiki/Sonetti_romaneschi) |
| `ws_aretino_sonetti_lussuriosi_1792` | Pietro Aretino, *Sonetti lussuriosi* (1792 edition) | Undecided | `audit_only_explicit_content` | The user authorized a metadata-and-form audit only. It cannot enter any corpus without a separate activation decision; its audit report retains no text samples. [Source](https://it.wikisource.org/wiki/Sonetti_lussuriosi_(edizione_1792)) |
| `biblioteca_italiana_tei_poetry` | Biblioteca Italiana TEI poetry collections | Core candidate | `blocked_pending_terms_and_access` | Biblioteca Italiana reports downloadable XML-TEI texts and broad literary coverage, which could allow high-quality poem segmentation. Direct access and the exact reuse terms for a chosen source must be confirmed before ingestion. [Library description](https://bibliodlcm.web.uniroma1.it/it/biblioteca-italiana) |
| `ws_category_sonetti` | Italian Wikisource `Categoria:Sonetti` | Discovery only | `discovery_only` | The category has thousands of pages but mixes authors, editions, periods, dialects, and non-standard variants. It is an index for named-source discovery, never a bulk-download source. [Category](https://it.wikisource.org/wiki/Categoria:Sonetti) |
| `project_gutenberg_italian_sonnets` | Project Gutenberg Italian poetry records | Excluded for this expansion | `not_prioritized` | The initial audit did not identify a material new collection of original Italian sonnets. The Petrarch result is a translation collection, so it does not add target-language data. |

## Required Activation Checks

Every `audit_then_include` source must complete the following ordered checks:

1. **Root-page and revision audit:** record the root URL, exact primary-text
   subpages, revision IDs or a dated source snapshot, edition metadata, and
   reuse terms.
2. **Extraction audit:** inspect bounded start/end samples and representative
   middle pages. Remove only identified site wrappers, line labels, and
   editorial apparatus; retain historical spelling, punctuation, and line
   breaks.
3. **Form audit:** include only records with exactly 14 cleaned poetic lines.
   A source heading that says `Sonetti` is useful evidence but does not replace
   the page-level line-count check.
4. **Duplicate audit:** compare normalized cleaned text with the active corpus
   and other candidate editions. A work may enter from one selected edition
   only.
5. **Split audit:** assign deterministic train/validation/test splits at poem
   level only after deduplication. No text-identical or near-identical poem may
   cross splits.
6. **Build and provenance publication:** add only processed text, the updated
   poem manifest, source-attribution records, and the build report to Git.
   Delete raw and interim workspaces after a successful build.

The source adapter should be generic enough to probe root-plus-subpage
Wikisource collections. Source-specific cleaning belongs in narrowly tested
rules, not a broad normalization pass that could remove historical text.

## Next Scheduled Checkpoint

The first activated expansion source is `ws_alfieri_rime_1912`. Its local
inspection report ran from 2026-07-18T11:16:11+00:00 to
2026-07-18T11:36:13+00:00 and inspected 195 pages. It retained 45 exact
14-line candidates, excluded 150 pages, and found no exact v1 duplicate. The
versioned builder copied the 921 active v1 poems and re-fetched only the 45
committed source revisions. The resulting `sonnets_expanded_v2` dataset has
966 processed poems with 772 train, 98 validation, and 96 test poems.

The next scheduled source audit is `ws_foscolo_sonetti`, using the same
revision-pinned, audit-only process before any activation decision.
