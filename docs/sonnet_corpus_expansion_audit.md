# Sonnet Corpus Expansion Audit

## Purpose and Decision Boundary

The active `sonnets_expanded_v5` corpus contains 1,875 included poems. It
preserves all 1,011 v4 poems and adds 864 revision-pinned sonnets from five
standard-literary-Italian sources. Vittoria Colonna is now the largest author
with 336 poems, followed by Francesco Petrarca with 313 and Gaspara Stampa
with 282.

This audit defines and records the reproducible route used to create v5. It did
not alter v4, its splits, or any completed experiment. Each activated source
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
| Vittorio Alfieri | 45 |
| Ugo Foscolo | 12 |
| Benedetto Varchi | 33 |
| Isabella Andreini | 196 |
| Vittoria Colonna | 336 |
| Gaspara Stampa | 282 |
| Ludovico Ariosto | 29 |
| Jacopo Sannazaro | 21 |
| **Total** | **1,875** |

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
| `ws_foscolo_sonetti` | Ugo Foscolo, *Sonetti* | Core | `activated` | The revision-pinned 1835-edition audit retained all 12 exact 14-line pages, which are active in `sonnets_expanded_v3`. [Source](https://it.wikisource.org/wiki/Sonetti_(Foscolo)) |
| `ws_varchi_infermita` | Benedetto Varchi, *Sonetti per la infermità, e guarigione di Cosimo I dei Medici* | Core | `activated` | The revision-pinned audit retained 33 exact 14-line, non-duplicate poems. The 1821 Magheri edition is active in `sonnets_expanded_v4`, with root and page revisions recorded in the committed snapshot. [Source](https://it.wikisource.org/wiki/Sonetti_per_la_infermit%C3%A0,_e_guarigione_di_Cosimo_I_dei_Medici) |
| `ws_andreini_rime_1601` | Isabella Andreini, *Rime* (1601) | Core | `activated` | V5 contains 196 exact 14-line sonnets from 189 pinned pages. Seven paired pages were split only at printed `SONETTO` headings. [Source](https://it.wikisource.org/wiki/Rime_(Andreini)) |
| `ws_colonna_rime_1760` | Vittoria Colonna, *Rime* (1760 edition) | Core | `activated` | V5 contains 336 exact 14-line sonnets from 336 pinned pages: 125 secular sonnets plus 211 eligible `Sonetti spirituali`. The final 17-line spiritual page remains excluded. [Source](https://it.wikisource.org/wiki/Rime_(Vittoria_Colonna)) |
| `ws_stampa_rime_1913` | Gaspara Stampa, *Rime* (1913 edition) | Core | `activated` | V5 contains 282 exact 14-line sonnets from the Gaspara Stampa root work only; 32 other forms and all Veronica Franco material remain excluded. [Source](https://it.wikisource.org/wiki/Rime_(Stampa)) |
| `ws_ariosto_rime_varie_1857` | Ludovico Ariosto, *Opere minori - Rime varie* (1857) | Core | `activated` | V5 contains 29 exact 14-line sonnets; three longer labeled forms remain excluded. [Source](https://it.wikisource.org/wiki/Opere_minori_(Ariosto)/Rime_varie) |
| `ws_sannazaro_rime_disperse` | Jacopo Sannazaro, *Rime disperse* | Core | `activated` | V5 contains 21 exact 14-line sonnets from pinned standalone pages linked by the root index; 16 other lyric forms remain excluded. [Source](https://it.wikisource.org/wiki/Rime_disperse) |
| `ws_belli_sonetti_romaneschi` | Giuseppe Gioachino Belli, *Sonetti romaneschi* | Auxiliary dialectal | `excluded_from_core_language_variety` | The approximately 2,042 Romanesco texts would form about 68% of a combined 3,020-poem v3-plus-Belli corpus, making dialectal language the dominant fine-tuning distribution. The page-level audit was stopped and the source is excluded from core training. It can be reconsidered only as a separately approved dialect-conditioned experiment. [Author page](https://it.wikisource.org/wiki/Autore:Giuseppe_Gioachino_Belli) [Collection](https://it.wikisource.org/wiki/Sonetti_romaneschi) |
| `ws_aretino_sonetti_lussuriosi_1792` | Pietro Aretino, *Sonetti lussuriosi* (1792 edition) | Excluded from strict corpus | `audit_only_explicit_content` | All 26 audited pages failed the project's exact 14-line gate. They remain excluded, and the local audit report contains hashes and metadata but no text samples. [Source](https://it.wikisource.org/wiki/Sonetti_lussuriosi_(edizione_1792)) |
| `biblioteca_italiana_tei_poetry` | Biblioteca Italiana TEI poetry collections | Core candidate | `blocked_pending_terms_and_access` | Biblioteca Italiana reports downloadable XML-TEI texts and broad literary coverage, which could allow high-quality poem segmentation. Direct access and the exact reuse terms for a chosen source must be confirmed before ingestion. [Library description](https://bibliodlcm.web.uniroma1.it/it/biblioteca-italiana) |
| `ws_category_sonetti` | Italian Wikisource `Categoria:Sonetti` | Discovery only | `discovery_only` | The category has thousands of pages but mixes authors, editions, periods, dialects, and non-standard variants. It is an index for named-source discovery, never a bulk-download source. [Category](https://it.wikisource.org/wiki/Categoria:Sonetti) |
| `project_gutenberg_italian_sonnets` | Project Gutenberg Italian poetry records | Excluded for this expansion | `not_prioritized` | The initial audit did not identify a material new collection of original Italian sonnets. The Petrarch result is a translation collection, so it does not add target-language data. |

## Pre-Audit Composition Gate

Before downloading individual texts, every candidate must pass a metadata-level
composition review. Record its language variety, period, register, form,
content, expected document count, estimated character or token contribution,
projected corpus share, author concentration, and expected audit runtime. Then
classify it as core, auxiliary for a concrete conditioned experiment, or
excluded.

Only core-compatible candidates proceed to a full page-level audit by default.
Auxiliary data requires an explicitly approved experiment and remains physically
and logically separate from the core corpus. Excluded sources receive no bulk
retrieval. This prevents a large accessible collection from silently redefining
the model's target language or style.

The committed machine-readable gate is
[`data/metadata/sonnet_composition_shortlist.csv`](../data/metadata/sonnet_composition_shortlist.csv).
It uses the current V4 average of approximately 523 characters and 214 BPE
tokens per sonnet for its size estimates. These are planning estimates, not
claims about final retained text. A regression test requires every source marked
`composition_gate_passed` or `audit_then_include` in the public source manifest to have a matching
`passed_core` shortlist record with the same role and URL.

### Passed Core Cohort

The five passed sources are deliberately heterogeneous in author and source
size. Their preliminary strict-sonnet estimate was 715-757 poems. The completed
page-level audit found 864 eligible candidates. The larger result is primarily
explained by the correction that added Colonna's 212-page `Sonetti spirituali`
subtree to its 125 secular pages.

The estimated serial audit time is 90-150 minutes at the established polite
MediaWiki request rate. The main uncertainty is the source-specific page
layout, particularly Andreini's paired-sonnet pages and Stampa's mixed forms.
This estimate covered the audit, not later snapshot creation or corpus building.

Pietro Bembo and Michelangelo Buonarroti are standard-Italian candidates but
are deferred because the accessible collections are only 75% proofread and
need a dedicated form/edition review. Torquato Tasso's *Rime d'amore* is
excluded from the current plan because its root is marked 25% complete and
contains substantial critical commentary. These are documented in the shortlist
so they cannot be accidentally treated as ready sources.

## Consolidated Audit Results

The committed summary is
[`data/metadata/sonnet_expansion_audit_results.csv`](../data/metadata/sonnet_expansion_audit_results.csv).
It contains the source-specific root revisions, audit timestamps, exact scope,
line-count outcomes, and activation state. Local detailed audit reports remain
uncommitted machine-local inspection artifacts. The activated processed poem
texts, immutable snapshots, manifest, build report, and attribution record are
committed as the reproducible v5 dataset.

| Source | Scoped pages | Candidates | Exact 14 lines | Other line counts | Exact v4 duplicates |
| --- | ---: | ---: | ---: | ---: | ---: |
| Isabella Andreini | 189 | 196 | 196 | 0 | 0 |
| Vittoria Colonna | 337 | 337 | 336 | 1 | 0 |
| Gaspara Stampa | 314 | 314 | 282 | 32 | 0 |
| Ludovico Ariosto | 32 | 32 | 29 | 3 | 0 |
| Jacopo Sannazaro | 37 | 37 | 21 | 16 | 0 |
| **Total** | **909** | **916** | **864** | **52** | **0** |

The strict form and duplicate gates supported activation of all 864 candidates.
The completed v5 build contains 1,875 poems. Vittoria Colonna represents 336
poems, approximately 17.9%; this is substantial but below the
language-variety concentration risk that excluded the Romanesco Belli source.

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

The Varchi activation completed in `sonnets_expanded_v4`: it contains 1,011
poems split into 809 train, 103 validation, and 99 test poems. Belli remains
excluded because Romanesco would dominate a combined core corpus. Aretino
remains excluded because none of its 26 audited pages passes the exact 14-line
gate.

The consolidated page-level audit and v5 activation are complete.
`sonnets_expanded_v5` contains 1,875 poems split into 1,486 train, 191
validation, and 198 test poems. Its committed processed texts, manifest, five
source snapshots, attribution record, and build report preserve the exact
lineage without altering v4 or historical experiments.
