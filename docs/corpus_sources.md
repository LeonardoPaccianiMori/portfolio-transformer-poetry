# Sonnet Corpus Source Audit

This document tracks candidate sources for the Italian sonnet corpus.

Status meanings:

- `candidate`: promising source, not yet approved for training.
- `approved`: source passed license, edition, form, and cleaning review.
- `excluded`: source reviewed and rejected, with reason recorded.

The corpus is limited to sonnets. Non-sonnet poems, prose, editorial notes, commentary, navigation text, and license boilerplate should not enter the primary training text.

## Why This Audit Matters

In a language-modeling project, the dataset is part of the model design. A tiny transformer trained on noisy or poorly documented text can appear to learn style while actually learning archive boilerplate, duplicated poems, editorial commentary, or leaked validation examples.

For this project, each poem should eventually become one metadata row and one cleaned text record. The cleaned text will later be tokenized and converted into PyTorch training batches. The expected tensor shape for token IDs will likely be:

```text
(batch, time)
```

where `batch` is the number of training sequences in a mini-batch and `time` is the context length.

## Inclusion Policy

Primary corpus inclusion requires:

- the poem is a sonnet, not just a generic lyric poem;
- the source URL and archive are recorded;
- license or reuse terms are recorded;
- edition or source notes are recorded when available;
- attribution status is recorded;
- original spelling and punctuation are preserved;
- line breaks are preserved;
- train/validation/test split is assigned by poem, not by line.

Uncertain poems may be kept in metadata but should be excluded from the first clean training split unless explicitly approved.

## Candidate Source Summary

Counts are initial estimates for planning. They must be verified by a source-audit pass and then by the extraction script.

| Priority | Author | Source / Collection | URL | Estimated / audited sonnets | Status | Notes |
|---:|---|---|---|---:|---|---|
| 1 | Giacomo da Lentini | *Poesie* | https://it.wikisource.org/wiki/Poesie_(Giacomo_da_Lentini) | 29 audited candidates | candidate | 27 poems in the explicit `Sonetti` section, plus 2 doubtful-attribution pages that are 14-line sonnets. Keep secure, correspondence, other-author, and doubtful poems separate. |
| 2 | Dante Alighieri | *Rime* | https://it.wikisource.org/wiki/Rime_(Dante) | 118 indexed poems; exact sonnet count requires extraction | candidate | Core early Italian source. The index is mixed-form and includes correspondence poems by other authors. Count sonnets by page-level line-count/form audit. |
| 3 | Guido Cavalcanti | *Rime* | https://it.wikisource.org/wiki/Rime_(Cavalcanti) | mixed index; exact sonnet count requires extraction | candidate | Essential Stilnovo source from a 1902 edition. Includes prefatory matter, non-sonnets, apocrypha/correspondence, and other displayed authors. |
| 4 | Cino da Pistoia | Author page / selected poems | https://it.wikisource.org/wiki/Autore:Cino_da_Pistoia | 123 author-page texts; exact sonnet count requires extraction | candidate | Important bridge toward Petrarch. Author page is useful, but the exact sonnet subset must be derived from categories or line counts. |
| 5 | Cecco Angiolieri | *Rime* | https://it.wikisource.org/wiki/Rime_(Angiolieri) | 129 numbered sonnets | candidate | Explicit sonnet sections: 108 main numbered sonnets plus 21 doubtful-attribution numbered sonnets. One nested response link should be inspected separately. |
| 6 | Folgore da San Gimignano | Author page / sonnet cycles | https://it.wikisource.org/wiki/Autore:Folgore_da_San_Gimignano | 38 likely sonnet texts | candidate | Explicit sonnet-cycle pages: 32 main sonnets plus a doubtful-attribution group. Strong form evidence. |
| 7 | Guittone d'Arezzo | *Rime* | https://it.wikisource.org/wiki/Rime_(Guittone_d%27Arezzo) | 239 explicit sonnet-section candidates; 12 edge cases | candidate | Large source with explicit `Sonetti d'amore` and `Sonetti ascetici e morali` sections. `Trattato d'Amore` should be line-count audited before inclusion. |
| 8 | Francesco Petrarca | *Canzoniere (Rerum vulgarium fragmenta)* | https://it.wikisource.org/wiki/Canzoniere_(Rerum_vulgarium_fragmenta) | 317 sonnets | candidate | Include despite later period because it substantially expands the sonnet corpus and creates a strong Trecento comparison source. Wikisource index is not form-labeled; use canonical 317-sonnet count and verify by line count. |

Planning estimate across these eight sources: approximately **950-1,050 usable sonnets** after filtering, with the final count depending on line-count audits for Dante, Cavalcanti, Cino, and Guittone's `Trattato d'Amore` edge cases.

## Current Build Result

The first full corpus-builder run produced:

- 1,125 manifest rows;
- 921 included processed sonnet files;
- 203 excluded candidate rows;
- no retained `data/raw/` or `data/interim/` directories after successful build.

Included rows by source:

| Source collection | Included sonnets |
|---|---:|
| *Canzoniere (Rerum vulgarium fragmenta)* | 313 |
| *Rime (Guittone d'Arezzo)* | 210 |
| *Rime (Angiolieri)* | 123 |
| *Rime (Dante)* | 84 |
| *Rime (Cino da Pistoia)* | 81 |
| *Rime (Cavalcanti)* | 45 |
| Folgore da San Gimignano sonnet cycles | 36 |
| *Poesie (Giacomo da Lentini)* | 29 |

The generated manifest and build report are stored in `data/metadata/`. Processed poem text files are stored in `data/processed/poems/` and are intended to be committed with attribution metadata.

## Per-Poem Metadata Fields

Each extracted poem should eventually have:

| Field | Purpose |
|---|---|
| `poem_id` | Stable internal identifier. |
| `author` | Normalized author name. |
| `title_or_first_line` | Human-readable title or first line. |
| `collection` | Source collection or page group. |
| `source_url` | Exact page URL. |
| `source_archive` | Archive name, such as Italian Wikisource. |
| `source_edition` | Edition or scan information when available. |
| `license_notes` | Reuse/license notes from the source. |
| `period` | Approximate century or date range. |
| `form` | `sonnet` for included poems. |
| `form_evidence` | Why we know it is a sonnet: index label, page title, line count, editorial note, etc. |
| `attribution_status` | Secure, doubtful, attributed, correspondence, or other. |
| `line_count` | Number of poetic lines after cleaning. |
| `text_path_raw` | Path to raw downloaded source. |
| `text_path_clean` | Path to cleaned poem text. |
| `split` | Train, validation, test, or excluded. |
| `include_in_training` | Boolean inclusion flag. |
| `notes` | Cleaning, source, or uncertainty notes. |

## Audit Method

This initial audit uses source indexes and representative poem pages. Sources with explicit sonnet sections can receive audited candidate counts before writing a downloader. Sources with mixed-form indexes must be counted by the extraction script using page-level evidence such as category labels, section names, and cleaned line count.

The next corpus-builder script should output a machine-readable manifest with one row per candidate poem and a `count_method` field:

- `explicit_index_section`
- `wikisource_category`
- `line_count_14`
- `canonical_external_count`
- `manual_exclusion`

## Audit Notes: Giacomo da Lentini

Source:

- Author / collection: Giacomo da Lentini, *Poesie*
- Archive: Italian Wikisource
- URL: https://it.wikisource.org/wiki/Poesie_(Giacomo_da_Lentini)
- Source edition note from page: Roberto Antonelli, Bulzoni Editore, Roma, 1979
- License notes from page: Creative Commons Attribution-ShareAlike; page metadata also indicates CC BY-SA 3.0 and GFDL
- Current status: `candidate`

### Inclusion Decision

Use this source for the sonnet corpus if extraction confirms clean poem boundaries and line counts.

Include sonnets even when attribution is doubtful, because the project is focused on sonnet-form language modeling rather than a strict single-author corpus. Do not silently treat doubtful poems as secure Giacomo da Lentini poems.

Metadata rule:

- `include_in_training = true` for sonnets unless a specific cleaning or licensing issue is found;
- `attribution_status = secure`, `doubtful`, `correspondence`, or another explicit value;
- `author` should record the poem's displayed author when the poem page gives one;
- `source_collection = Poesie (Giacomo da Lentini)`.

### Form Evidence

The collection page has a clear `Sonetti` section. This is the primary form evidence for the first extraction pass.

Each extracted poem should still be checked for:

- 14 poetic lines after cleaning;
- preserved line breaks;
- absence of non-poem text in the cleaned record.

### Cleaning Decisions

Raw text must remain unchanged.

Processed text should:

- remove Wikisource navigation, page chrome, headers, footers, and license boilerplate;
- remove displayed line-number markers such as `4`, `8`, `11`, and `14`;
- preserve poem line breaks;
- preserve original spelling and punctuation except for approved editorial-bracket handling;
- remove square brackets around editorial letter expansions, for example `ben[e]` becomes `bene`.

Record bracket handling in metadata:

- `editorial_brackets_removed = true`
- `normalization_notes = "Removed square brackets around editorial letter expansions; raw text preserved separately."`

### Example Extraction URLs

Use these pages as early extraction tests:

- https://it.wikisource.org/wiki/Poesie_(Giacomo_da_Lentini)/Sonetti/Io_m%27aggio_posto_in_core_a_Dio_servire
- https://it.wikisource.org/wiki/Poesie_(Giacomo_da_Lentini)/Sonetti/Amor_%C3%A8_uno_desio_che_ven_da_core
- https://it.wikisource.org/wiki/Poesie_(Giacomo_da_Lentini)/Sonetti/Oi_deo_d%27amore

### Open Checks Before Approval

- Confirm whether all included sonnet pages use the same layout and line-number convention.
- Confirm whether the Wikisource edition/license notes are sufficient for redistribution of cleaned derived text, or whether the public repo should distribute only scripts and metadata while ignoring downloaded raw/processed text.

### Audited Count

The source page contains 27 poems in the explicit `Sonetti` section.

Breakdown:

| Category | Count | Inclusion policy |
|---|---:|---|
| Ordinary Giacomo da Lentini sonnets, numbered XX-XXXVIII | 19 | Include as `attribution_status = secure` unless a poem page says otherwise. |
| Tenzone with the Abate di Tivoli, XVIIIa-XVIIIe | 5 | Include as correspondence sonnets; preserve displayed author. |
| Tenzone with Jacopo Mostacci and Pier della Vigna, XIXa-XIXc | 3 | Include as correspondence sonnets; preserve displayed author. |
| Doubtful-attribution pages that are 14-line sonnets | 2 | Include as `attribution_status = doubtful`. |
| Doubtful-attribution page that is not a sonnet | 1 | Exclude from primary sonnet corpus. |

Total candidate sonnets from this source: **29**.

The explicit `Sonetti` section includes five poems displayed as authored by someone other than Giacomo da Lentini:

- XVIIIa, `Oi deo d'amore`, by Abate di Tivoli
- XVIIIc, `Qual om riprende altrù'`, by Abate di Tivoli
- XVIIIe, `Con vostro onore facciovi uno 'nvito`, by Abate di Tivoli
- XIXa, `Solicitando un poco meo savere`, by Iacopo Mostacci
- XIXb, `Però c'Amore non si pò vedere`, by Pier della Vigna

These should be included if they are sonnets, but the poem-level `author` field should record the displayed author, not automatically `Giacomo da Lentini`.

The three `Dubbie attribuzioni` pages are:

| Page | Line-count audit | Inclusion policy |
|---|---:|---|
| https://it.wikisource.org/wiki/Poesie_(Giacomo_da_Lentini)/Dubbie_attribuzioni/Membrando_l%27amoroso_dipartire | 45 lines | Exclude from primary sonnet corpus. |
| https://it.wikisource.org/wiki/Poesie_(Giacomo_da_Lentini)/Dubbie_attribuzioni/Lo_badalisco_a_lo_specchio_lucente | 14 lines | Include as doubtful sonnet. |
| https://it.wikisource.org/wiki/Poesie_(Giacomo_da_Lentini)/Dubbie_attribuzioni/Guardando_basalisco_velenoso | 14 lines | Include as doubtful sonnet. |

## Audit Notes: Dante Alighieri

Source:

- Author / collection: Dante Alighieri, *Rime*
- Archive: Italian Wikisource
- URL: https://it.wikisource.org/wiki/Rime_(Dante)
- Source date metadata: XIII secolo
- License notes from page: CC BY-SA 3.0 and GFDL metadata; footer also states Creative Commons Attribution-ShareAlike with possible additional terms
- Current status: `candidate`

### Inclusion Decision

Use this source, but do not rely on the collection index alone to identify sonnets.

The page indexes 118 numbered poems across sections such as `Rime della Vita Nuova`, `Rime allegoriche e dottrinali`, `Altre rime d'amore e di corrispondenza`, `Rime per la donna pietra`, and `Rime varie del tempo dell'esilio`. The index includes non-sonnet forms and correspondence poems by other displayed authors.

Metadata rule:

- include only records that the extraction audit identifies as sonnets;
- preserve displayed author when the index marks a poem as by another poet, such as Dante da Maiano, Cino da Pistoia, Cecco Angiolieri, Guelfo Taviani, or Aldobrandino Mezzabati;
- use `source_collection = Rime (Dante)`;
- set `attribution_status = correspondence` for poems included because they are part of a Dante correspondence sequence but are displayed as authored by someone else.

### Count Status

The source has **118 indexed poems**, but the exact sonnet count is **not audited yet** because the index is mixed-form. Final count should come from page-level extraction using 14-line checks and category/form evidence.

## Audit Notes: Guido Cavalcanti

Source:

- Author / collection: Guido Cavalcanti, *Rime*
- Archive: Italian Wikisource
- URL: https://it.wikisource.org/wiki/Rime_(Cavalcanti)
- Source edition: Ercole Rivalta, Zanichelli, Bologna, 1902
- Source date metadata: XIII secolo
- License notes from page: CC BY-SA 3.0 and GFDL metadata; footer also states Creative Commons Attribution-ShareAlike with possible additional terms
- Current status: `candidate`

### Inclusion Decision

Use this source after page-level form filtering.

The index is not a clean sonnet-only index. It includes prefatory/source-critical material, `Il trattato d'amore`, poems grouped by chronology, uncertain-period poems, posterior poems, and poems displayed as authored by correspondents such as Guido Orlandi, Gianni Alfani, and Bernardo da Bologna.

Metadata rule:

- include only pages classified as sonnets by line count or category/form evidence;
- preserve apocryphal, correspondence, and displayed-author information;
- use `source_collection = Rime (Cavalcanti)`;
- record `source_edition = Ercole Rivalta, Zanichelli, Bologna, 1902`.

### Count Status

Exact sonnet count is **not audited yet**. The final count should be produced by the extraction script because the source is a mixed-form edition rather than an explicit sonnet index.

## Audit Notes: Cino da Pistoia

Source:

- Author page: Cino da Pistoia
- Archive: Italian Wikisource
- URL: https://it.wikisource.org/wiki/Autore:Cino_da_Pistoia
- Related collection: https://it.wikisource.org/wiki/Rime_scelte_di_M._Cino_da_Pistoia
- Related source scan/index: https://it.wikisource.org/wiki/Indice:Le_Rime_di_Cino_da_Pistoia.djvu
- Current status: `candidate`

### Inclusion Decision

Use this source, but treat it as a page/category-driven extraction source rather than a single clean collection.

The author page lists **123 texts by Cino da Pistoia**, but it does not itself prove that all 123 are sonnets. Search results and representative pages show useful category evidence such as `Sonetti`, `Testi di Cino da Pistoia`, and `Testi del XIV secolo`.

Metadata rule:

- prefer poem pages that carry a `Sonetti` category or pass a 14-line audit;
- keep the 1862 *Rime scelte* edition notes when poems are sourced from that collection;
- record `source_collection = Rime (Cino da Pistoia)` or `Rime scelte di M. Cino da Pistoia` depending on the page source;
- exclude poems that do not pass form checks.

### Count Status

Exact sonnet count is **not audited yet**. Planning estimate remains around **100-123** candidate sonnets until the page-level manifest is generated.

## Audit Notes: Cecco Angiolieri

Source:

- Author / collection: Cecco Angiolieri, *Rime*
- Archive: Italian Wikisource
- URL: https://it.wikisource.org/wiki/Rime_(Angiolieri)
- Source date metadata: XIII secolo
- License notes from page: CC BY-SA 3.0 and GFDL metadata; footer also states Creative Commons Attribution-ShareAlike with possible additional terms
- Current status: `candidate`

### Inclusion Decision

Use this source. It has explicit sonnet sections and is strong for comic-realistic contrast.

Breakdown:

| Category | Count | Inclusion policy |
|---|---:|---|
| `I sonetti di Cecco Angiolieri`, numbered I-CVIII | 108 | Include as main Cecco sonnets unless page-level extraction finds a form issue. |
| `Sonetti di dubbia attribuzione`, numbered CIX-CXXIX | 21 | Include as doubtful sonnets and preserve `attribution_status = doubtful`. |
| Nested response link under CVII | 1 | Inspect separately before inclusion. |

Total numbered candidate sonnets: **129**.

Metadata rule:

- use `source_collection = Rime (Angiolieri)`;
- record `attribution_status = secure` for I-CVIII and `doubtful` for CIX-CXXIX;
- inspect the nested response page separately rather than silently merging it into the numbered count.

## Audit Notes: Folgore da San Gimignano

Source:

- Author page: Folgore da San Gimignano
- Archive: Italian Wikisource
- URL: https://it.wikisource.org/wiki/Autore:Folgore_da_San_Gimignano
- Current status: `candidate`

### Inclusion Decision

Use this source. It has strong form evidence because the author page groups works under `Sonetti` and the cycle pages are explicitly sonnet cycles.

Breakdown:

| Group | URL | Count | Inclusion policy |
|---|---|---:|---|
| `Sonetti dei mesi` | https://it.wikisource.org/wiki/Sonetti_dei_mesi | 14 | Include. |
| `Sonetti per l'armamento di un cavaliere` | https://it.wikisource.org/wiki/Sonetti_per_l%27armamento_di_un_cavaliere | 5 | Include. |
| `Sonetti della "Semana"` | https://it.wikisource.org/wiki/Sonetti_della_%22Semana%22 | 8 | Include. |
| `Sonetti politici e moraleggianti` | https://it.wikisource.org/wiki/Sonetti_politici_e_moraleggianti | 5 | Include. |
| `Sonetti di dubbia attribuzione` | https://it.wikisource.org/wiki/Sonetti_di_dubbia_attribuzione | 6 likely | Include as doubtful after page-level count confirmation. |

Total likely candidate sonnets: **38**.

Metadata rule:

- record cycle name in `collection` or `source_subcollection`;
- record `form_evidence = explicit sonnet-cycle page`;
- preserve thematic grouping because these poems differ from love/Stilnovo sonnets.

## Audit Notes: Guittone d'Arezzo

Source:

- Author / collection: Guittone d'Arezzo, *Rime*
- Archive: Italian Wikisource
- URL: https://it.wikisource.org/wiki/Rime_(Guittone_d%27Arezzo)
- Source edition: Francesco Egidi, Laterza, Bari, 1940
- Source date metadata: XIII secolo
- License notes from page: CC BY-SA 3.0 and GFDL metadata; footer also states Creative Commons Attribution-ShareAlike with possible additional terms
- Current status: `candidate`

### Inclusion Decision

Use this source, but keep love sonnets and ascetic/moral sonnets separate because they are stylistically different.

Breakdown:

| Category | Count | Inclusion policy |
|---|---:|---|
| `Sonetti d'amore`, numbered 1-138 | 138 | Include as explicit sonnet-section candidates. |
| `Sonetti ascetici e morali`, numbered 139-239 | 101 | Include as explicit sonnet-section candidates, but preserve subcollection label. |
| `Trattato d'Amore`, numbered 240-251 | 12 | Treat as edge cases; include only after page-level line-count/form confirmation. |

Total explicit sonnet-section candidates: **239**.

Metadata rule:

- use `source_collection = Rime (Guittone d'Arezzo)`;
- record `source_edition = Francesco Egidi, Laterza, Bari, 1940`;
- preserve `source_subcollection = Sonetti d'amore` or `Sonetti ascetici e morali`;
- do not merge `Trattato d'Amore` into the primary sonnet corpus without line-count audit.

## Audit Notes: Francesco Petrarca

Source:

- Author / collection: Francesco Petrarca, *Canzoniere (Rerum vulgarium fragmenta)*
- Archive: Italian Wikisource
- URL: https://it.wikisource.org/wiki/Canzoniere_(Rerum_vulgarium_fragmenta)
- Source date metadata: XIV secolo
- License notes from page: CC BY-SA 3.0 and GFDL metadata; footer also states Creative Commons Attribution-ShareAlike with possible additional terms
- Current status: `candidate`

### Inclusion Decision

Use this source as part of the full corpus. Because Petrarca contributes a large block of sonnets, preserve source and author metadata so we can measure whether the model becomes overly Petrarchan.

The Wikisource index lists all 366 poems but does not label form per item. Britannica gives the canonical count of **317 sonnets** in the *Canzoniere*. The extraction script should verify line counts and exclude canzoni, sestine, ballate, and madrigals.

Metadata rule:

- use `source_collection = Canzoniere (Rerum vulgarium fragmenta)`;
- set `period = XIV secolo`;
- set `form_evidence = canonical_external_count + line_count_14` after extraction;
- preserve whether a poem belongs to `Rime in vita di Madonna Laura` or `Rime in morte di Madonna Laura`.

### Dataset Balance Decision

Create at least two dataset versions:

- `core_pre_petrarch`: sources before Petrarca, useful for earlier sonnet style.
- `expanded_with_petrarch`: all approved sources including Petrarca.

This is not a postponement. It is an explicit experiment design choice: compare whether adding a large Petrarca block improves form quality while changing style distribution.
