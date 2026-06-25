# Broader Italian Pretraining Corpus Source Audit

This document tracks candidate sources for a broader Italian pretraining corpus.

The goal is to build a from-scratch "vintage Italian" language model before
fine-tuning on the curated sonnet corpus.

The sonnet corpus remains the final task corpus. This broader corpus is a
first-stage language-modeling corpus intended to teach general Italian,
historical vocabulary, syntax, prose structure, and literary texture.

## Current Decision

The broader pretraining corpus has a different repository policy from the sonnet
corpus.

Commit to Git:

- source-audit documents;
- source manifests;
- provenance metadata;
- download/build scripts;
- token-count reports;
- evaluation reports.

Do not commit to normal Git by default:

- large downloaded raw archives;
- large interim extraction files;
- large processed pretraining text files;
- large encoded token arrays.

If the processed broader corpus becomes small enough to review and commit
comfortably, this policy can be revisited. Otherwise, use reproducible build
scripts, Git LFS, release artifacts, or external artifact storage after an
explicit decision.

## Access And Reuse Follow-Up

Status date: 2026-06-25.

This section records the first follow-up on the two blocking source-policy
questions:

1. Can OVI/TLIO be used for model training?
2. Can Italian Wikisource CC BY-SA text be used as a fallback source?

### OVI / TLIO Status

Current status: `blocked_pending_permission`.

What public sources confirm:

- TLIO is the best conceptual match for this project because it is described as
  a corpus of Italian texts before 1375.
- A public corpus overview reports about 1,780 TLIO texts and about 20M words,
  including prose and poetry.
- The TLIO page describes GATTO as an online system that shows the corpus text
  used by the dictionary editors.

What is not confirmed:

- official bulk download route;
- explicit permission for model training;
- permission to redistribute processed derived corpus text;
- database/edition-specific reuse constraints;
- whether automated access is acceptable.

Browsing note:

- Attempts to open the official OVI/TLIO web pages from the browsing tool timed
  out during this audit pass.
- Secondary sources link to the official TLIO and OVI pages, but secondary
  sources are not enough to approve reuse for training.

Decision:

Do not use OVI/TLIO as training data until official terms or direct permission
are obtained.

Allowed now:

- use OVI/TLIO as evidence that enough old Italian text likely exists;
- use it as a source-discovery reference;
- record its reported scale and period coverage in planning documents.

Not allowed yet:

- bulk download;
- scraping;
- training on extracted TLIO text;
- committing TLIO-derived processed text.

Recommended action:

Find an official contact or access page and ask whether noncommercial academic
model training is allowed, whether bulk text access exists, and whether local
derived training text may be stored but not redistributed.

### Italian Wikisource Status

Current status: `approved_fallback_with_attribution_policy`.

What public sources confirm:

- Italian Wikisource is a digital library containing public-domain texts or
  freely licensed texts.
- Its collection description says texts are either public domain or under
  CC BY-SA.
- Wikimedia provides bulk dumps for Italian Wikisource. The latest dump listing
  includes `itwikisource-latest-pages-articles.xml.bz2`.

Policy decision:

Italian Wikisource can be used as the practical fallback source, including
CC BY-SA pages, if we preserve attribution and license metadata.

Required metadata for each Wikisource-derived text:

- source page URL;
- title;
- author or anonymous status;
- Wikisource page revision or dump date if available;
- source archive: `Italian Wikisource`;
- source license label, such as public domain or CC BY-SA;
- source edition or scan metadata when available;
- extraction date;
- cleaning notes;
- whether text was modified during cleaning.

CC BY-SA handling:

- prefer public-domain pages where possible;
- include CC BY-SA pages only when attribution/license metadata is recorded;
- do not present CC BY-SA text as public domain;
- public reports should state that some training text may come from CC BY-SA
  Wikisource pages if such pages are included;
- processed large corpus text should not be committed to normal Git by default.

Reasoning:

Creative Commons BY-SA allows sharing and adaptation, including commercial use,
when attribution, license link, change notices, and share-alike requirements are
respected. This is compatible enough for this project if we treat attribution
and license metadata as part of the dataset, not as an afterthought.

### Practical Source Direction

Use this order:

1. Keep OVI/TLIO as the ideal target, but blocked pending official reuse clarity.
2. Build the first practical broader corpus probe from curated Italian
   Wikisource pages.
3. Add Liber Liber, Project Gutenberg, and Biblioteca Italiana only after their
   source-specific terms are audited.

Next practical artifact:

`docs/broader_italian_wikisource_worklist.md`

This should list candidate authors and works from Italian Wikisource, with
period, genre, likely source URL, license/status field, and inclusion priority.

## Why This Corpus Exists

The current sonnet corpus is too small to train a realistic language model from
scratch by itself.

Current sonnet-corpus scale:

| Split / unit | Count |
| --- | ---: |
| Train poems | 736 |
| Train character tokens | about 383k |
| Train BPE tokens | about 157k |

This is enough for learning the PyTorch pipeline and for a sonnet fine-tuning
stage. It is not enough for a from-scratch model to learn stable Italian grammar,
syntax, and semantic coherence.

The new training plan is:

1. Train from scratch on broader public-domain Italian text.
2. Continue training or fine-tune on curated sonnets.
3. Compare against the sonnet-only from-scratch baseline.
4. Compare against an externally pretrained local model fine-tuned on the same
   sonnet corpus.

## Period Policy

Use a tiered period policy:

| Tier | Period | Role | Preference |
| --- | --- | --- | --- |
| A | origins to 1375 | ideal "vintage Italian" pretraining core | highest |
| B | 1376-1400 | close Trecento expansion | high |
| C | 1401-1600 | historical Italian expansion if needed for scale | medium |
| D | post-1600 | fallback only for scale experiments | low |

The strongest project story is Tier A plus carefully labeled Tier B. Tier C is
acceptable if the model needs more data, but it should be reported separately as
broader historical Italian rather than strict old Italian.

## Target Scale

Practical target:

| Stage | Target |
| --- | ---: |
| first usable broader corpus | 10M-25M BPE tokens |
| stronger local pretraining corpus | 50M-100M BPE tokens |
| stretch corpus if sources allow | 100M+ BPE tokens |

The first serious model target can be 70M parameters if we accept slow training.
Data volume should drive the final size decision. A larger model trained on too
little text can memorize or produce unstable outputs.

## Candidate Source Summary

| Priority | Source | Period fit | Estimated scale | Access | Reuse status | Current recommendation |
| ---: | --- | --- | ---: | --- | --- | --- |
| 1 | OVI / TLIO old Italian corpus | excellent: origins to 1375 | about 1,780 texts / 20M words reported | online consultation; bulk access unclear | must verify | Audit first; best conceptual fit, uncertain practical reuse. |
| 2 | Italian Wikisource | mixed, filterable by work/author/period | large; current dump around hundreds of MB compressed | public Wikimedia dumps and page API | public domain or CC BY-SA, per work | Use as first practical retrieval route after filtering and license metadata. |
| 3 | Liber Liber / Progetto Manuzio | mixed; many classics and older public-domain works | thousands of works reported | website downloads, often text/RTF/ePub/PDF | must verify per work and site terms | Good fallback/expansion source after terms audit. |
| 4 | Project Gutenberg | mixed; public-domain clearance under US law | many Italian works, but less period-targeted | public downloads/API-style mirrors | public-domain clearance, plus trademark/use rules | Useful for selected public-domain Italian classics. |
| 5 | Biblioteca Italiana / scholarly digital editions | likely strong for canonical texts | unknown until access audit | site-specific | must verify | Candidate for high-quality editions if reuse permits. |
| 6 | Vulgaris-style 1200-1600 corpus direction | excellent conceptual precedent | paper describes 1200-1600 resources | research paper; dataset access to verify | must verify | Use as methodological reference and possible source lead. |

## Source Notes

### 1. OVI / TLIO Old Italian Corpus

Source links:

- https://it.wikipedia.org/wiki/Tesoro_della_lingua_italiana_delle_Origini
- https://it.wikipedia.org/wiki/Corpus

Why it matters:

- best match for a "vintage Italian" language model;
- covers Italian texts from the origins to the symbolic cutoff of Boccaccio's
  death in 1375;
- reported as about 1,780 texts and about 20M words;
- includes prose and poetry;
- includes different old Italian varieties, not only Tuscan.

Risks:

- bulk download/access is unclear;
- reuse for model training is unclear;
- corpus text may be under database, edition, or site-specific terms even where
  the underlying medieval works are public domain;
- extraction may be technically harder than Wikisource.

Audit questions:

- Is there an official bulk access route?
- Are terms compatible with model training?
- Can we store derived processed text locally?
- Can public repo users rebuild the corpus?
- Is attribution required at corpus, text, or edition level?

Recommendation:

Treat OVI/TLIO as the ideal source, but do not build on it until access and reuse
terms are verified. If bulk reuse is not allowed, use it as a reference for
source discovery and period coverage, not as training data.

### 2. Italian Wikisource

Source links:

- https://it.wikipedia.org/wiki/Wikisource
- https://dumps.wikimedia.org/itwikisource/latest/

Why it matters:

- practical bulk retrieval path through Wikimedia dumps;
- public documentation says Wikisource hosts public-domain or freely licensed
  texts;
- Italian Wikisource has many texts and page metadata;
- current dump files are available, including a `pages-articles.xml.bz2` dump.

Risks:

- not all texts are period-compatible;
- some transcriptions use modern editions, notes, commentary, navigation, and
  templates;
- CC BY-SA works require attribution and share-alike consideration;
- filtering by historical period is not trivial from dump text alone;
- page markup must be cleaned carefully.

Audit questions:

- Which authors/works before 1375 are available and complete?
- Which pages have scan-backed proofread status?
- Can we extract clean text without commentary and page chrome?
- How should we preserve attribution for CC BY-SA pages?
- Should CC BY-SA texts be included in training data for this public project, or
  should we restrict to public-domain pages only?

Recommendation:

Use Wikisource as the first practical build target if OVI/TLIO bulk reuse is
blocked or unclear. Start with a curated list of early authors/works, not the
whole dump.

Candidate work groups:

- Dante: *Vita nuova*, *Convivio*, *De vulgari eloquentia* only if Italian
  translation policy is clear, *Commedia* if accepted for broader corpus.
- Boccaccio: *Decameron* and selected volgare works.
- Petrarch: volgare works, especially non-sonnet material if available.
- early chronicles and volgarizzamenti where source metadata is clear.

### 3. Liber Liber / Progetto Manuzio

Source link:

- https://it.wikipedia.org/wiki/Liber_Liber

Why it matters:

- large Italian digital library;
- reported to contain thousands of digitized works;
- many works are classics and out of copyright;
- may provide simpler text/ePub files than raw Wikisource markup.

Risks:

- site terms and per-work reuse need direct verification;
- some texts may be modern editorial versions;
- period metadata may require manual work;
- files may include headers, volunteer notes, and formatting artifacts.

Audit questions:

- What are the exact reuse terms for downloaded texts?
- Are text files or ePubs available for the early Italian works we need?
- Are edition/transcription notes sufficient for provenance?
- Does the site permit automated downloads, or should we manually list approved
  URLs?

Recommendation:

Use Liber Liber as a targeted expansion source after terms are verified. Do not
bulk scrape until access policy is known.

### 4. Project Gutenberg

Source links:

- https://en.wikipedia.org/wiki/Project_Gutenberg
- https://www.gutenberg.org/

Why it matters:

- strong public-domain clearance tradition under US law;
- easy downloads;
- useful fallback for canonical Italian works.

Risks:

- Italian historical coverage may be uneven;
- editions may be modernized;
- Project Gutenberg trademark and redistribution rules must be respected;
- legal public-domain status can differ by country.

Audit questions:

- Which Italian-language medieval/Trecento works are present?
- Are the Italian texts original-language or translations?
- Which editions are used?
- Do we need to strip Project Gutenberg header/footer boilerplate?

Recommendation:

Use for selected texts, not as the main corpus source.

### 5. Biblioteca Italiana And Scholarly Digital Editions

Source examples to audit:

- Biblioteca Italiana;
- university-hosted editions;
- author-specific scholarly projects.

Why it matters:

- likely higher-quality editions for canonical Italian literature;
- may include reliable metadata and TEI/XML.

Risks:

- reuse may be restricted;
- download access may be limited;
- critical editions may have separate rights even when the underlying text is
  medieval.

Recommendation:

Audit for high-quality specific works. Do not assume reusable training rights.

### 6. Vulgaris-Style 1200-1600 Direction

Source link:

- https://arxiv.org/abs/2010.05993

Why it matters:

- shows a relevant research direction for medieval and early modern Italian
  varieties;
- 1200-1600 is a useful fallback range if strict pre-1375 data is too small;
- supports the idea of author/time/style metadata in the corpus.

Risks:

- the paper is not itself a reusable corpus license;
- dataset availability and terms need separate verification.

Recommendation:

Use as methodological guidance and a source-discovery lead.

## Initial Inclusion Rules

Include a text only when we can record:

- title;
- author or anonymous status;
- approximate date or period bucket;
- source URL;
- source archive;
- source edition or transcription note;
- license/reuse status;
- genre if known;
- language/variety if known;
- whether the text is prose, poetry, mixed, commentary, translation, or
  documentary.

Exclude or defer:

- texts with unclear reuse rights;
- modern commentary without explicit approval;
- translations into modern Italian unless they are intentionally part of a later
  comparison corpus;
- pages where the primary text cannot be separated from notes or navigation;
- duplicated works from multiple editions unless an edition policy is defined.

## Proposed Dataset Versions

| Version | Description | Purpose |
| --- | --- | --- |
| `old_italian_core_pre1375` | verified pre-1375 Italian/volgare texts only | period-pure pretraining core |
| `old_italian_wikisource_selected` | curated Wikisource early Italian works | practical first build |
| `historical_italian_1200_1600` | broader medieval/early modern corpus | scale expansion |
| `historical_italian_plus_sonnets` | broader corpus plus sonnet fine-tuning lineage | final from-scratch path |

## Recommended Next Steps

1. Verify OVI/TLIO access and reuse terms.
2. Build a candidate worklist for Italian Wikisource early texts.
3. Decide whether CC BY-SA Wikisource text is acceptable for training and public
   project documentation.
4. Define a broader-corpus manifest schema, separate from but compatible with
   the sonnet manifest.
5. Implement a small source-probe script that counts candidate Wikisource pages
   and estimates cleaned text size before downloading any large dump.

## Current Recommendation

Start with source audit and metadata only. Do not download or build the broader
corpus until the OVI/TLIO access question and the Wikisource CC BY-SA policy
question are resolved.

If OVI/TLIO is reusable, it should be the primary corpus.

If OVI/TLIO is not reusable or is practically inaccessible, use a curated
Wikisource/Liber Liber/Gutenberg corpus with strict period labels and provenance.
