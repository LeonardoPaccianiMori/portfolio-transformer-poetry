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
2. Build the first practical broader corpus probe from curated public-domain
   prose works in Project Gutenberg, because Gutenberg has explicit per-book
   landing pages, public-domain status fields, and stable download metadata.
3. Add Liber Liber works as targeted candidates when the underlying work is out
   of copyright and the manifest records the Liber Liber license layer. Treat
   Liber Liber edition material as separate from the public-domain underlying
   text.
4. Add Italian Wikisource and Biblioteca Italiana candidates after their
   source-specific terms, attribution requirements, and extraction path are
   recorded.

Current practical artifact:

`docs/broader_italian_prose_worklist.md`

This lists candidate prose works from Project Gutenberg, Liber Liber, and
Italian Wikisource, with period, genre, likely source URL, license/status field,
and inclusion priority.

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

Source links:

- https://liberliber.it/
- https://liberliber.it/opere/libri/licenze/
- https://liberliber.it/autori/autori-b/giovanni-boccaccio/
- https://liberliber.it/autori/autori-b/matteo-bandello/
- https://liberliber.it/autori/autori-b/pietro-bembo/

Why it matters:

- large Italian digital library with a clear author/work index;
- many works are classics and out of copyright;
- may provide simpler text/ePub files than raw Wikisource markup;
- contains important prose candidates, including Boccaccio's *Decameron*,
  *Elegia di Madonna Fiammetta*, *Filocolo*, and *Trattatello in laude di
  Dante*; Bandello's *Novelle*; and Bembo's *Gli Asolani* and *Prose della
  volgar lingua*.

Reuse status:

- Liber Liber says e-books can be downloaded and used personally for free.
- It distinguishes works free of copyright from works protected by copyright.
- For works free of copyright, it says the works can be used freely, while
  protected edition elements such as layout and covers are distributed, unless
  otherwise stated, under Creative Commons Attribution-NonCommercial-ShareAlike
  4.0.
- It defines "free of copyright" pragmatically as works whose authors,
  translators, and editors have been dead for more than 70 years, with a note
  that legal exceptions may apply.

Project policy:

- use Liber Liber as a candidate-discovery and fallback text source;
- prefer extracting plain primary text, not cover/layout/synopsis material;
- record both the underlying-work public-domain status and the Liber Liber
  edition/license layer;
- do not use Liber Liber texts marked as protected by copyright;
- do not assume commercial reuse for Liber Liber edition material because of the
  CC BY-NC-SA layer;
- do not bulk scrape the site; use manually approved URLs or a respectful,
  source-aware downloader only after the manifest is defined.

Decision on 2026-07-07:

- active Liber Liber editions explicitly licensed `CC BY-NC-SA 4.0` are
  approved for this project;
- every row must retain source, edition, contributor, and exact license
  attribution;
- this corpus/model lineage is non-commercial and share-alike unless those
  editions are replaced or separately licensed.

Risks:

- some texts may be modern editorial versions;
- period metadata may require manual work;
- files may include headers, volunteer notes, and formatting artifacts;
- non-commercial restrictions on Liber Liber edition elements are not ideal for
  a dataset that might later be published broadly.

Audit questions:

- Are text files or ePubs available for the early Italian works we need?
- Are edition/transcription notes sufficient for provenance?
- Does the site permit automated downloads, or should we manually list approved
  URLs?
- Can the corpus builder cleanly remove Liber Liber site/edition wrapper text
  while retaining attribution metadata?

Recommendation:

Use Liber Liber as a targeted expansion source, not as the first automated bulk
source. It is useful for high-value prose candidates, but the manifest must
preserve license/provenance metadata and mark the edition layer separately.

Initial prose-only Liber Liber candidates:

| Priority | Author | Work | Approx. period | Genre | Inclusion note |
| ---: | --- | --- | --- | --- | --- |
| 1 | Giovanni Boccaccio | *Decameron* | 1349-1353 | prose novelle / frame narrative | Strongest prose candidate; include before later Renaissance prose if licensing and edition metadata are clean. |
| 2 | Giovanni Boccaccio | *Trattatello in laude di Dante* | 14th c. | literary biography / prose | Good old Italian prose; useful style bridge from Dante/Boccaccio period. |
| 3 | Giovanni Boccaccio | *Elegia di Madonna Fiammetta* | 1343-1344 | prose fiction | Strong prose candidate; verify edition source and remove paratext. |
| 4 | Giovanni Boccaccio | *Filocolo* | 1336-1338 | prose romance | Strong but long; useful if edition text is clean. |
| 5 | Matteo Bandello | *Novelle* | 1554/1573 | prose novelle | Later Tier C scale expansion; include only in a labeled 1200-1600 corpus. |
| 6 | Pietro Bembo | *Prose della volgar lingua* | 1525 | linguistic prose dialogue/treatise | Useful for historical Italian prose and language discussion; Tier C. |
| 7 | Pietro Bembo | *Gli Asolani* | 1505/1530 | prose dialogue | Tier C literary prose; include only after core medieval prose. |

Explicit exclusions from this audit pass:

- Ariosto, *Orlando Furioso*: exclude because it is verse, even though it is a
  major historical Italian text.
- Ariosto, *Rinaldo*: exclude because it is verse/fragments, not prose.
- Any *Rime*, canzonieri, ottave, capitoli, or poem collections from Liber
  Liber: exclude from the broader prose pretraining corpus. They may be useful
  only for a separate poetry experiment.

### 4. Project Gutenberg

Source links:

- https://www.gutenberg.org/
- https://www.gutenberg.org/browse/languages/it
- https://www.gutenberg.org/policy/terms_of_use.html
- https://www.gutenberg.org/policy/permission.html
- https://www.gutenberg.org/ebooks/44549
- https://www.gutenberg.org/ebooks/69898
- https://www.gutenberg.org/ebooks/30766
- https://www.gutenberg.org/ebooks/26961
- https://www.gutenberg.org/ebooks/71218

Why it matters:

- strong public-domain clearance tradition under US law;
- explicit landing pages with per-book public-domain status;
- Italian-language browse page makes candidate discovery practical;
- Project Gutenberg's permissions page says most permission requests do not need
  a custom response because most e-books are public domain in the United States,
  including commercial use, derivative works, and republication;
- good fallback for old Italian prose, chronicles, commentaries, treatises, and
  narrative prose.

Reuse status:

- use only Project Gutenberg e-books whose landing page says public domain in
  the USA;
- obey Gutenberg's site rules: use main landing-page links in documentation,
  use mirrors/catalogs for automated downloads, and strip Gutenberg header/footer
  boilerplate from training text;
- record that copyright status is US-based and may differ in other countries.

Risks:

- Italian historical coverage may be uneven;
- editions may be modernized;
- Project Gutenberg trademark and redistribution rules must be respected;
- legal public-domain status can differ by country;
- some strong-looking texts are translations into English or modern Italian and
  should be excluded unless explicitly intended for a different experiment.

Audit questions:

- Which editions are used?
- Do we need to strip Project Gutenberg header/footer boilerplate?
- How should we handle mixed prose/poetry works, such as Dante's *Vita nuova*?

Recommendation:

Use Project Gutenberg as the first practical automated probe for the broader
prose corpus, because the source metadata and public-domain status are explicit.
Start with a small manually curated manifest, estimate text size, then expand.

Initial prose-only Project Gutenberg candidates:

| Priority | Author | Work | PG eBook | Approx. period | Genre | Inclusion note |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | Sidrac / anonymous tradition | *Il libro di Sidrach: testo inedito del secolo XIV* | 44549 | 14th c. text | encyclopedic / philosophical prose | Strong Tier A/B prose candidate; large plain text file; public domain in USA. |
| 2 | Matteo Villani | *Cronica di Matteo Villani*, vols. 1-5 | 69898 and related author page | 14th c. | chronicle prose | Very strong old Italian prose candidate; include all volumes after verifying related eBook IDs. |
| 3 | Jacopo Alighieri | *Chiose alla cantica dell'Inferno* | 30766 | early 14th c. text, later edition | commentary prose | Useful commentary prose; remove modern editor introduction if separable. |
| 4 | Caterina da Siena | *Libro della divina dottrina: Dialogo della divina provvidenza* | 26961 | late 14th c. | mystical/spiritual prose | Strong Tier B prose; source mentions Biblioteca Italiana images, so keep provenance. |
| 5 | Dante Alighieri | *La vita nuova* | 71218 | 1294 | mixed prose and poetry | Do not include as-is in prose corpus; include only if prose sections can be extracted cleanly and poems excluded. |
| 6 | Giovanni Boccaccio | *Decameron* if Italian edition is present | TBD | 1349-1353 | prose novelle | Search Gutenberg catalog directly during manifest creation; if only translations are present, exclude those translations. |
| 7 | Niccolo Machiavelli | Italian prose works if present | author page 563 includes one Italian mixed/dramatic item | 16th c. | political/prose or drama | Defer for Tier C; current visible Italian item is *La mandragola - La Clizia - Belfagor*, which includes drama and is not an ideal prose-pretraining core. |

Explicit exclusions from this audit pass:

- Dante, *Divina Commedia*: exclude because it is poetry.
- Ariosto, *Orlando Furioso*: exclude because it is poetry.
- English translations of Italian works, such as Gutenberg's English *The
  Banquet (Il Convito)*: exclude from Italian pretraining.
- Drama-only works: defer unless we create a separate dialogue/drama corpus.
- Modern novels and essays: defer to a labeled post-1600 scale experiment, not
  the vintage core.

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
- poetry and verse works from the broader pretraining corpus, even if they are
  historically important;
- mixed prose/poetry works unless the corpus builder can extract prose sections
  cleanly and record that extraction decision.

Prose-only rule:

The broader Italian pretraining corpus should start as a prose corpus. The
sonnet corpus already handles poetry specialization. Keeping the broader corpus
prose-only reduces contamination from verse lineation, rhyme, and meter, and
makes the pretraining objective focus on grammar, syntax, discourse coherence,
historical vocabulary, and narrative/expository structure. Poetry can be added
only as a named, separate experiment.

## Proposed Dataset Versions

| Version | Description | Purpose |
| --- | --- | --- |
| `old_italian_core_pre1375` | verified pre-1375 Italian/volgare texts only | period-pure pretraining core |
| `old_italian_wikisource_selected` | curated Wikisource early Italian works | practical first build |
| `old_italian_prose_pg_liberliber_probe` | curated Project Gutenberg and Liber Liber prose candidates | first practical prose-only source probe |
| `historical_italian_1200_1600` | broader medieval/early modern corpus | scale expansion |
| `historical_italian_plus_sonnets` | broader corpus plus sonnet fine-tuning lineage | final from-scratch path |

## Recommended Next Steps

1. Implement a Liber Liber source adapter for the approved Creative Commons
   prose rows.
2. Probe each added work and emit source, period, genre, edition, licensing, and
   attribution metadata.
3. Implement the complete processed-text builder after the measured candidate pool is
   large enough to justify a named corpus version.
4. Train a new BPE tokenizer on the selected broader corpus; do not reuse the
   sonnet-only vocabulary as if it covered the new domain.
5. Verify OVI/TLIO access and reuse terms in parallel, but do not block practical
   corpus expansion on OVI/TLIO.
6. Build a candidate worklist for Italian Wikisource early prose if Gutenberg
   and Liber Liber remain below the target.

## Current Recommendation

The expanded Project Gutenberg probe is complete. Eight prose works produced
4,342,736 cleaned characters and 718,512 whitespace-delimited units; see
`data/metadata/broader_prose_probe_report.json`. This is not enough for the
planned 10M-25M BPE-token first corpus, so the approved Creative Commons source
pool must be measured before the complete corpus build.

The Liber Liber probe is also complete. Eight active prose works produced
7,374,920 extracted characters and 1,278,303 whitespace-delimited units after
generic archive-wrapper removal. Combined measured size is 11,717,656
characters. Exact training size remains provisional until source-specific
paratext cleaning and broader-corpus BPE tokenization are complete.

If OVI/TLIO is reusable, it should be the primary corpus.

For immediate progress, decide whether to expand the source pool before the
processed build. The current character count is likely below the 10M-25M BPE
token target, but this must be confirmed with a tokenizer trained on the final
selected corpus. If OVI/TLIO is not reusable, use labeled Tier C or curated
Wikisource prose with strict period, genre, license, and provenance metadata.
