# Exhaustive Corpus Expansion Policy

## Objective

The corpus track now aims to maximize **eligible, usable, non-duplicate Italian
training tokens**, rather than stopping after a shortlist of famous works. This
applies equally to:

1. historical general text;
2. historical non-sonnet poetry;
3. verified sonnets for specialization and evaluation.

"Exhaustive" has an operational meaning: enumerate every record exposed by the
major reusable archives in the committed archive registry, record every archive
checked, and continue archive discovery until the next pass finds no material
new eligible source. It cannot mean a provable crawl of every page on the web.

## Candidate Pool Versus Training Corpus

Maximizing the candidate pool does not mean concatenating every download.
Every candidate must retain source and license evidence, then pass:

- primary-text availability and non-empty extraction;
- Italian language-variety review;
- editorial-apparatus and OCR-quality review;
- exact and near-duplicate edition checks;
- author, source, period, genre, and form concentration measurement;
- sonnet train/validation/test leakage checks.

The project keeps rejected and blocked sources in the registry so a failed
archive is not repeatedly rediscovered and silently reconsidered.

## Three Corpus Outputs

### Historical General Text

Includes prose, narrative, theatre, letters, treatises, histories, documents,
memoirs, translations/volgarizzamenti, and other usable historical registers.
Core coverage is origins through 1800. Selected standard-literary nineteenth-
century text is retained as a separately capped bridge.

### Historical Non-Sonnet Poetry

Includes long poems and all other eligible non-sonnet poetic forms. Canonical
works such as the *Commedia*, *Orlando innamorato*, *Orlando Furioso 1532*, and
*Gerusalemme liberata* enter this stage. Mixed lyric collections enter only
after explicit sonnet units have been removed.

### Sonnets

Every archive is searched for explicit or structurally verifiable sonnets, not
only famous collections. A poem enters only after form, line structure,
language variety, duplicate status, and provenance are verified. Existing V6
validation/test identities remain held out from every training stage. Training
sampling is balanced by author and epoch so a large collection such as Tasso's
*Rime* cannot dominate merely because it is easy to extract.

## Staged Curriculum

1. Historical general adaptation with limited modern/instruction preservation replay.
2. Historical non-sonnet poetry adaptation with historical prose and preservation replay.
3. Low-learning-rate sonnet specialization with author/epoch-balanced sampling.

The nineteenth-century bridge candidate pool may be large, including Manzoni's
*Promessi Sposi*, Nievo's *Confessioni di un Italiano*, Leopardi's *Zibaldone*,
and Artusi's *La scienza in cucina*. Its exposure share remains separately
capped; the current recommendation is at most 10%, with the exact value frozen
after full cleaning and deduplication. Long works are measured individually so
random sample medians do not hide author/source concentration.

## Archive Order

1. Complete canonical BibIt TEI extraction across all assigned roles.
2. Enumerate all Italian Project Gutenberg records and deduplicate against BibIt.
3. Enumerate Italian Wikisource work roots from API/dump metadata.
4. Enumerate the complete Liber Liber catalog under its non-commercial share-alike terms.
5. Audit ELTeC, Internet Archive, Gallica, Internet Culturale, BEIC, OVI/TLIO,
   MIDIA, DiaCORIS, and other registry candidates for exact permission and bulk access.
6. Use scan OCR only when corrected text is unavailable, rights are explicit,
   and measured OCR quality passes a separate gate.

The canonical archive list and status are maintained in
[`data/metadata/corpus_archive_expansion_registry.csv`](../data/metadata/corpus_archive_expansion_registry.csv).
No GPU training restarts until this expansion, cleaning, deduplication, split,
and final mixture freeze are complete.

## Current Coverage

- Biblioteca Italiana: role-specific TEI audit, review resolution, and bounded
  processed build complete.
- Project Gutenberg Italian: all 1,112 records in the 2026-08-09 Gutendex
  snapshot enumerated with conservative metadata-only routing and overlap
  signals. A 42-text deterministic value gate passed coarse availability,
  encoding, Italian-language, and alphabetic-content checks for every sample;
  it projects approximately 215.5 million cleaned characters across 416
  eligible probe candidates. The subsequent complete probe acquired all 416
  texts and measured 244.4 million cleaned characters. It resolves the flagged
  language and numeric-content cases, separates full-source duplicates from
  embedded duplicate segments, and detects one protected held-out-sonnet
  occurrence. Fifteen full-source duplicate candidates and one macaronic work
  remain outside the core; six embedded-work sources, one held-out-overlap
  source, and two bilingual/multilingual editions require source-specific
  extraction. The frozen metadata queue contains 673 records: 563 work-date,
  79 missing-period, 25 translation-edition, and six language-variety reviews.
  Metadata-resolution pass 1A acquired all 673 primary texts without error and
  conservatively resolved 309 records from direct evidence. Authoritative pass
  1B then closed all 364 holds through SBN/ICCU, strictly corroborated Wikidata,
  or narrow primary-text evidence. Its final accounting contains 415 eligible
  probes, four separately conditioned records, 88 post-1900 exclusions, and
  166 documented unresolved exclusions. The subsequent pass-1B full-text probe
  audits exactly 167 `pass_1b`/`eligible_probe` records from cache: 56,807,893
  cleaned characters, no exact or near duplicates internally, no overlap above
  the frozen threshold with the prior 416 Gutenberg probes, BibIt, or existing
  corpora, and no protected V6 held-out-sonnet overlap. Five bounded anomalies
  are resolved; one Venetian-dialect poetry source is outside the standard core.
  No Gutenberg text, V7 split, or mixture weight is active.
- Italian Wikisource, Liber Liber, and the remaining registry archives: archive-
  scale inventory pending.
