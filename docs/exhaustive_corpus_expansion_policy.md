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

Checkpoint 6A closes the terms, access, and composition decision for the twelve
previously unresolved registry rows. Its official evidence, decisions, and
composition gates are published in
[`data/metadata/corpus_archive_terms_evidence_v1.csv`](../data/metadata/corpus_archive_terms_evidence_v1.csv),
[`data/metadata/corpus_archive_resolution_v1.csv`](../data/metadata/corpus_archive_resolution_v1.csv),
and
[`data/metadata/corpus_archive_composition_gate_v1.csv`](../data/metadata/corpus_archive_composition_gate_v1.csv).
Six rows may proceed only to bounded metadata/source inventory: ELTeC Italian,
Internet Archive, Gallica, Internet Culturale, BEIC, and MIDIA. This status is
not permission to acquire corpus text. Four rows retain concrete access,
permission, or OCR blockers; BibIt Scrittori d'Italia is closed as scan-only
with high canonical-overlap risk, and Google Books remains discovery-only. The
full accounting and constraints are in the
[`checkpoint-6A report`](../reports/corpus_archive_registry_resolution_v1.md).

Checkpoint 6B completes those six bounded inventories. The normalized
[`candidate ledger`](../data/metadata/corpus_archive_candidate_inventory_v1.csv),
compact [`archive summary`](../data/metadata/corpus_archive_inventory_summary_v1.csv),
and [`checkpoint report`](../reports/corpus_archive_candidate_inventory_v1.md)
account for 114,971 published decision rows and every record filtered outside
the public historical boundary. Exactly 4,634 rows are inactive candidates for
later archive-specific source/rights/quality audits; 4,393 conditioned-language
rows remain outside the standard queue. Complete raw responses remain in the
ignored local cache. No corpus text, activation, V7 split, mixture weight, cache
deletion, or GPU work is part of checkpoint 6B. The next registry step is the
final open-ended archive-discovery pass in checkpoint 6C.

Checkpoint 6C completes that frozen discovery pass with 35 metadata queries
across nine independent repository, dataset, code, and curated-index surfaces.
Its [`query matrix`](../data/metadata/corpus_archive_discovery_queries_v1.csv),
[`official evidence`](../data/metadata/corpus_archive_discovery_evidence_v1.csv),
[`candidate decisions`](../data/metadata/corpus_archive_discovery_decisions_v1.csv),
and [`checkpoint report`](../reports/corpus_archive_discovery_v1.md) resolve 16
candidate or surface decisions. The approved stop rule found material new
source boundaries, so it does not release the project directly to checkpoint
7. ILC-CNR contributes three standard-queue audit candidates: the 4,311,182-word
Rosmini corpus and 40 Bellini letters as capped Ottocento auxiliaries, plus 56
opera libretti from 1636-1705 as a core-compatible historical verse/drama
source. The Oxford Text Archive contributes a 43-record Italian-language item
inventory. At least ten works cover an underrepresented early-modern register.
Codice Pelavicino remains separately
conditioned because it mixes Italian and Latin. Checkpoint 6D must resolve these
bounded source audits before cross-archive canonicalization. No corpus text is
acquired or activated by discovery.

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
  The subsequent extraction and canonicalization audit accounts for the full
  587-record cached extraction scope. Its resolved build materializes 566
  standard sources with 292,353,625 retained source characters, excludes 15
  fully covered editions, removes ten embedded canonical segments while
  preserving surrounding unique text, retains only selected Italian primary
  text from two multilingual editions, and quarantines the protected Cino V6
  sonnet. All 611 candidate fourteen-line windows are resolved: 499 standard
  sonnets are materialized pending V7, four duplicates are excluded, 106
  non-sonnet false positives stay in their broader-text role, and two verified
  Occitan/Milanese sonnets remain conditioned and inactive. Six conditioned
  source records are also stored in a physically separate inactive shard.
  Final exact/near checks cover all 566 standard records and 1,352 BibIt or
  current-corpus references with zero residual pair at the frozen threshold.
  Manifests retain byte ranges, hashes, source rights, and the unresolved
  candidate-level poem-author status. No Gutenberg text has a V7 split,
  training-mixture weight, or Ottocento exposure yet, and no GPU work is active.
- Italian Wikisource: checkpoint 4A inventory and composition gate complete
  against the SHA-1-pinned `20260801` `page`, `categorylinks`, and `linktarget`
  metadata dumps. The inventory accounts for 117,297 main-namespace pages in
  22,165 structural work roots and projects 225,685,176 wikitext bytes. It
  identifies 6,863 provisional historical/Ottocento candidates projecting
  64,834,722 bytes (28.7% of the archive projection), while 382 explicit
  language-variety roots remain conditioned and 14,920 roots remain held,
  excluded, or cross-archive references. A stratified 30-revision inspection
  produces 17 primary-text signals and 13 page-level reviews. Candidate author
  proxies already show concentration risk: Emilio Salgari projects 13.5% of
  candidate bytes before cleaning and deduplication. Wikitext bytes are not
  cleaned characters or tokens, current CC BY-SA 4.0 site terms do not replace
  underlying-work/source-scan verification, and no text is activated. Next,
  resolve bounded metadata/source-scan holds and approve only composition-
  compatible roots for page-level extraction.
- Italian Wikisource checkpoint 4B candidate resolution is complete. The pinned
  `20260801` metadata link graph maps 6,092 of 6,863 candidates to 1,335 exact
  `Indice:` scans; 242 scans support multiple candidate roots, so their shared-
  edition boundaries remain explicit. A metadata-only page-level audit queue
  contains 4,641 roots projecting 16,353,125 wikitext bytes: 1,413 historical-
  general, 1,724 historical non-sonnet poetry, 1,245 Ottocento-bridge, and 259
  standard-sonnet roots. Another 1,447 roots remain held for scan-level
  language conflicts, 771 lack a direct scan link, three link to multiple
  editions, and one links to a redirect. These 2,222 holds collapse into 823
  review units. Of 3,883 checkpoint-4A language-review rows, 2,857 contain only
  citation-index evidence and/or explicit standard Italian; 1,026 retain
  genuine nonstandard or unresolved language evidence. This correction avoids
  propagating false language hazards but promotes no held row. The queue does
  not authorize page-text acquisition or extraction. Next, propose revision-
  pinned page-boundary extraction and the full BibIt/Gutenberg/current-corpus/
  protected-V6 overlap probe for the 4B-eligible queue.
- Liber Liber: complete through its deterministic inactive resolved build.
- Remaining registry archives: checkpoints 6A-6C are complete. The final
  discovery pass added inactive ILC-CNR and Oxford Text Archive boundaries and
  scheduled their bounded checkpoint-6D audits. None of the 4,634 checkpoint-6B
  inventory candidates or the checkpoint-6C discoveries authorizes corpus-text
  download or activation.
