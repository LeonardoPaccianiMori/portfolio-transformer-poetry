# PAISÀ Corpus Attribution And Local-Use Record

## Status

PAISÀ passed the release, provenance, and local-disk activation preflight and
was locally acquired on 2026-07-30. It is approved only for the defined
non-commercial PAISÀ to historical-prose to V5-sonnet rescue curriculum. It has
not yet been used in model training.

The machine-readable preflight evidence is in
[`reports/paisa_release_activation_audit.json`](../../reports/paisa_release_activation_audit.json).
The aggregate acquisition result is in
[`reports/paisa_modern_italian_v1_build_report.json`](../../reports/paisa_modern_italian_v1_build_report.json).

## Source And Terms

- Corpus: PAISÀ Corpus of Italian Web Texts
- Official description: [Corpus Italiano PAISÀ description](https://www.corpusitaliano.it/en/contents/description.html)
- Official persistent release record: [CLARIN ERCC handle 20.500.12124/3](https://hdl.handle.net/20.500.12124/3)
- Release checked: `paisa.raw.utf8.gz`, exposed from the official release route
  with a declared size of 546,911,754 bytes on 2026-07-30.
- Corpus license: CC BY-NC-SA.
- Source-material license families reported by the corpus publisher: CC BY-SA
  and CC BY-NC-SA.
- Reuse role: local, non-commercial research/training only. The derived model
  lineage carries the non-commercial/share-alike restrictions; this is not
  merely a citation requirement.

The publisher describes approximately 380,000 documents from approximately
1,000 websites, totaling approximately 250 million words. It states that every
document carries an XML `text` element with `id` and `url` provenance fields.
The description page does not publish an individual source-license category
alongside every document. The acquisition builder must therefore retain every
document's `id` and `url` as the attribution inventory and must not invent a
more specific per-document license claim.

## Completed Local Acquisition

The builder downloaded `paisa.raw.utf8.gz`, verified the full 546,911,754-byte
payload, and recorded SHA-256
`6c6bee67ad491f858568793ee95831f2b0128eebe42afe4cb566c26823afb4ba`.

- Parsed documents: 387,592
- Retained non-empty, exact-deduplicated documents: 375,388
- Excluded exact duplicates: 12,204
- Retained text: 1,422,136,589 characters and 220,693,165 whitespace-delimited
  words
- Train split: 371,612 documents, 218,451,712 words
- Validation split: 3,776 documents, 2,241,453 words

The split is assigned deterministically from each retained document's SHA-256
content fingerprint, after exact duplicate removal. Consequently, no exact
duplicate can occur in both splits. The builder preserves spelling and
punctuation, normalizing only line endings and transport whitespace. It inserts
`<|endoftext|>` between documents for later language-model training.

The local attribution inventory records every parsed document's `id`, `url`,
status, fingerprint, and split. The temporary archive and interim files were
deleted after successful validation.

## Required Credit

Lyding, V. / Stemle, E. / Borghetti, C. / Brunello, M. / Castagnoli, S. /
Dell'Orletta, F. / Dittmann, H. / Lenci, A. / Pirrelli, V. (2014): "The PAISÀ
Corpus of Italian Web Texts". In *Proceedings of the 9th Web as Corpus Workshop
(WaC-9)*, Association for Computational Linguistics, Gothenburg, Sweden, April
2014, pp. 36-43. Corpus license: CC BY-NC-SA.

## Repository And Model Policy

Do not commit the PAISÀ release, extracted document text, document-level
attribution inventory, BPE token files, or PAISÀ-derived checkpoints to the
public repository. Commit only this attribution record, the official source
links and license notice, deterministic acquisition/build code, configs, and
aggregate non-text reports.

## Current Curriculum Boundary

The PAISÀ-to-historical curriculum is now fixed in
[`docs/paisa_historical_rescue_curriculum.md`](../../docs/paisa_historical_rescue_curriculum.md).
It keeps PAISÀ and historical prose as separate sequential stages and uses only
their training partitions for a fresh 16k BPE tokenizer. The local preparation
report records 8,016,457 PAISÀ and 4,000,540 historical tokenizer-sample
characters, with no validation text included.

The tokenizer, separate encoded streams, GPU calibration, and fixed staged
training plan are complete. The next scheduled action is the locally retained
PAISÀ modern-Italian stage. Its best-validation checkpoint will remain local
under the same CC BY-NC-SA boundary and will initialize the fixed historical
annealing stage.
