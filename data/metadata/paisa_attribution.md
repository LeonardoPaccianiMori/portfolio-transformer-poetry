# PAISÀ Corpus Attribution And Local-Use Record

## Status

PAISÀ passed the release, provenance, and local-disk activation preflight on
2026-07-30. It is approved only for the defined non-commercial PAISÀ to
historical-prose to V5-sonnet rescue curriculum. This record does not mean that
PAISÀ text has already been acquired or used in training.

The machine-readable preflight evidence is in
[`reports/paisa_release_activation_audit.json`](../../reports/paisa_release_activation_audit.json).

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

## Scheduled Acquisition Checkpoint

The next PAISÀ checkpoint is a local acquisition-and-inventory builder. It is
complete only when it has: downloaded the official release with a recorded
SHA-256 hash; parsed its document boundaries; retained one local attribution
record per document using the published `id` and `url` fields; produced
aggregate document/word/character statistics and deterministic source splits;
and deleted temporary raw/interim extraction files after its local processed
outputs are validated.
