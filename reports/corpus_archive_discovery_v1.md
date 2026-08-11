# Checkpoint 6C: Final Archive Discovery Pass

Audit date: `2026-08-11`

## Outcome

The frozen 35-query matrix covered 9 independent discovery surfaces. It resolved 16 candidate or surface decisions from 16 pinned evidence records.

The evidence-based stop rule did **not** close directly into checkpoint 7: material source boundaries were found. They remain metadata-only and require checkpoint 6D before cross-archive canonicalization.

## Standard-queue bounded audits

| Candidate | Role | Materiality | Next action |
|---|---|---|---|
| Corpus Antonio Rosmini - Serbati | `auxiliary_capped_ottocento_bridge` | pass: 4,311,182 words necessarily exceed the 1,000,000-character floor | Run a bounded format, primary-text, overlap, and concentration audit; no full exposure without a capped bridge experiment. |
| Digital edition of opera libretti | `core_training_candidate` | pass: 56 underrepresented seventeenth-century works exceed the 10-work scarcity threshold | Run a bounded XML extraction, quality, overlap, embedded-sonnet, and concentration audit. |
| Bellini Digital Correspondence | `auxiliary_capped_ottocento_bridge` | pass: 40 underrepresented epistolary units exceed the 10-work scarcity threshold | Run a bounded TEI extraction, primary-text, overlap, and concentration audit; activate nothing. |
| Oxford Text Archive Italian-language collection | `core_training_candidate` | pass: at least 10 works from an underrepresented early-modern register | Inventory all 43 records, verify item terms/dates/languages, then probe only compatible unique candidates against existing corpora. |

## Conditioned and held discoveries

- **Codice Pelavicino** — `conditioned_auxiliary_experiment_required_inactive`. conditioned pass: more than 100 documents, but not a standard-Italian source Keep outside the standard queue; require a separately approved mixed-language experiment before extraction.

## Registry closure

- New inactive registry boundaries: `ilc_cnr_historical_corpora`, `oxford_text_archive`
- Closed or excluded discoveries: 11
- Existing broader-pool subtotal: 626,379,622 characters
- New corpus characters acquired or activated: 0

## Frozen constraints

- Discovery results and official evidence are metadata only.
- Item-level terms override repository-level metadata licenses.
- Canonical derivatives of BibIt remain excluded even when technically downloadable.
- Mixed Italian/Latin or Venetian HTR resources remain outside the standard-Italian queue.
- No corpus text, V7 split, mixture weight, cache deletion, or GPU work is authorized.
- Next checkpoint: 6D bounded ILC-CNR and Oxford Text Archive audit.
