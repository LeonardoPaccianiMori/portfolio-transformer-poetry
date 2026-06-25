# Broader Pretraining Manifest Schema

The broader pretraining manifest is a machine-readable table with one row per
candidate prose source.

Recommended file:

```text
data/metadata/broader_prose_sources_manifest.csv
```

The sonnet manifest has one row per poem. This manifest has one row per source
work, because the broader pretraining corpus will contain longer prose works
that may later be split into chapters, passages, and training chunks.

## Core Rule

One manifest row equals one source work or one source volume.

Do not create one row per sentence, paragraph, token chunk, or PyTorch batch.
Those are later dataset transformations. The manifest records source-level
provenance and policy decisions.

## Field Definitions

| Field | Type | Required | Description |
| --- | --- | ---: | --- |
| `source_id` | string | yes | Stable internal ID, such as `pg_sidrac_44549`. |
| `title` | string | yes | Source title from the landing page or source audit. |
| `author` | string | yes | Normalized author or anonymous/tradition label. |
| `source_archive` | string | yes | Archive name, such as `Project Gutenberg` or `Liber Liber`. |
| `source_collection` | string | no | Archive subcollection, author page, or catalog grouping. |
| `landing_page_url` | string | yes | Human-facing page used for attribution and metadata. |
| `download_url` | string | no | Direct text/ePub download URL, if already known. |
| `ebook_id` | string | source-specific | Project Gutenberg eBook number. Required for Project Gutenberg rows. |
| `language` | string | yes | Source language. Should be Italian or historical Italian/volgare for included rows. |
| `period_bucket` | string | yes | Project period tier. See allowed values below. |
| `approx_date` | string | no | Human-readable date or century. |
| `genre` | string | yes | Human-readable genre, such as `chronicle prose`. |
| `text_kind` | string | yes | `prose`, `mixed`, `poetry`, or `drama`. |
| `inclusion_status` | string | yes | Whether this row enters the first probe, is conditional, deferred, or excluded. |
| `public_domain_status` | string | yes | Source-specific rights/status note. |
| `license_notes` | string | yes | Reuse/license notes copied or summarized from the source audit. |
| `edition_notes` | string | no | Edition, editor, transcription, scan, or archive-layer notes. |
| `source_release_date` | string | no | Release date from source metadata if available. |
| `source_last_updated` | string | no | Last source update date if available. |
| `expected_clean_text_path` | string | no | Planned processed text path if the row is later built. |
| `token_count_report_path` | string | no | Path to a later token-count report for this source. |
| `split` | string | no | `train`, `validation`, `test`, or `excluded`. Leave empty before splitting. |
| `boilerplate_strategy` | string | source-specific | How source headers, footers, wrappers, and site text should be removed. Required for Project Gutenberg rows. |
| `mixed_text_strategy` | string | conditional | How prose is separated from poetry in mixed works. Required for `conditional_extract_prose`. |
| `cleaning_notes` | string | no | Human-readable cleaning notes. |
| `audit_notes` | string | no | Source, period, genre, access, or extraction uncertainty notes. |

## Allowed Values

### `source_archive`

Use one of:

- `Project Gutenberg`
- `Liber Liber`
- `Italian Wikisource`
- `Biblioteca Italiana`

### `period_bucket`

Use one of:

- `tier_a_pre_1375`
- `tier_a_b_borderline`
- `tier_b_1376_1400`
- `tier_c_1401_1600`
- `tier_d_post_1600`
- `unknown`

### `text_kind`

Use one of:

- `prose`
- `mixed`
- `poetry`
- `drama`

The first broader pretraining corpus is prose-only. Poetry and drama rows must
be excluded or deferred. Mixed rows are allowed only when `inclusion_status` is
`conditional_extract_prose` and `mixed_text_strategy` explains how poetry is
removed.

### `inclusion_status`

Use one of:

- `include_probe`: active first-pass source-probe candidate.
- `conditional_extract_prose`: usable only if prose can be separated cleanly.
- `audit_then_include`: likely useful, but blocked until source-specific
  metadata or license details are recorded.
- `tier_c_scale`: later 1401-1600 scale-expansion candidate.
- `defer`: intentionally postponed.
- `exclude`: explicitly not used.

### `split`

Use one of:

- empty string before split assignment;
- `train`;
- `validation`;
- `test`;
- `excluded`.

Splits should be assigned after cleaned text sizes are known. A source work can
be split internally later, but that requires care: a long prose work split into
multiple passages must avoid near-duplicate leakage between train, validation,
and test.

## Source-Specific Rules

### Project Gutenberg

Project Gutenberg rows must record:

- landing page URL;
- eBook ID;
- public-domain status from the landing page;
- boilerplate strategy.

The builder must strip Gutenberg header/footer boilerplate before training.
The manifest records the source-level metadata; the cleaned text will be a
derived artifact.

### Liber Liber

Liber Liber rows must record two separate ideas:

- the underlying old work may be out of copyright;
- the Liber Liber edition/layout/wrapper layer may have its own license terms.

This matters because the model should train on primary text, not silently absorb
archive wrapper text or modern edition material. From a project-governance view,
the manifest makes this distinction explicit and reviewable.

## Data Science Role

This manifest is dataset governance.

For modeling, "more text" is not automatically better. We need to know whether a
row is old Italian prose, later Renaissance prose, poetry, drama, a translation,
or modern commentary. These labels let us build named dataset versions and later
interpret model behavior.

Example:

```text
old_italian_core_pre1375
historical_italian_1200_1600
historical_italian_plus_sonnets
```

If a model improves after adding Tier C prose, we should be able to say that
clearly instead of pretending everything came from the same period.

## PyTorch Role

PyTorch does not train on the manifest directly.

The pipeline will be:

```text
manifest row
  -> source download or local source file
  -> cleaned prose text
  -> BPE tokenizer.encode(text)
  -> token IDs
  -> training chunks shaped like (batch, context_length)
```

For causal language modeling, each training chunk becomes:

```text
x = token_ids[0:context_length]
y = token_ids[1:context_length + 1]
```

The model receives `x` and learns to predict `y`. The manifest is the layer that
lets us understand where those token IDs came from.

## Acceptance Criteria

The broader manifest schema is ready for builder implementation when:

- Project Gutenberg candidates can be represented with eBook ID and
  public-domain metadata;
- Liber Liber candidates can represent the underlying-work status separately
  from the archive license layer;
- poetry and drama can be explicitly excluded;
- mixed works can require a prose-extraction strategy;
- source-level fields can later connect to cleaned-text paths and token-count
  reports;
- the schema explains the path from source metadata to PyTorch examples.
