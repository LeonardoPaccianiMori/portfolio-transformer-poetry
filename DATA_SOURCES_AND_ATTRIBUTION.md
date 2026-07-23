# Data Sources And Attribution

This file is the public index for every dataset source used or considered by
this project. The project is not limited to public-domain text. A suitable
Creative Commons or other explicitly licensed source must be considered on the
same basis as public-domain material when its terms permit this non-commercial
research/training use and all stated conditions can be honored. Public-domain
status is never a criterion for excluding otherwise suitable permitted data.

## Canonical Source Records

The full source-level records live in committed machine-readable manifests:

| Dataset | Canonical source list | Attribution detail |
| --- | --- | --- |
| Classical Italian sonnets | [`data/metadata/poems_manifest.csv`](data/metadata/poems_manifest.csv) | [`data/metadata/attribution_summary.md`](data/metadata/attribution_summary.md) and [`docs/data_attribution.md`](docs/data_attribution.md) |
| Classical Italian sonnets, expanded v2 | [`data/metadata/sonnets_expanded_v2_manifest.csv`](data/metadata/sonnets_expanded_v2_manifest.csv) | [`data/metadata/sonnets_expanded_v2_attribution.md`](data/metadata/sonnets_expanded_v2_attribution.md) and the committed [`Alfieri revision snapshot`](data/metadata/wikisource_snapshots/ws_alfieri_rime_1912.json) |
| Classical Italian sonnets, expanded v3 | [`data/metadata/sonnets_expanded_v3_manifest.csv`](data/metadata/sonnets_expanded_v3_manifest.csv) | [`data/metadata/sonnets_expanded_v3_attribution.md`](data/metadata/sonnets_expanded_v3_attribution.md) and the committed [`Foscolo edition snapshot`](data/metadata/wikisource_snapshots/ws_foscolo_sonetti.json) |
| Classical Italian sonnets, expanded v4 | [`data/metadata/sonnets_expanded_v4_manifest.csv`](data/metadata/sonnets_expanded_v4_manifest.csv) | [`data/metadata/sonnets_expanded_v4_attribution.md`](data/metadata/sonnets_expanded_v4_attribution.md) and the committed [`Varchi revision snapshot`](data/metadata/wikisource_snapshots/ws_varchi_infermita.json) |
| Classical Italian sonnets, expanded v5 | [`data/metadata/sonnets_expanded_v5_manifest.csv`](data/metadata/sonnets_expanded_v5_manifest.csv) | [`data/metadata/sonnets_expanded_v5_attribution.md`](data/metadata/sonnets_expanded_v5_attribution.md), the five committed source snapshots in [`data/metadata/wikisource_snapshots/`](data/metadata/wikisource_snapshots/), and the [`v5 build report`](data/metadata/sonnets_expanded_v5_build_report.json) |
| Prospective standard-Italian sonnet sources | Not active corpus data | [`data/metadata/sonnet_composition_shortlist.csv`](data/metadata/sonnet_composition_shortlist.csv) records source URLs, reuse terms, attribution obligations, composition estimates, and the pre-audit decision for every shortlisted or excluded candidate. |
| Broader Italian prose, including inactive audited candidates | [`data/metadata/broader_prose_sources_manifest.csv`](data/metadata/broader_prose_sources_manifest.csv) | [`data/metadata/broader_prose_attribution.md`](data/metadata/broader_prose_attribution.md) and [`docs/broader_italian_corpus_expansion_audit.md`](docs/broader_italian_corpus_expansion_audit.md) |

Each manifest row records the work, author, archive, source URL, rights or
public-domain status, license notes, edition details, cleaning plan, and
inclusion decision. An `audit_then_include`, `defer`, or `exclude` row is not
training data.

## Active License Policy

- **Public-domain sources:** retain source and edition provenance even where no
  attribution condition is legally required.
- **Creative Commons sources:** retain the exact license identifier and link,
  attribution, creator/editor/contributor credit, source URL, modification
  notice, and any non-commercial or share-alike obligations.
- **Other licensed sources:** include only after the source terms explicitly
  permit this project's non-commercial training use. Record the exact allowed
  use and every required notice, credit, access condition, and downstream model
  restriction before activation.
- **Unclear terms:** do not download for training, process, or include until
  permission or terms clarify the intended use.

The current broader-corpus track includes Liber Liber editions under
CC BY-NC-SA 4.0. That lineage is non-commercial and share-alike. Italian
Wikisource candidates require attribution and retention of their page-level
Creative Commons/GFDL metadata. These restrictions apply to the associated
dataset and model lineage, not merely to a citation in this repository.

## Required Update For Every Activated Source

Before a source becomes active training data, update its manifest row and this
index's linked attribution material with:

1. source, work, author, edition, and stable landing-page URL;
2. exact license or permission evidence and its link;
3. required credit, attribution text, notices, and downstream restrictions;
4. extraction date, cleaning changes, and any retained source revision or dump
   date;
5. whether the source contributes to the vintage or expanded broader-corpus
   version, and whether it appears in train, validation, or test data.

No source may be treated as public domain merely because the underlying author
or work is old. Likewise, permitted licensed data must not be excluded solely
because it is not public domain.
