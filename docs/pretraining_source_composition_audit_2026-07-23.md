# Pretraining Source Composition Audit: 2026-07-23

## Scope And Decision Boundary

This is a metadata, licensing, and corpus-composition gate. It is not a
page-level extraction or activation decision. No row in
`data/metadata/broader_prose_sources_manifest.csv` changes status here, and no
candidate below may enter a corpus build until it passes the revision-pinned
representative-text audit.

The current `expanded_italian_1200_1800_v1` corpus contains 17,294,440 encoded
BPE tokens. The 75M target therefore needs 57,705,560 further unique BPE
tokens. The detailed calculation is in
[`reports/pretraining_data_gap.md`](../reports/pretraining_data_gap.md).

## Source Licensing Routes

| Route | Reuse position for this project | Required record |
| --- | --- | --- |
| Italian Wikisource | Eligible when the page-level CC BY-SA/GFDL metadata, attribution, and pinned revision are retained. | Work URL, author, source edition, license label, root and subpage revisions, extraction date, and cleaning rule. |
| Liber Liber | Existing approved route under the stated CC BY-NC-SA 4.0 edition layer. | Existing edition/digitization attribution and non-commercial/share-alike restriction. |
| Project Gutenberg | Existing approved route for its stated US public-domain eBooks, subject to source-specific cleanup. | eBook URL/ID, landing-page status, edition note, download date, and boilerplate rule. |
| PAISÀ / Corpus Italiano | Candidate route only. The corpus is a licensed, contemporary-Italian collection rather than historical prose. | Per-document license and attribution data, download terms, document-level provenance, and a separately defined curriculum experiment. |
| Internet Culturale | Candidate route only. Its terms permit non-commercial, attribution-preserving, share-alike reuse unless an item states otherwise. | Item-level owner, source URL, exact applicable terms, attribution wording, and download date. |

The license route alone is insufficient. A work also needs a compatible language
variety, period, genre, and representative-text check.

## Historical Core Candidates

These candidates are compatible with the `expanded_italian_1200_1800` core:
Italian prose, within the period target, and useful for author/register
diversity. Their token estimates are deliberately broad planning estimates from
the published volume/page structure and the current tokenizer fertility. They
are not measured extraction results.

| Candidate group | Role | Evidence and composition value | Estimated BPE contribution | Next audit |
| --- | --- | --- | ---: | --- |
| Pietro Giannone, *Istoria civile del Regno di Napoli*, vols. 1-5 | Core training | Five 1770 volumes are listed by the author category. Volume 1 has a 1770 BEIC source edition and a TXT export; the series adds long southern Italian historical/legal prose from an author not in the current corpus. | 1.5M-3.5M | Treat each volume as a separate source; pin the index and selected pages; inspect front matter, notes, and end boundaries. |
| Paolo Sarpi, *Istoria del Concilio tridentino*, vols. 1-3 | Core training | The work is historical prose from 1619, arranged as eight books. Wikisource exposes the work as three 1935 source-edition volumes, so the editor and edition must be retained in attribution. | 0.8M-2.0M | Verify all three volume roots, distinguish primary prose from the editor's apparatus, and inspect samples across every book. |
| Pietro Verri, *Storia di Milano* | Core training | 1783 historical prose adds eighteenth-century Lombard civic/history register and a new author. | 0.3M-1.0M | Confirm complete hierarchy and separate preface/title matter from the historical text. |
| Cesare Beccaria, *Dei delitti e delle pene* | Core training | A complete 1764 legal/philosophical treatise. It is small, but it diversifies the eighteenth-century register. | 0.05M-0.15M | Extract only Beccaria's primary work; exclude Voltaire commentary and related response texts. |
| Pietro Verri, *Osservazioni sulla tortura*, *Meditazioni sulla economia politica*, and *Discorso sull'indole del piacere e del dolore* | Core training | Complete 1769-1773 legal, economic, and philosophical prose. These are small individually but provide controlled eighteenth-century register diversity. | 0.1M-0.4M combined | Audit as separate works and prevent duplication between annotated and unannotated *Meditazioni* editions. |

All listed Wikisource transcriptions must preserve their displayed CC BY-SA/GFDL
metadata and exact source/edition attribution even where the underlying author
is long out of copyright.

## Explicitly Rejected Or Deferred Candidates

| Candidate | Decision | Reason |
| --- | --- | --- |
| Ludovico Antonio Muratori, *Annali d'Italia* | Excluded at metadata gate | The inspected volume-1 index says that pages remain untranscribed and reports only 25% quality. Its potential size is not a reason to ingest incomplete text. |
| Traiano Boccalini, *Ragguagli di Parnaso* | Excluded at metadata gate | The index explicitly reports pages still to be transcribed. Do not use a partial transcription to fill the scale gap. |
| Vittorio Alfieri, *Vita* | Excluded at metadata gate | The available work is dated 1804, outside the approved 1200-1800 period range. |
| Italian Wikisource *La scienza nuova* | Deferred | The completed source-specific audit found embedded editorial markers and references. The cleaner must not make broad, unverified deletions. |

## Licensed General-Italian Scale Candidate

PAISÀ / Corpus Italiano is not appropriate for the historical core: it consists
of contemporary Italian web text. However, its publisher reports approximately
250M tokens, full raw/annotated download availability, and a document set made
from freely distributable CC BY-SA and CC BY-NC-SA material.

Its role is therefore **a separately defined auxiliary curriculum experiment**:

1. train the same from-scratch model on a provenance-preserving PAISÀ subset;
2. continue pretraining on the expanded historical prose corpus;
3. fine-tune on the V5 sonnet corpus;
4. compare that result with the historical-only parent using the same held-out
   sonnets, prompts, generation settings, memorization checks, and qualitative
   review.

This route could solve the scale problem, but it would test a different research
claim: *modern Italian language pretraining adapted to historical Italian and
sonnets*, rather than a purely vintage Italian model. It is not activated by
this audit. Its document-level license and attribution inventory must be
verified before download because the corpus contains more than one Creative
Commons license family.

## Scale Conclusion

Even the full historical-core candidate batch is expected to add only a few
million BPE tokens. It is worth collecting for better historical coverage, but
it will not close a 57.7M-token gap by itself. The project must therefore choose
one of these bounded paths after the historical candidate audits finish:

1. accept a smaller historical-only corpus and train with a lower data budget;
2. find and license many more historical texts through a larger permitted
   collection; or
3. run the separately documented modern-to-historical PAISÀ curriculum
   experiment alongside the historical-only parent.

## Sources Consulted

- [Giannone volume 1 index](https://it.wikisource.org/wiki/Indice%3AGiannone_-_Istoria_civile_del_regno_di_Napoli%2C_1770%2C_Vol.1.djvu) and [Giannone author book category](https://it.wikisource.org/wiki/Categoria%3ALibri_di_Pietro_Giannone)
- [Sarpi, *Istoria del Concilio tridentino*](https://it.wikisource.org/wiki/Istoria_del_Concilio_tridentino)
- [Beccaria, *Dei delitti e delle pene*](https://it.wikisource.org/wiki/Dei_delitti_e_delle_pene)
- [Pietro Verri author page](https://it.wikisource.org/wiki/Autore%3APietro_Verri), [*Storia di Milano*](https://it.wikisource.org/wiki/Storia_di_Milano/Prefazione), and [*Discorso sull'indole del piacere e del dolore*](https://it.wikisource.org/wiki/Discorso_sull%27indole_del_piacere_e_del_dolore)
- [Muratori volume-1 index](https://it.wikisource.org/wiki/Indice%3AAnnali_d%27Italia%2C_Vol._1.djvu) and [Boccalini volume-1 index](https://it.wikisource.org/wiki/Indice%3ABoccalini_-_Ragguagli_di_Parnaso_I.djvu)
- [PAISÀ / Corpus Italiano](https://www.corpusitaliano.it/en/index.html)
- [Internet Culturale terms of use](https://www.internetculturale.it/it/15/termini-d-uso)
