# Broader Italian Prose Worklist

This worklist tracks candidate prose texts for the broader Italian pretraining
corpus.

The first broader corpus should be prose-only. The sonnet corpus already handles
poetry specialization, so this corpus should teach grammar, syntax, discourse
flow, historical vocabulary, and narrative or expository structure without
verse-specific lineation, rhyme, and meter.

## Inclusion Rules

Include only texts that are:

- Italian or historical Italian/volgare, not English translations;
- prose, not verse;
- usable under recorded source/reuse terms;
- traceable to a source landing page and source edition note;
- separable from introductions, editorial notes, site boilerplate, and download
  headers/footers.

Mixed prose/poetry works are allowed only if the prose sections can be extracted
cleanly and the manifest records that poetry was excluded.

## First Probe: Project Gutenberg

Project Gutenberg is the first practical probe source because each candidate has
a landing page with language, eBook number, source metadata, and public-domain
status.

Required metadata fields:

- source archive: `Project Gutenberg`;
- landing page URL;
- eBook number;
- title;
- author;
- language;
- public-domain status as stated on the landing page;
- release date and last update date when present;
- source edition/original publication notes;
- plain text URL discovered from the landing page;
- cleaning notes, including Gutenberg header/footer removal.

| Priority | Status | Author | Work | URL | Period bucket | Genre | Notes |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | include_probe | Sidrac / anonymous tradition | *Il libro di Sidrach: testo inedito del secolo XIV* | https://www.gutenberg.org/ebooks/44549 | Tier A/B | encyclopedic / philosophical prose | Strong old-prose candidate. Landing page states Italian text and public domain in USA. |
| 2 | include_probe | Matteo Villani | *Cronica di Matteo Villani*, vols. 1-5 | https://www.gutenberg.org/ebooks/author/55483 | Tier A | chronicle prose | All five Italian volumes are verified: eBooks 69898-69902, same 1825-1826 Moutier edition, public domain in the USA. |
| 3 | include_probe | Jacopo Alighieri | *Chiose alla cantica dell'Inferno di Dante Alighieri* | https://www.gutenberg.org/ebooks/30766 | Tier A/B text, later edition | commentary prose | Include only primary chiose/commentary text; remove modern editor introduction where separable. |
| 4 | include_probe | Caterina da Siena | *Libro della divina dottrina: Dialogo della divina provvidenza* | https://www.gutenberg.org/ebooks/26961 | Tier B | mystical / spiritual prose | Strong late-14th-century prose candidate. Keep Biblioteca Italiana provenance note from PG page. |
| 5 | conditional_extract_prose | Dante Alighieri | *La vita nuova* | https://www.gutenberg.org/ebooks/71218 | Tier A | mixed prose and poetry | Do not include as-is. Include only if the builder can extract prose sections and exclude poems. |
| 6 | audit_needed | Giovanni Boccaccio | *Decameron*, Italian edition if present | TBD | Tier A | prose novelle | Search PG catalog during manifest creation. Exclude if only translations are present. |

## Second Probe: Liber Liber

Liber Liber is a targeted source, not the first automated bulk source.

Required metadata fields:

- source archive: `Liber Liber`;
- landing page URL;
- title;
- author;
- underlying-work public-domain rationale;
- Liber Liber license/edition layer;
- source edition notes;
- download format;
- cleaning notes, including wrapper/paratext removal.

| Priority | Status | Author | Work | URL | Period bucket | Genre | Notes |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | include_probe | Anonymous | *Il Novellino* | https://liberliber.it/autori/autori-n/novellino/il-novellino/ | Tier A | short prose narratives | Highest-period-value candidate: late-13th-century material in a later 118-novella edition. |
| 2 | include_probe | Giovanni Villani | *Nuova cronica* | https://liberliber.it/autori/autori-v/giovanni-villani/nuova-cronica/ | Tier A | chronicle prose | Large critical edition; complementary to Matteo Villani. |
| 3 | include_probe | Dino Compagni | *Cronica delle cose occorrenti ne' tempi suoi* | https://liberliber.it/autori/autori-c/dino-compagni/cronica-delle-cose-occorrenti-ne-tempi-suoi/ | Tier A | chronicle prose | Written 1310-1312; strong period match. |
| 4 | include_probe | Giovanni Boccaccio | *Decameron* [Mondadori] | https://liberliber.it/autori/autori-b/giovanni-boccaccio/decameron-mondadori/ | Tier A | prose novelle / frame narrative | Use only this edition to avoid duplicate training text. |
| 5 | include_probe | Giovanni Boccaccio | *Filocolo* | https://liberliber.it/autori/autori-b/giovanni-boccaccio/filocolo/ | Tier A | prose romance | Large early Boccaccio prose candidate. |
| 6 | include_probe | Giovanni Boccaccio | *Elegia di Madonna Fiammetta* | https://liberliber.it/autori/autori-b/giovanni-boccaccio/elegia-di-madonna-fiammetta/ | Tier A | prose fiction | Psychological prose narrative from 1343-1344. |
| 7 | include_probe | Giovanni Boccaccio | *Trattatello in laude di Dante* | https://liberliber.it/autori/autori-b/giovanni-boccaccio/trattatello-in-laude-di-dante/ | Tier A | literary biography / prose | Period-compatible biographical and critical prose. |
| 8 | include_probe | Franco Sacchetti | *Il Trecentonovelle* | https://liberliber.it/autori/autori-s/franco-sacchetti/il-trecentonovelle/ | Tier B | prose novelle | Written mainly in the 1390s; useful spoken and dialectal prose. |
| 9 | conditional_extract_prose | Giovanni Sercambi | *Novelle* | https://liberliber.it/autori/autori-s/giovanni-sercambi/novelle/ | Tier C | mixed prose and poetry | Include only after poetic interludes can be removed reproducibly. |
| 10 | tier_c_scale | Matteo Bandello | *Novelle* | https://liberliber.it/autori/autori-b/matteo-bandello/ | Tier C | prose novelle | Use only in a labeled 1200-1600 expansion corpus. |
| 11 | tier_c_scale | Pietro Bembo | *Prose della volgar lingua* | https://liberliber.it/autori/autori-b/pietro-bembo/ | Tier C | linguistic prose dialogue/treatise | Useful for language-history prose, not core old Italian. |
| 12 | tier_c_scale | Pietro Bembo | *Gli Asolani* | https://liberliber.it/autori/autori-b/pietro-bembo/ | Tier C | prose dialogue | Later literary prose expansion. |

## Explicit Exclusions

| Source | Work | Reason |
| --- | --- | --- |
| Project Gutenberg / Liber Liber | Dante, *Divina Commedia* | Poetry. Exclude from broader prose pretraining. |
| Project Gutenberg / Liber Liber | Ariosto, *Orlando Furioso* | Poetry. Exclude from broader prose pretraining. |
| Project Gutenberg / Liber Liber | Ariosto, *Rinaldo* | Verse/fragments. Exclude from broader prose pretraining. |
| Project Gutenberg | Dante, *The Banquet (Il Convito)* | English translation on PG landing page. Exclude from Italian pretraining. |
| Project Gutenberg | Boccaccio, *The Decameron of Giovanni Boccaccio*, eBook 23700 | English translation by John Payne. Exclude from Italian pretraining. |
| Project Gutenberg | Castiglione, *The Book of the Courtier*, eBook 67799 | English translation. Exclude from Italian pretraining. |
| Project Gutenberg | Vasari, *Lives of the Most Eminent Painters, Sculptors and Architects* | Available Gutenberg volumes are English translations. Exclude from Italian pretraining. |
| Any source | *Rime*, canzonieri, ottave, capitoli, poem collections | Poetry. Keep separate from broader prose corpus. |
| Any source | Drama-only works | Defer unless we create a dialogue/drama dataset version. |

## Creative Commons Policy

The user approved Creative Commons sources on 2026-07-07. The selected Liber
Liber digital editions use `CC BY-NC-SA 4.0`. The manifest records the license,
source edition, and named digitization/revision contributors for attribution.
This source track is non-commercial and share-alike. The builder must emit a
corpus attribution document, and a future commercial model must not silently
reuse these edition files.

## Expanded Gutenberg Probe Result

The expanded live probe completed on 2026-07-07. Its committed report is
`data/metadata/broader_prose_probe_report.json`.

| Measure | Result |
| --- | ---: |
| successfully fetched prose works | 8 |
| cleaned characters | 4,342,736 |
| whitespace-delimited units | 718,512 |
| exact BPE tokens | not counted before broader-tokenizer training |

The existing sonnet BPE tokenizer cannot encode every character in these works,
as established by the first four-work probe. The expanded probe therefore
deliberately omits BPE counts rather than reusing an incompatible vocabulary. A
new BPE tokenizer will be trained after the broader corpus contents are
selected.

This probe is substantially below the planned 10M-25M BPE-token first-corpus
target. The next checkpoint is to implement the Liber Liber source adapter and
measure the approved Creative Commons source set before implementing the
complete processed-text builder.

## Liber Liber Probe Result

The live Creative Commons probe completed on 2026-07-07. Its report and
attribution are committed as:

- `data/metadata/broader_prose_liber_liber_probe_report.json`;
- `data/metadata/broader_prose_attribution.md`.

| Measure | Result |
| --- | ---: |
| successfully fetched prose works | 8 |
| extracted characters after generic wrapper removal | 7,374,920 |
| whitespace-delimited units | 1,278,303 |
| failed works | 0 |

Combined with the Gutenberg probe, the currently measured pool contains
11,717,656 characters and 1,996,815 whitespace-delimited units. These are probe
measurements, not final corpus statistics. Boundary inspection confirmed that
literary content was retained, but some title-page lines and isolated digital
edition notes remain. The processed-corpus builder must apply and test
source-specific paratext removal before tokenizer training.
