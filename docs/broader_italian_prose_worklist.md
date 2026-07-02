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
| 2 | include_probe | Matteo Villani | *Cronica di Matteo Villani*, vol. 1 | https://www.gutenberg.org/ebooks/69898 | Tier A | chronicle prose | Strong historical prose. Find and add related volumes from the author page during manifest creation. |
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
| 1 | audit_then_include | Giovanni Boccaccio | *Decameron* | https://liberliber.it/autori/autori-b/giovanni-boccaccio/ | Tier A | prose novelle / frame narrative | Highest-value Liber Liber prose candidate. Record whether using Mondadori or UTET edition. |
| 2 | audit_then_include | Giovanni Boccaccio | *Trattatello in laude di Dante* | https://liberliber.it/autori/autori-b/giovanni-boccaccio/ | Tier A/B | literary biography / prose | Good period-compatible prose. |
| 3 | audit_then_include | Giovanni Boccaccio | *Elegia di Madonna Fiammetta* | https://liberliber.it/autori/autori-b/giovanni-boccaccio/ | Tier A | prose fiction | Strong prose fiction candidate. |
| 4 | audit_then_include | Giovanni Boccaccio | *Filocolo* | https://liberliber.it/autori/autori-b/giovanni-boccaccio/ | Tier A | prose romance | Large prose candidate; useful after smaller probe works. |
| 5 | tier_c_scale | Matteo Bandello | *Novelle* | https://liberliber.it/autori/autori-b/matteo-bandello/ | Tier C | prose novelle | Use only in a labeled 1200-1600 expansion corpus. |
| 6 | tier_c_scale | Pietro Bembo | *Prose della volgar lingua* | https://liberliber.it/autori/autori-b/pietro-bembo/ | Tier C | linguistic prose dialogue/treatise | Useful for language-history prose, not core old Italian. |
| 7 | tier_c_scale | Pietro Bembo | *Gli Asolani* | https://liberliber.it/autori/autori-b/pietro-bembo/ | Tier C | prose dialogue | Later literary prose expansion. |

## Explicit Exclusions

| Source | Work | Reason |
| --- | --- | --- |
| Project Gutenberg / Liber Liber | Dante, *Divina Commedia* | Poetry. Exclude from broader prose pretraining. |
| Project Gutenberg / Liber Liber | Ariosto, *Orlando Furioso* | Poetry. Exclude from broader prose pretraining. |
| Project Gutenberg / Liber Liber | Ariosto, *Rinaldo* | Verse/fragments. Exclude from broader prose pretraining. |
| Project Gutenberg | Dante, *The Banquet (Il Convito)* | English translation on PG landing page. Exclude from Italian pretraining. |
| Any source | *Rime*, canzonieri, ottave, capitoli, poem collections | Poetry. Keep separate from broader prose corpus. |
| Any source | Drama-only works | Defer unless we create a dialogue/drama dataset version. |

## First Probe Result

The first live probe completed on 2026-07-02. Its committed report is
`data/metadata/broader_prose_probe_report.json`.

| Measure | Result |
| --- | ---: |
| successfully fetched prose works | 4 |
| cleaned characters | 2,502,986 |
| whitespace-delimited units | 417,955 |
| exact BPE tokens | unavailable with the sonnet tokenizer |

The existing sonnet BPE tokenizer cannot encode every character in these
works. This is expected because its base vocabulary was learned from the much
smaller sonnet corpus. The report records each incompatibility instead of
misreporting a partial token total. A new BPE tokenizer will be trained after
the broader corpus contents are selected.

This probe is substantially below the planned 10M-25M BPE-token first-corpus
target. The next source checkpoint is therefore to add more verified prose
volumes, starting with related Project Gutenberg volumes and the audited Liber
Liber Boccaccio works, before implementing the complete processed-text builder.
