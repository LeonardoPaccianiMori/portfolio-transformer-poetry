# Qualitative Generation Review

Generation directory: `outputs/generations/sonnet_corpus_scaling_v5_best`

## Completed Review Protocol

- Reviewer: assistant-generated assessment from direct text inspection; this is
  not independent human evaluation.
- Ratings use `low`, `medium`, or `high` consistently.
- The 14-line target is decoder enforced, so it is not evidence of learned
  sonnet structure on its own.
- Every sample is retained, including weak and failed output.

## Prompt: amor

- Prompt text: `Amor`

- Seed: `1337`

- Generated file: `outputs/generations/sonnet_corpus_scaling_v5_best/amor.txt`

### Qualitative Assessment

- Sonnet-like structure: low

- Language/style plausibility: low

- Coherence: low

- Repetition problems: low

- Memorization concern: low

- Strongest failure mode: locally literary words are combined into ungrammatical
  and semantically incompatible clauses.

- Notes: This sample is less visibly repetitive than the V4 counterpart, but
  it still lacks a stable speaker, syntax, and poetic argument.

### Generated Text

```text
Amor, lor sciolse
Amor bastardi a tentar bìa l'irrore.
Pietoso Allor così dicendo: - Voi vedeste
Figlio - che che il Podestà miglior ama sentía;
Al Cielo non è di ciò sì bagnato,
Poichè prendei coi migliori nodi cogli,
e di più in altro tornerete discende e vitù
Amore; ma come i suoi primi accenti
Gli guiderdoni pecca, presso al vero il poeta,
E testimoneremo calamito, ch'i' muoia,
Sicchè dentro à presa al Gismondo, si guardi
Le coscienze a l'alte piaghe e meffi marti;
Anima scampò, e non giova rintenere.
Dico 'l mio onore e la 'mpresi.
```

## Prompt: donna

- Prompt text: `Donna`

- Seed: `1338`

- Generated file: `outputs/generations/sonnet_corpus_scaling_v5_best/donna.txt`

### Qualitative Assessment

- Sonnet-like structure: low

- Language/style plausibility: low

- Coherence: low

- Repetition problems: low

- Memorization concern: low

- Strongest failure mode: phrase-level plausibility without grammatical or
  semantic continuity between lines.

- Notes: The output has varied vocabulary and a few classical-looking turns,
  but it does not describe a coherent situation or relation.

### Generated Text

```text
Donna del Crucifisso Amore dal come,
Mi tolse la bilanzia a quest'arte;
Con quella donna che d'onorar conforto,
E sospirando lo 'n guerra fu pace;
Tal che lo viver del pulcino
Chi altieramente amara, e serena ardente,
Che non sia né borsa di vele d'oro,
Deh ond'io non son possente.
Dentro dal bel viso belle e leggiadre carte,
Delle santissima spine e d'argento
Di donne mortal limpidi queste farine,
Ogni piacer che tra noi amica
Ignoria mia, o me nulla onesta.
Parole del mio Crcero in ciascun lato
```

## Prompt: io_son

- Prompt text: `Io son`

- Seed: `1339`

- Generated file: `outputs/generations/sonnet_corpus_scaling_v5_best/io_son.txt`

### Qualitative Assessment

- Sonnet-like structure: low

- Language/style plausibility: low

- Coherence: low

- Repetition problems: low

- Memorization concern: low

- Strongest failure mode: a sequence of grammatical fragments that never
  resolves into a coherent speaker or statement.

- Notes: Several lines have smoother local syntax than their V4 counterpart,
  but the full text remains semantically disconnected.

### Generated Text

```text
Io son ver oggi di me tocca e non nodri il colore.
Ma, quanto s'intenda lor l'opera ardente,
di condotta, d'incontro via pietra,
prenda gran gloria e ch'a volo andar poi
dia mia vita e acqua e ferro svegli.
Benci, fessa che 'ngegno l'inutile!
Ben le offende, otto volte scorri
tale percossa e piena al seno,
ché di cibo piangiente si etternal scorta,
dicendo Francia un dí vent'in su l'albergo suo,
giá se 'l ciel viva e 'l veneno.
Ed or pur la donna mia ha confortata
perché l'acque prive e disagi,
e quanto piú ti va nudricando li occhi;
```

## Prompt: solo_et_pensoso

- Prompt text: `Solo et pensoso`

- Seed: `1340`

- Generated file: `outputs/generations/sonnet_corpus_scaling_v5_best/solo_et_pensoso.txt`

### Qualitative Assessment

- Sonnet-like structure: low

- Language/style plausibility: low

- Coherence: low

- Repetition problems: low

- Memorization concern: low

- Strongest failure mode: unstable word forms and abrupt semantic jumps after
  the familiar opening.

- Notes: The sample has more compact lines and a few plausible lyrical
  fragments, but no sustained sonnet argument, rhyme plan, or metre.

### Generated Text

```text
Solo et pensoso molto,
l'ha parlato.
DI la calorá mia rise in bon domi.
Seguita ad alcuni, la imagine teme,
tempo la pregozzerá et la medicina.
Delle donne di lei riluce,
Col Sol, perché mi parea d'intorno.
Talor di dolce man mi veggio albergo,
et e' m'ebbe vinti sí di gioia et dipartita.
O scaldato volo, io già non mi fui sí sempre
che non volse ai lacci miei,
ch'i' 'l credetti lascio mai. Ahi non m'è tolto
in dubbio essi sentero et levaro
da' miseri miei dubbi, s'elli scarra,
```

## Prompt: line_start

- Prompt text: `Amor, `

- Seed: `1341`

- Generated file: `outputs/generations/sonnet_corpus_scaling_v5_best/line_start.txt`

### Qualitative Assessment

- Sonnet-like structure: low

- Language/style plausibility: low

- Coherence: low

- Repetition problems: low

- Memorization concern: low

- Strongest failure mode: malformed neologism-like words and disconnected
  assertions despite surface rhetorical punctuation.

- Notes: The lower automatic repetition score does not translate into a
  coherent poem; line count remains decoder controlled rather than learned.

### Generated Text

```text
Amor, qual Scimenta,
Etrui del parculelao suo mortal caccia?
Non so se di men gli è sparito,
Pur trova colui che solamente
Soave è queste sue leggi.
Veggio Mercha delle Muse o, in cui troppi sostegni
S'intendea infin l'alta Dia
Da chiamar che disface in lor libertà?
Fortemberosa cominciar perché l'alma le falce
Tal che ragna un quei divini autori,
Mentre sente l'ombre, stimar suole
Dell'incende più sensibil Morte amara.
L'altra umilitate, ch'Amore coglie, e non mai;
Ma par, ch'i' dica al latino lacrime e priva,
```
