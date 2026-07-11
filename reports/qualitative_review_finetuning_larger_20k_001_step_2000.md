# Qualitative Generation Review

Generation directory: `outputs/generations/finetuning_larger_20k_001_step_2000`

## Review Provenance

- This is an assistant-authored first-pass review based on the saved outputs.

- It can be revised by the project owner after independent reading.

- For positive fields, `high` means the desired property is clearly present.

- For problem fields, `high` means the problem is severe.

- The 14-line target is imposed by decoding control, so it is not evidence that
  the model learned sonnet rhyme, octave/sestet organization, or argument structure.

## Prompt: amor

- Prompt text: `Amor`

- Seed: `1337`

- Generated file: `outputs/generations/finetuning_larger_20k_001_step_2000/amor.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: medium

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: grammatical and semantic discontinuity across lines.

- Notes: The output has readable individual words and an elevated, historical-poetic register. However, clauses do not develop a stable speaker, syntax, or argument. The 14 lines come from the decoding target, not from learned sonnet form.

### Generated Text

```text
Amor, che per amareggiare non senti? -
Secondo altero il sanguinoso desio,
così poi che gli occhi ubbidienti dice;
Secondo il suo dolore in cui sentir è lontano:
"Altri! Ahi, gravoso! Forse altro mal perdo.
Poscia che tu miri, e d'esser più patenti,
tutto che saranno sopresi i celestiali ferri.
I' sono o udir che spiriti o erbe
fur dorate in terra, sì ne l'ultimo del primo
stato del mondo fieristi, e sì mi degnate
che mai non mi ingannin un godo o canne:
e donde io compagnia vivo lo veggio,
che non sabato la mia penna leggiera.
E quel che pietoso dare di te languire
```

## Prompt: donna

- Prompt text: `Donna`

- Seed: `1338`

- Generated file: `outputs/generations/finetuning_larger_20k_001_step_2000/donna.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: medium

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: dialogue-like punctuation without coherent dialogue or syntax.

- Notes: The sample is lexically more Italian-like than the earlier BPE baseline and uses punctuation confidently. It repeatedly shifts speaker and grammatical relation, and `Vogliamo sonettozioso?` shows surface imitation rather than controlled poetic language.

### Generated Text

```text
Donna, non sa che si sia, quand'era povera.
— Credo, s'io vi doma, sí, per rispetto!
— credete ch'ell'ha tenuto; — che faranc'hai?
– O luciquel, –a sproni?
– Alla torc'ogni cosa amata.
— Chi fa l'uomo e cuito e come fello? — Elli sa. — accordar, —
i' avrei provato? — — Io morto. —
Con rose — conto giglio.
E appresso di lei non vedea nipote
piú d'un ovo terren qua: ché cosí la vede e 'l vede.
Oimè, mala voglia! — dice — sí ch'ella è sua corte.
— Oimè, Morat  a mia colpa, g'è doma e strazio?
Vogliamo sonettozioso? — Ahi, per mia cortesia.
Ma ch'ella cerca tanto in cura quant'io, Ettor, ten',
```

## Prompt: io_son

- Prompt text: `Io son`

- Seed: `1339`

- Generated file: `outputs/generations/finetuning_larger_20k_001_step_2000/io_son.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: medium

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: plausible local phrases that fail to compose a semantic whole.

- Notes: This continuation has recognizable poetic constructions, including `Maladetta` and `dolcezza`, but grammatical roles drift and individual lines do not support a coherent claim. The fixed line count should not be read as learned sonnet structure.

### Generated Text

```text
Io son anzi sorte angelico!
Maladetta, io posso provare quanto fosse
già mai conoscenza o criatura o scrittore.
Sì che altri me lagrimosi mi par dolcere.
E per amarla mostrarla, son pregato,
poi che t'amor la lascia in braccio.
E cosí si sconforta — dolcezza —
– Amor, maestr'ogni cosa altero.
E fatt'ha la denti posti sopr'un lato:
ch'io t'amor dritto 'l portan del canto
vostro coro intorno intorno, et lietamente,
e 'l segno che mi conduce a ridi;
quando move il bel foco, e la terra m'adduce
rompendo 'l petto dentro 'l mio cor traendo sua
```

## Prompt: solo_et_pensoso

- Prompt text: `Solo et pensoso`

- Seed: `1340`

- Generated file: `outputs/generations/finetuning_larger_20k_001_step_2000/solo_et_pensoso.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: medium

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: prompt-adjacent style imitation without a grammatical continuation.

- Notes: The Petrarch-like prompt is followed by archaizing particles, apostrophes, and literary nouns, but the continuation rapidly loses syntax. The memorization check is low risk; the output does not reproduce a long span from the prompted Petrarch sonnet.

### Generated Text

```text
Solo et pensoso sol et 'nvitro
lo spazio et 'l vago legno et alto
spirto tal ch'io non credea non essere
dre, et terra tenere d'altrui lagrime.
Deh, : or esse bene si disarmilla,
et que' che stanno bene, in modo è sconsolato,
ch'i' non so che trarmi il ciel al meo;
e talani ne vede arde la sua disarmata;
e son del tutto condannato, dolcissimo,
che non sdegna chi mi pente:
però s'inganni per tutto 'l grande abito
ad allegrezza e di ben fare, et amor sprego
senza ragion tuttavia, che mille volte
d'altrui si aumentano per satisfare.
```

## Prompt: line_start

- Prompt text: `Amor, `

- Seed: `1341`

- Generated file: `outputs/generations/finetuning_larger_20k_001_step_2000/line_start.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: medium

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: syntactically related-looking clauses with inconsistent meanings.

- Notes: The output preserves the punctuation-bearing prompt and maintains Italian-like word boundaries. It has no sustained semantic development, and several lines combine plausible tokens into implausible relations, such as `quel nodo non si coce`.

### Generated Text

```text
Amor, è, tra troppo termine?
E 'l novo coral mi fa duro di farla purgare:
ché per meraviglia quel nodo non si coce:
ché vergogna quella pietà che fai,
e che ta' non è degno di sé quella.
Dunque è cotoi, onne, festi, versi, corami.
E 'l dì ed Amor mi fa venire in Cicilia,
per uscirne a sí son presi l'andare
quanto piú ti puoi indietro,
e sonne sì venute a tale,
ch'ella mi dispresta ben sí la cagione.
Ed ella gentile, po tener lei risponder,
ché de la lingua m'è aperta com d'umiltà.
L'ambasciator m'è stato insegnato,
```
