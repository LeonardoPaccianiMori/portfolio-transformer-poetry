# Qualitative Generation Review

Generation directory: `outputs/generations/sonnet_control_pretrained_fresh_best`

## Review Provenance

- This is an assistant-authored first-pass review based on the saved outputs.

- The 14-line target is decoder enforced and is not evidence of learned sonnet form.

## Prompt: amor

- Prompt text: `Amor`

- Seed: `1337`

- Generated file: `outputs/generations/sonnet_control_pretrained_fresh_best/amor.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: medium

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: locally plausible clauses without a coherent argument.

- Notes: The sample has readable word boundaries and historical-poetic texture, but phrases such as `patrone` and `Eranpre` disrupt grammar and meaning.

### Generated Text

```text
Amor, tamente perdoni.
E altri di lui mi diceva al patrone:
Ivi arrivai
così dove voi siete, sì vi discendra;
ed altre nozze, com' voi vedete,
per vostro benevol vi prestanza.
Come siete voi naturalmente dolente?
Gli occhi vostri vivo di Dio dolermi,
ch'è sì gran borgo d'Eranpre! – Notrani
Sol vostra dolce vista, figura turbata;
e vo alta di vetro pietà e per la vostra fede
lo domandate, che poco vale.
Ma per voi siano Dio vani e maestriali e' vostri,
tanto sarò gitta, che lo doppio loco
```

## Prompt: donna

- Prompt text: `Donna`

- Seed: `1338`

- Generated file: `outputs/generations/sonnet_control_pretrained_fresh_best/donna.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: medium

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: grammatical relations change without a stable subject or scene.

- Notes: Most tokens are readable Italian-like words, but the lines do not develop a speaker, addressee, or internally consistent meaning.

### Generated Text

```text
Donna, non sa che si sia, quand'era povera,
che bando, pensier, vivea oste, ed ancor non si
le riconducerebbe in cielo. Però la mando
la che gradisce lo lupo che 'nanzi al mondo
lo batte perché meni in lingua irra,
e l'uomo e la lingua à tale salute.
E tutte queste provedendo è fredda;
sì 'l ferire per lo freddo caldo
che non è lá dentro del mondo, ma ben
et verde de' lassi, quand'ei noll'osa, et la
chiaccia, et ched la sua ruggine à muta.
Se alcuna volta di tal freda enversa.
Quella ragione, che 'ntende, lo gran caro dono
a mia presenza vergane sodisfaccia.
```

## Prompt: io_son

- Prompt text: `Io son`

- Seed: `1339`

- Generated file: `outputs/generations/sonnet_control_pretrained_fresh_best/io_son.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: medium

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: broken syntax despite recognizable poetic vocabulary.

- Notes: Lines such as `Poscia per li miei dí ogni pensier fa` resemble the target register, but malformed words and unstable syntax prevent a coherent continuation.

### Generated Text

```text
Io son anzi sorte angelico;
et s'alcun m'aiute a mirar le man sue antiche
ch'aperte appareno, faccer chi nol vede.
Però vid'io nel corpo uccisa,
et fra gli altri miei spirti prendo guardia.
Comne cose inseme interabili,
et cosí mia vita mette in opera:
onde tanto può ciascun dolore dura.
Poscia per li miei dí ogni pensier fa,
torna, il bel tempo et cor levando spesso
lor del canto de' divini lume.
Tempo mi estima: Ancor non pensa.
Pur lodando, s'avess'imagini lo spirito
gentile, pensando lo cielo de li benvoli,
```

## Prompt: solo_et_pensoso

- Prompt text: `Solo et pensoso`

- Seed: `1340`

- Generated file: `outputs/generations/sonnet_control_pretrained_fresh_best/solo_et_pensoso.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: medium

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: prompt-adjacent archaizing language without sustained grammar.

- Notes: The prompt is followed by apostrophes and literary vocabulary, but the continuation does not form a readable Petrarch-like meditation. The copying check remains low risk.

### Generated Text

```text
Solo et pensoso sovrano.
Et non è se non amato lasso:
Se quanto fia felice la speme;
se non fusse temperato et potrebbe
la mertate. – Tu: Tu se' infermo,
tu savia, Amore, non me n'inganni.
Così, se non fosse che 'l ferro ha nel punto,
venire tenebroso al solito male;
anima, ch'avrai donato a me quello che danno
che mi spincontra et par dolcissimo,
che mi sdegna chi mi credea imparperare;
et non vil pena ogn'om bene à 'n séguto;
poi m'agrato, e mancando la beltà:
che mille volte la radice aurea salva.
```

## Prompt: line_start

- Prompt text: `Amor, `

- Seed: `1341`

- Generated file: `outputs/generations/sonnet_control_pretrained_fresh_best/line_start.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: medium

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: related-looking clauses that do not resolve semantically.

- Notes: This is among the more fluent samples locally, but its imagery and causal relations remain inconsistent across lines.

### Generated Text

```text
Amor, sentendo che 'l successor non aveniva né seguir co 'l pensier vostro,
pur per meraviglia quel suo bel costume
che vi curava recare.
Ma madonna, che mal agrata era stata morta
quando restava vive,
non avria dubissimo l'amicizia
perché sí come volgarmente fu fattore
adoperando le fu in questo lauro.
Felice aspectar non era l'andare
quanto piú indarno aggio di Giove;
ma pur lungamente il pensero, e tal lor follia
con minore speranza, che l'inganno trasfora,
fu il pensero; ma non è scoperto:
ché 'l cor è qua giú per la via,
```
