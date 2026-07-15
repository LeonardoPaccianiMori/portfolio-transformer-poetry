# Qualitative Generation Review

Generation directory: `outputs/generations/sonnet_control_rope_pretrained_fresh_best`

## Review Instructions

- Fill in each `TODO` field after reading the generated text.

- Use `low`, `medium`, or `high` consistently within this report.

- Judge the generated text as model output, not as a polished poem.

- Keep weak and failed samples in the report.

## Prompt: amor

- Prompt text: `Amor`

- Seed: `1337`

- Generated file: `outputs/generations/sonnet_control_rope_pretrained_fresh_best/amor.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: medium

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: locally plausible poetic language does not form a coherent speaker, addressee, or argument.

- Notes: The sample has historical-poetic texture, but opaque forms such as `anese`, `dolggio`, and `Scieggia` plus unstable grammatical relations prevent a readable continuation.

### Generated Text

```text
Amor, che per mille disamori et mille penne pari, si discosta da me e dove quella eterno temenza di drappio che sotto vena,
com'è la coscienza nostra affocata,
ardendo li spettatori scherzi et conversi.
Fui io piangendo, ed amo e misera, il cor mi disfu:
poi si vide in giro intera allegrezza
quando di me o di donna turba
vi fece onore, o di degna anese,
volgendomi a del primo canto 'l viso,
vergogn'om mi degna Dio honestamente
e, se 'l forun godo mille che mi doni
lo compagnia onde lo tuo cor mi maraviglio,
poi prender per leggier tempo Dio e noi,
e 'l dare di te dolggio e medicina.
Scieggia, donna, ovunquemi leggiadra et costuma,
```

## Prompt: donna

- Prompt text: `Donna`

- Seed: `1338`

- Generated file: `outputs/generations/sonnet_control_rope_pretrained_fresh_best/donna.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: medium

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: recognizable archaizing words are connected by malformed syntax and shifting referents.

- Notes: Phrases such as `sarà insemeraa o derete`, `manincon unguenti`, and `la da alcuna stella` retain the target register only at a local level and do not establish coherent meaning.

### Generated Text

```text
Donna,
non sa che giova nulla, no 'l poco,
che bando, pena vostra vivea o dolcezza;
piú, che 'l valor grave soffrire.
Et è che 'l mio bel viso sarà insemeraa o derete;
torni durator di merzè forte: et non bisogna
manincon unguenti, et piango, et basci quel terrem,
acute Amore et ch'a chi per posa la compia.
Ora non m'aill'a provveder mi sovene,
dico che e' non fa mai com'un umana virtute.
Et non crediate tu che la danza di Siena
la da alcuna stella il cor per sé acconcia?
Bellezza ch'i' nol contento; ché ciò che amor dice
sofferse per gir l'acqua soave.
```

## Prompt: io_son

- Prompt text: `Io son`

- Seed: `1339`

- Generated file: `outputs/generations/sonnet_control_rope_pretrained_fresh_best/io_son.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: medium

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: sentence fragments and imagery shift before a stable proposition can develop.

- Notes: The continuation contains readable vocabulary and occasional syntactic fragments, but `facondiente`, `vol perdonor`, and the abrupt move to `Solo a la croce` leave the voice and meaning inconsistent.

### Generated Text

```text
Io son anzi mercenaria,
d'una et di mia ventura e se no le guance
conoscenza o s'estima, o donne, o altri o quel che procede.
Lasso, si rinnega fra loro;
e sal aversità fa in mal'aria,
e no gallo a reale guardia della
chiave. Questo spesse fiate suol poi
ciascun dì vincer gloria accusare,
humiliare l'amico e 'l facondiente,
perch'allora potesse i cordoni,
o i begli occhi ch'avvenendone infreno,
vol perdonor, che già sete acque,
per l'alpe dov'escie quando move.
Solo a la croce lo fa sentire
```

## Prompt: solo_et_pensoso

- Prompt text: `Solo et pensoso`

- Seed: `1340`

- Generated file: `outputs/generations/sonnet_control_rope_pretrained_fresh_best/solo_et_pensoso.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: medium

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: prompt-adjacent literary language breaks into malformed forms and disconnected images.

- Notes: The sample retains the solitary, reflective register only superficially; forms such as `dre`, `dagit`, `roe`, and `etgo` make it less readable than a coherent Petrarch-like meditation.

### Generated Text

```text
Solo et pensoso intelletto;
presevi non fatto, se non amare,
lasso!, et per vertù et per nostra vera riviera.
dre, poi terra et canto, lascerò in amare,
: or esse benedizioni menziane e vane non
me ne ridesser in baste. I' sommo mento,
per vigor et per diletto è de li occhi giorno,
animosa frigida superba agghiaccio.
Et men grave dritornami, però ch'amore
dagit Amor non si pente: et da indi a pochi et quasi a punto
ad ultimo d'orar fui lungo agghiaccio.
Tu mi semblerai et roe ne l'intorno,
che augel salde che con etgo attraversa
a carra 'l pane et 'l drappo in aggio nostra
```

## Prompt: line_start

- Prompt text: `Amor, `

- Seed: `1341`

- Generated file: `outputs/generations/sonnet_control_rope_pretrained_fresh_best/line_start.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: medium

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: fluent-looking local phrases are interrupted by invented-looking words and unresolved syntax.

- Notes: The line contains some poetic vocabulary but fails to maintain a grammatical thought; `couman baccate`, `direschera`, `slungata`, and `tropice` are conspicuous malformed forms. The low copying score supports treating it as generated rather than reproduced training text.

### Generated Text

```text
Amor, è, che troppo alto amante s'avea cui
di belle piaghe couman baccate;
ma non vi cornarci, per stare a menar
co’ vicini vergognosa il vincerle.
	Ancora mi direschera et ombra mia,
però ch'a non avria duro albergo gittate.
Creder a me come saia, e credo una cocca,
ove fu cortese d'esso, che di certo ci avesse
lo meo coraggio sapesse? E veramente
in sonetto dovete
Giove ch'è sì slungata del mio cor,
mi dislor quand'i' non son tropice,
ché poche fosser'annegendo,
ché quelle che fa 'l vento soave
```
