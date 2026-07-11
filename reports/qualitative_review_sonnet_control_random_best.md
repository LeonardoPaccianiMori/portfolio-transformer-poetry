# Qualitative Generation Review

Generation directory: `outputs/generations/sonnet_control_random_best`

## Review Provenance

- This is an assistant-authored first-pass review based on the saved outputs.

- The 14-line target is decoder enforced and is not evidence of learned sonnet form.

## Prompt: amor

- Prompt text: `Amor`

- Seed: `1337`

- Generated file: `outputs/generations/sonnet_control_random_best/amor.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: low

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: frequent malformed words and fused fragments.

- Notes: The output contains many unstable forms such as `beltato`, `celestiranite`, and `creaangelicapeta`, which make the text substantially less readable than the pretrained counterpart.

### Generated Text

```text
Amor, che per preso, piacenti la cielo:
consenti'al pace canto, la sua beltato,
poi quella et ella orgognasse dice;
che sté morir lagrimar po
che col dice: ploco crudel durare.
Or graviso naturaldato pente
perdo e pian visoggogna
e d'esser il paura i:
ma de' presi fosse celestiranite dolente.
Or legaste rivolgo che spiri:
ch'a vo alta mio ritrovo ane e pergiura:
hond'io creaangelicapeta di e vergosce;
et e degna Dio le mi piacer i
che 'nun gogliose sarne gi', usar natura sonde,
```

## Prompt: donna

- Prompt text: `Donna`

- Seed: `1338`

- Generated file: `outputs/generations/sonnet_control_random_best/donna.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: low

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: word-boundary failure and ungrammatical token fusion.

- Notes: `Donnaa`, `credevalor`, and `ternolqua` illustrate that the random arm has not acquired stable lexical units from the limited sonnet corpus.

### Generated Text

```text
Donnaa mor sa gente gioi,
che mai schi poco nemica bando,
pensi vostra vivea fanno dolcede.
Or li la quando credevalor gramio so;
ché che mecoche fuoche amare me lusorte
ave a ne l'dimostranza? fu dura.
Dischiamomi dicendo: e la verso mi eto to
et pianita portate mor quel al speme
cqu'Amore che bene per àné rito intenda.
Som'farmi 'n Amor tristar 'l ella ra
etdivinti, non vedeste quanta piú no
qualunque ternolqua: al sotto incolpi fia voglioduo?
alcun bene che non sí mostrargombra marre,
sí n'aver male l'amorore?.
```

## Prompt: io_son

- Prompt text: `Io son`

- Seed: `1339`

- Generated file: `outputs/generations/sonnet_control_random_best/io_son.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: low

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: malformed local strings prevent sentence-level interpretation.

- Notes: A few recognizable words remain, but most lines mix fragments into unreadable syntax and do not form a plausible poetic voice.

### Generated Text

```text
Io son anzi miei che mi in cor vil peno,
ove mirabil che occhi la vien mar po so.
E sè te vidi stagione, 'l mi mio sente,
Ve per va con fustei piena.
Po do nel nel ne mi inselente a re che sodella;
E a — dolci confetti — gli fermal . gli che —
accia li m'han dovrla sentepostii. vuole,
Così n'à mai ti man, potei ti gli è pensol di
E infredonimi corte,
esti sente se voi ammira. convidi;
"sia la vetromoveder io a pena servige;
Con vita vostrarosier membra,
nanzi gli le la mi ch'io sua vi.
Tu Per mente,ella compagnia
```

## Prompt: solo_et_pensoso

- Prompt text: `Solo et pensoso`

- Seed: `1340`

- Generated file: `outputs/generations/sonnet_control_random_best/solo_et_pensoso.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: low

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: corruption immediately after the prompt and throughout the continuation.

- Notes: The continuation uses some surface vocabulary from the domain but repeatedly fuses and distorts words, so it is not a readable continuation of the Petrarch-like opening.

### Generated Text

```text
Solo et pensoso sole. et 'nvitro
onde et fin bel dolcelasso!
Seco Amorta cie tal latesta, riviso unidre,ce temterra podesta,
vela tormento. molti lor : piove et né parte:
netto et que mi già fera terra in indexanda.
I'alcun et donna per sta? et ce me è destro,
sempre giorno, anime batto arde verde sua disno
vi mia vecchio mi noi, se l'almatemoro,
Ch'entro notte Amor chiedi al cor parle et giri
e me mondo ognarlodesiderio fatal: che occhi fuggendo.
Né non so seman notte sí stella
lamentar ne una rescora amaro 'l saldo
che ffrente attensuperbatti sembido
perchor, et l'ora anchor occhi, a a Signoro
```

## Prompt: line_start

- Prompt text: `Amor, `

- Seed: `1341`

- Generated file: `outputs/generations/sonnet_control_random_best/line_start.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: low

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: unstable morphology and incomplete semantic relations.

- Notes: The text has punctuation and some Italian-looking connective words, but strings such as `ducercoglia` and `conosggiato` prevent stable local meaning.

### Generated Text

```text
Amor, vo tra. man, a Doí non cui
che bellezze ed sa umanaggio potesse amand'io
olverso meraviglia quel ad né sicuino
tanto 'l meno fere.
quella madonna, vo, mi mio che che conosggiato,
quando non sufoco, mi sia non ducercoglia.
Ché perché che quasi crederia forte beltate:
ed ciò suo veder ho par per parte.
Poi de' cuor lo son o ciò sapesse?
E Oh che con per divenir noia n'abbia
e talor, pensero che mi doglia
lora innanzi ed e l'altri 'n m'accorto,
e fosser vi — allegrezza ch'i' naccia
ché son ne Cato che lui contro
```
