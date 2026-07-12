# Qualitative Generation Review

Generation directory: `outputs/generations/sonnet_control_pretrained_warmup_cosine_best`

## Review Provenance

- This is an assistant-authored first-pass review based on the saved outputs.

- The 14-line target is decoder enforced and is not evidence of learned sonnet form.

## Prompt: amor

- Prompt text: `Amor`

- Seed: `1337`

- Generated file: `outputs/generations/sonnet_control_pretrained_warmup_cosine_best/amor.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: medium

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: dialogue-like local phrases without a coherent scene or argument.

- Notes: The sample has readable words and punctuation but unstable grammar, including abrupt speaker changes and fragments such as `Eudana` and `huom creato`.

### Generated Text

```text
Amor, tamente perdoni.
E altri non è bel c'alternana, et dopo arrosto,
e poi pare che ella temenza discenda;
ed esp'io mi piacer più, ché dice:
— Io non credea ch'altro segno,
or fossi vivo? — – Come? —
– Il suo dolce desire dice:
— Sono sí presi gli occhi tuoi?
— Oimè, or messere, or veggio per mio cor,
come che tu miri. — Or voce huom creato,
ché tu non miro el segno.
E Dio t'ave piacer di vedermi l'arme?
E sí che mi duol come spento!
Volto mi maravigliate, la mia pena non mi fia 'l simiglia.
```

## Prompt: donna

- Prompt text: `Donna`

- Seed: `1338`

- Generated file: `outputs/generations/sonnet_control_pretrained_warmup_cosine_best/donna.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: medium

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: syntactically related-looking clauses that do not form stable meaning.

- Notes: This is locally readable but does not build a clear statement. Repeated connective patterns and invented compounds such as `vogliodanza` remain visible.

### Generated Text

```text
Donna, non sa che si sia, quand'era povera,
che bando, pensier, vivere. Ed eo non piú,
ché crede con grave tormento.
Ma se con rancore che 'l pesta, s'abbia a 'nato,
come sterpo e consiglio conviene,
mercè forte e forte, e l'uomo e 'l suo veneno
desina s'eo e vol non può usar cortesia.
Questo à lo gran senno ch'àn le parti,
per cui sie fallace, e sospira.
Son fatti ciechi a briga: o camici,
o convien, ched la dia vogliodanza e serra.
od alcuna parola s'allunga per ragione;
ma mentre che 'ntende ch'io parli sentone,
non parlo a gruppo, se non si parte.
```

## Prompt: io_son

- Prompt text: `Io son`

- Seed: `1339`

- Generated file: `outputs/generations/sonnet_control_pretrained_warmup_cosine_best/io_son.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: medium

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: plausible poetic vocabulary without semantic continuity.

- Notes: The continuation has some fluent fragments, but syntax drifts and imagery does not develop into a comprehensible poetic argument.

### Generated Text

```text
Io son anzi sorte angelico;
et s'alcun m'aiute a mirar le man sue bellezze
s'onori, tanto su, ch'agguagliar le parole
procedano i·lagrimei, et la mostrano a natura:
prendon d'infiat'a neve quel ch'ogne
coregge 'l su' cerchio erba spessa.
Non dico io, chel su' corso de l'erba
amorosa m'han l'agguaglia come fa' la mia
inferno et 'l cor non pote né gli allori
sol dir né l'orme né la foga, se non ch'i' amo,
che ne la mia maggior parte, et apre la croce.
Con quei ch'i'ro infiammar non comporto,
ma mi fa sua vendetta, e 'n tempo e lor da
Dio un altro mondo pari; et l'accende e alluma,
```

## Prompt: solo_et_pensoso

- Prompt text: `Solo et pensoso`

- Seed: `1340`

- Generated file: `outputs/generations/sonnet_control_pretrained_warmup_cosine_best/solo_et_pensoso.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: medium

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: prompt-adjacent archaizing texture without stable grammar.

- Notes: This output is broadly comparable to the constant-rate result: literary particles and vocabulary are present, but the continuation remains semantically fragmented.

### Generated Text

```text
Solo et pensoso sol et 'nvitro
lo spazio et 'l vago destriere,
onde fia forza tal la speme;
non poria celar terra tenere.
Quand'i' vo' che lo sento essempio
sì cambiarei in un giorno a seguir sua
voglia in mare. Sìlgo lo potessi avere,
en disio umile et solido che possa;
ne fan vita; et par che fosse fonte:
mentre che 'nsieme dolci spirti gittando
per l'alto pensero una carne giá mai.
Così la piantaiman vo' che legger vo'!
Che operai sol per notte o le mani
s'alleggi ne l'inferno, come augel saluta
```

## Prompt: line_start

- Prompt text: `Amor, `

- Seed: `1341`

- Generated file: `outputs/generations/sonnet_control_pretrained_warmup_cosine_best/line_start.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: medium

- Coherence: low

- Repetition problems: high

- Memorization concern: low

- Strongest failure mode: unusually long, repetitive prose-like continuation rather than verse.

- Notes: This is the longest output and has the highest repeated-4-gram ratio in the batch. It preserves local Italian-like words but does not maintain a sonnet-like line rhythm or coherent meaning.

### Generated Text

```text
Amor, sentendo che 'l successor non aveniva né seguir co 'l pensier vostro, pur per meraviglia. Però comandai a Dio che vi dessi fedel quella, a ciò che io non sperassi fare; ma avendo tanti foco la vostra virtute in raccoglier per festi, me vi dessi dispiacere.
Ma io preghetto fui dal voler mio divotamente accorto, ed ho officio de lo cor fermo, ché per me stesso son rimasto.
Ciò era per me che mi facesti, quantunque la cagione:
ché cosí mi tenessi alquanto, o da ventura o da lato o da lato
d'umiltà ragionar mi sento in parole,
che com'è l'om mia donna, del pane e 'l roeno,
che non abandona vita, come fa colui a cui
son fatto sí, levomi gli occhi.
Pien' che 'l terzo ardore tpo ed empio:
ché l'un e l'altro anello morrei
che dormissero presso a la sempre allegrezza. Et
gli occhi suoi de me parlari,
nullo accogliente ti riprenderò,
com'io mille volte vivo aver donne,
```
