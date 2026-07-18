# Qualitative Generation Review

Generation directory: `outputs/generations/sonnet_control_quality_swiglu_larger_stable_eval_best`

## Review Scale

- Ratings use `low`, `medium`, or `high` consistently within this report.

- The generated text is assessed as model output, not as a polished poem.

- Weak and failed samples are retained as evidence.

## Prompt: amor

- Prompt text: `Amor`

- Seed: `1337`

- Generated file: `outputs/generations/sonnet_control_quality_swiglu_larger_stable_eval_best/amor.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: medium

- Coherence: low

- Repetition problems: low

- Memorization concern: low

- Strongest failure mode: broken syntax and incompatible semantic relations

- Notes: The opening has some archaic lexical texture, but the continuation does not sustain a coherent proposition. The 14-line shape is decoder enforced, not evidence of learned sonnet structure.

### Generated Text

```text
Amor, lor dice benvolta, che basti,
mostra la sua finta mente dal fin non ha più;
ma s'elli è di lata, patirà più lunga.
E, dove è miglior amaro
che 'l cielo è morte, fa 'l sasso, sì ben cangiato.
Cau' è di paura sì voler ricever,
che non à men 'l discendere tardo,
ch'i' vi teme avere i suoi reali,
quando l'omo a santi permise,
com'a Soddomi transpiri spento.
Per ogni altra vista è detto: "Ei per tal coral palo
Quel ch'è qui egli è vissuto.
Ché possiate prome la vostra avolteri'e voi".
E sia 'l vero e 'l vostro nome cangiando?
```

## Prompt: donna

- Prompt text: `Donna`

- Seed: `1338`

- Generated file: `outputs/generations/sonnet_control_quality_swiglu_larger_stable_eval_best/donna.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: low

- Coherence: low

- Repetition problems: low

- Memorization concern: low

- Strongest failure mode: malformed word forms and disconnected images

- Notes: The prompt is retained, but forms such as `Donnare` and the sequence of images do not form a plausible poetic sentence. The line count is enforced by decoding.

### Generated Text

```text
Donnare, discreto, dispiacere:
Amore è colei, ben dess'eo in fenice,
et tu more.
– Vedil or dica tutto,
e vien la voglia, d'oro omo,
di traverso e di nero, et di terra altissima,
e navigò a terra lassù.
Non me incresca la bontà mia et tuo argomenti,
che non d'acqua, anzi dol mi richiamo
in belle parti de l'opre sí angeliche
belle allegri e 'n be' mie menti
quel che 'n far non tien piacer ognore.
– Là 'nsomma vol cosí, sí la ranza del
tempo languir no bencero in ciascun punto
```

## Prompt: io_son

- Prompt text: `Io son`

- Seed: `1339`

- Generated file: `outputs/generations/sonnet_control_quality_swiglu_larger_stable_eval_best/io_son.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: low

- Coherence: low

- Repetition problems: low

- Memorization concern: low

- Strongest failure mode: ungrammatical constructions and broken word boundaries

- Notes: Several short fragments resemble Italian tokens, but they do not combine into a stable speaker, situation, or sentence. The nearest training overlap remains low risk.

### Generated Text

```text
Io son vertì la metá che, dolente?
Piangi son sí sciolta,
che non sottra mi pigliar di condocto a bastone
che non sgombrasse! ch'a voler tu andar poi.
Alla mia vita non affannai
quando è presso a cinquand'è 'n su' pregi!
Ben le offende e tolle' gli occhi.
- Ahi, figliuola, - di chi t'inganna
quanti ne' granetter avere e quanto tuoFrancia
procaccierai vent'ante, ché se' mort'a la morte,
per quei viva e sì soave.
Tu sai pur la donna - ha ben vinta
e divatevi che disaventuro oste.
E' l'ira guarda et li atti al ben quanto
```

## Prompt: solo_et_pensoso

- Prompt text: `Solo et pensoso`

- Seed: `1340`

- Generated file: `outputs/generations/sonnet_control_quality_swiglu_larger_stable_eval_best/solo_et_pensoso.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: low

- Coherence: low

- Repetition problems: low

- Memorization concern: low

- Strongest failure mode: nonsense lexical inventions and no coherent proposition

- Notes: The prompt resembles a Petrarch opening, but the continuation is not a continuation of the source poem: the memorization check finds only a 16-character longest common span. The 14 lines are decoder enforced.

### Generated Text

```text
Solo et pensoso dei pollastri pai,
ch'è calzolaio e fraldi ed erserae
e piccidente la foglia d'oro; et tu viazzino
di far languiscina e 'l fiore de le lor doglie
hanno 'n me mercede contanti,
et di se stessi viandanti sempre freddo
e co' piedi, et sempre con mie fidanza
con confondo hui: et e poi fermi.
Amor che 'l ciel, non volse aitare
altra dora dal piacer or che lasti,
quello, ch'io videro e 'l pianto morto
ch'essi senz'aluleva alto il cor per il corredo;
scarsi, d'amare lagrime con brache,
da che vien fiume scon o segno!
```

## Prompt: line_start

- Prompt text: `Amor, `

- Seed: `1341`

- Generated file: `outputs/generations/sonnet_control_quality_swiglu_larger_stable_eval_best/line_start.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: low

- Coherence: low

- Repetition problems: low

- Memorization concern: low

- Strongest failure mode: fragmented syntax and incompatible semantic images

- Notes: The text has a few plausible poetic tokens but does not form grammatical lines or a coherent progression. The target line count was supplied by the generation controller.

### Generated Text

```text
Amor, il misero vede sì che la pesculìa
Or morta, ch'a sé non bagna il saluto
limpichetto sono presso la figura, quando solamente
lo campa, immantenente;
e 'l pianto grande odore, che noccio,
in vista trovo nel cor verde aperta
mi sente a torno tornato, ove noi,
Un disonante che gravoso mangiare: pure digiuni,
Vide prova dissolver voi lei, che de
Montettar ci sente vela, stimar guardate.
Ben fa più de le beffe d'Amore
ch'altra umilitate, ch'Amore como voi non giudichi,
né natura d'alcuno arduna.
Ma s'egli è molta gloria leggiadra
```
