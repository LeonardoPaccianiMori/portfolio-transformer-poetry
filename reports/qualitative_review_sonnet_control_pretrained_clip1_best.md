# Qualitative Generation Review

Generation directory: `outputs/generations/sonnet_control_pretrained_clip1_best`

## Review Provenance

- This is an assistant-authored first-pass review based on the saved outputs.

- The 14-line target is decoder enforced and is not evidence of learned sonnet form.

## Prompt: amor

- Prompt text: `Amor`

- Seed: `1337`

- Generated file: `outputs/generations/sonnet_control_pretrained_clip1_best/amor.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: medium

- Coherence: low

- Repetition problems: high

- Memorization concern: low

- Strongest failure mode: repeated words and malformed phrases replace a coherent argument.

- Notes: The archaizing surface is locally plausible, but `perdono` recurs excessively and forms such as `gragrue` and `fryso` undermine readability.

### Generated Text

```text
Amor, tamente perdoni.
E altri di lui mi diceva al patrone:
Ivi arrivai
così dove voi siete, sì vi discendra;
ed altre nozze, com' voi vedete,
entrate, giubbiate, taverne durare.
Or gragrue martirie, perdono,
e piangendo e la stagione d'un'antiqua patena.
E per voi sisma portar non penseria:
e fare - rivolgi - perdono
e divorar tutti i tuoi danno e perdono. -
Re creatura mi disse: - Dàte perdono
e mostratele onne e piacer vostro, - un servidore
e rime e duol come fryso: ché vén contento,
```

## Prompt: donna

- Prompt text: `Donna`

- Seed: `1338`

- Generated file: `outputs/generations/sonnet_control_pretrained_clip1_best/donna.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: medium

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: malformed words and unstable syntax.

- Notes: Several lines have recognizable Italian-like vocabulary, but forms such as `hustin humanitate`, `ladroa`, and `Fequend'ogni` prevent a readable progression.

### Generated Text

```text
Donna, non sa che si sia, quand'era povera,
che di pena tinge a l'imbocca ed ad un ladro
le frondi gravi che rine hustin humanitate.
Che falso modo di quel ladroa ne l'angoscia?
Fequend'ogni mi credetti e vedetti?
Ove di vertú, aita, madre morta,
certo m'avesse accorta da lei a perpetuo.
E solo intendo darmi farmi 'n alcun modo;
ma poi dico: et lassata la scriva.
Ma piú novamente alberga la bella vista
cosí, fia alcuna d'esta mia morte.
Se alcuna 'mpresa m'è da sé acconcia,
da qualunque lato no 'l sent'aggia intero:
poi che sol non m'è soldo, né mille
```

## Prompt: io_son

- Prompt text: `Io son`

- Seed: `1339`

- Generated file: `outputs/generations/sonnet_control_pretrained_clip1_best/io_son.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: medium

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: syntactically incomplete clauses despite recognizable poetic vocabulary.

- Notes: The beginning has historical-poetic texture, but grammar breaks in `faccer chi nol vede` and the final lines do not resolve the speaker's thought.

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
Pur lodando la mente mossa,
ché 'l desio lassú stare non puòro
```

## Prompt: solo_et_pensoso

- Prompt text: `Solo et pensoso`

- Seed: `1340`

- Generated file: `outputs/generations/sonnet_control_pretrained_clip1_best/solo_et_pensoso.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: medium

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: disconnected poetic-looking fragments with malformed word forms.

- Notes: The sample has a plausible lexicon and rhetorical questions, but `encherodella`, shifting subjects, and unresolved phrases prevent a coherent meditation.

### Generated Text

```text
Solo et pensoso sovrano.
Et non è questa sola speranza?
Ma con quanto fructo et con quanto aspro viso
universale, terra oscura et verde et folle.
E: qual essegu infermitate et dispieta
et dolcezza et fera pavento, inde la mente?
De lo spirto et de le tenebre è gita.
Io vivo desta' ne lo stilo ingegno:
tu che 'l cor ch'ora passa con l'alma dolci spirto?
O caro costume, lagrimare?
De la felicissima encherodella villa
che tu mi fulgori in terra, sí secura,
ch'i' non ho sì giá mai, ognora amara et salva.
Questïo volo, di fuor di sè, quanto mi dispersi,
```

## Prompt: line_start

- Prompt text: `Amor, `

- Seed: `1341`

- Generated file: `outputs/generations/sonnet_control_pretrained_clip1_best/line_start.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: medium

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: clauses look related but never form a stable scene or argument.

- Notes: This output is byte-identical to the unclipped baseline for the same prompt and seed. It remains locally fluent but semantically inconsistent across all fourteen lines.

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
