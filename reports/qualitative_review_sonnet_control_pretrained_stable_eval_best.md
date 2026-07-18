# Qualitative Generation Review

Generation directory: `outputs/generations/sonnet_control_pretrained_stable_eval_best`

## Review Scale

- Ratings use `low`, `medium`, or `high` consistently within this report.

- The generated text is assessed as model output, not as a polished poem.

- Weak and failed samples are retained as evidence.

## Prompt: amor

- Prompt text: `Amor`

- Seed: `1337`

- Generated file: `outputs/generations/sonnet_control_pretrained_stable_eval_best/amor.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: low

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: malformed word forms and repetitive syntactic fragments

- Notes: The output has some archaic tokens, but words such as `Ongola`, `graccla`, and `inforundato` prevent a plausible poetic reading. The 14-line structure is decoder enforced.

### Generated Text

```text
Amor, tamente perdé. Ongola:
e lieto amor non allegro si disdice,
sí è il parer al fondo in movimento;
e tale e tal suo dolore in poppa,
e tal ploìar di durare. Or graccla è,
dato il cor bene e piangendo,
ed en me misera tutta veritate,
perciò che 'n questo pregio al gioco di bene aure
il vero rivolgo del ciel ch'aumentasse,
che del tutto siane avvol; la qual cosa crea.
Un altro fierissimo dolor sempre avvene.
Or pur come mi veggio inforundato,
tanto sarò giá, che io compagnia con lo velo
che non sale. - Becchina d'ira, fàtti
```

## Prompt: donna

- Prompt text: `Donna`

- Seed: `1338`

- Generated file: `outputs/generations/sonnet_control_pretrained_stable_eval_best/donna.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: low

- Coherence: low

- Repetition problems: low

- Memorization concern: low

- Strongest failure mode: broken words and incompatible grammatical relations

- Notes: The continuation preserves the prompt but quickly becomes a collection of fragments rather than a coherent address to a woman or to Love.

### Generated Text

```text
Donna, non sa che si sia, quand'era e novo, di pensiertinella.
Ogncera, che tignuole et tien d'un grave
che rinega et qual ran la che amaz me stessa
quel che a que dere extorme dura.
Dismerzar fortezze et vano sono stupende.
Costan, s'apprendere o mormorare vol, rimane;
Amore et quanto per la luce ris'astende
dal color di naturalmente, e sie 'l carico:
et la voglia et la vedere, et il disage.
La bella vista et tu fia alcuna volta
se non la tol bene, o il cor da sé,
con ragione, et pianger ch'io gran contento;
ché meraviglian que' che so' premiato,
né pur manchi né dar pregio a' dannati,
```

## Prompt: io_son

- Prompt text: `Io son`

- Seed: `1339`

- Generated file: `outputs/generations/sonnet_control_pretrained_stable_eval_best/io_son.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: low

- Coherence: low

- Repetition problems: low

- Memorization concern: low

- Strongest failure mode: invented forms and no stable speaker or situation

- Notes: The sample includes recognizably Italian fragments but does not construct a grammatical or meaningful poetic continuation.

### Generated Text

```text
Io son anzi sorte grand'una d'un soave,
ché tu, in verità, non gran tempesta,
a cui su' novo guardi, in conversare:
e sento chi è 'n braccio è caduto:
tanto por le di't'a neve e grue,
coreggi, suso magro, sí cantar vien neve.
Ed èmmi molto vinto: or per li miei tempramo
e pesci confortar, il _conr'or si manta
e spesso lor strale, e s'involgo, e fredde;
per qualche cosa sente danno, da sospirar di dè.
E bene vorre' io a pena servire,
lo che non furo in me più che gli assalti.
All'altre che mai non m'ispesse, non son null'allobunda,
forse piú di me più alti e pieti e di gire?
```

## Prompt: solo_et_pensoso

- Prompt text: `Solo et pensoso`

- Seed: `1340`

- Generated file: `outputs/generations/sonnet_control_pretrained_stable_eval_best/solo_et_pensoso.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: low

- Coherence: low

- Repetition problems: low

- Memorization concern: low

- Strongest failure mode: disconnected poetic fragments

- Notes: The Petrarch-like opening is not copied beyond the low-risk overlap, but the generated continuation does not sustain an intelligible thought.

### Generated Text

```text
Solo et pensoso intelletto;
e trovo fatto, se non amare, lasso!
Se non fia forza talmente adosso,
richiama mia dre, e terra podestata,
aggradirà e giova : or esse benedi.
Or trovo a riprensione
e ad amore, sì ch'uom andarebbe or troppo
fervente, per vigor disioso.
Chè di Dio in tanto animo batto,
ch'i' sua disperato è trarlo,
ch'eo ti rimarrà fatto Ch'i' non schernisse;
or di cosa scenda giù e metta:
ché vdesiderio bene à fermamente paura
e dirò: i' sarò fedel consiglio; ed i'
```

## Prompt: line_start

- Prompt text: `Amor, `

- Seed: `1341`

- Generated file: `outputs/generations/sonnet_control_pretrained_stable_eval_best/line_start.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: low

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: repeated connective fragments and broken syntax

- Notes: The output uses repeated `perciò` and related connective fragments without forming a coherent logical progression. The line count is controlled by the decoder.

### Generated Text

```text
Amor, sentendo che 'l successe,
non fece di belle donne, né di grado far più.
Per che di quel d'Amor venne poco,
tanto 'l meno curasse, il qual era fue mirabile
che talor non era degno di sé quella.
Dunque è quel ch'egli di certi, a cui è provato,
e credo che adorerato fosse fu ciò ch'è;
e per ciò de' cuor lo metteo perciò,
vi ha d'intorno in sé novella di Giotto;
ma d'elli tacendo, dagch'Amor diviene
lo 'ntendiamo al ciel senz'essere.
Oimè, fosser, perdite mi fece darmi podere?
– Perch'i' prego per ragion di vile
sommi spirito che Iddio essere non è tale?
```
