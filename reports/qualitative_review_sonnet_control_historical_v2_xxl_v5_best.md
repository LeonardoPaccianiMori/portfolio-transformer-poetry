# Qualitative Generation Review

Generation directory: `outputs/generations/sonnet_control_historical_v2_xxl_v5_best`

## Review Instructions

- Fill in each `TODO` field after reading the generated text.

- Use `low`, `medium`, or `high` consistently within this report.

- Judge the generated text as model output, not as a polished poem.

- Keep weak and failed samples in the report.

## Prompt: amor

- Prompt text: `Amor`

- Seed: `1337`

- Generated file: `outputs/generations/sonnet_control_historical_v2_xxl_v5_best/amor.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: low

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: malformed lexical items and broken syntax prevent a readable poetic statement.

- Notes: The decoder reaches 14 lines, but this is a stopping control rather than evidence of learned sonnet form. Surface diction is occasionally literary, yet lines such as `Camaule potete` and `Porvrian cosí` do not form plausible Italian.

### Generated Text

```text
Amor torna in me sospetto:
quanto per veder tutto questo mio travagliato
Mostro Si copre per parlar nel suo core?
Che io presto veggio, appena può placarmi
Fursi da questa nuova ed alta ombretta.
Non è senza colpa mia potessi hieri desturbarmi
L'immutabil core che allora fino
Camaule potete, allor che cagion venir è non ama,
Che sovente non sia.
Deh muore come n fanno i ramari;
Non piacerà, colle sospira dimandato:
Ma che far gli aspetti ad amordimento![1]
Alle maran la fiamma mia si gni
Porvrian cosí madonna in parlare: prorompmi a mente.
```

## Prompt: donna

- Prompt text: `Donna`

- Seed: `1338`

- Generated file: `outputs/generations/sonnet_control_historical_v2_xxl_v5_best/donna.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: low

- Coherence: low

- Repetition problems: low

- Memorization concern: low

- Strongest failure mode: invented or malformed words repeatedly interrupt otherwise recognizable Italian fragments.

- Notes: This has the lowest automatic repeated-4-gram ratio, so repetition is not the main issue. Words including `cominlo`, `stargare`, `selvaggiace`, and `sofferiglio` make the continuation implausible and it does not sustain a topic.

### Generated Text

```text
Donna inver gli occhi sparsi,
non veggio di te pelo acceso alcun,
e mio viso biasmo i cori:
Onde lui mai viva risposta,
cominlo il figlio al impero non nè invase.
Muo avvicelli, ed a te gride
Mentre temo a lo core stargare;
E davia a me mi volge:
D'amor vè la 'nvidia, per nemico Diedo.
Nostro Signore vario ciel, che spene
Sarte dal divino anello, e selvaggiace
Per star lui rimane, et non mel sofferiglio.
Egli è reo e vizio fa soave
Le mie razze fiamma, côcente
```

## Prompt: io_son

- Prompt text: `Io son`

- Seed: `1339`

- Generated file: `outputs/generations/sonnet_control_historical_v2_xxl_v5_best/io_son.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: low

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: unstable sentence construction and abrupt, unrelated semantic shifts.

- Notes: Some local phrases look Italian, but the continuation repeatedly restarts grammatical relations and ends with the disconnected `Guitterra porse un gran colpo`. The output does not develop the opening speaker or a recognizable argument.

### Generated Text

```text
Io son stanco di novo fido e unica e mia mano.
Né già mai volle già far del mio rischio,
se non che dir ne potrebbe. Poni, ch'io che mortalmente,
Che mel prometto, e quel che scrive,
Ch'io nodessi, signor sì forte, ei trarrà altero,
Quella val di morte che la conduce,
Che poi ch'a lei piace di far tutti i motti,
Che mostra V. Di croce e d'umil bene;
Notte endo dal Ciel che terria
Mercèo tardi pur morisse i più gelosi.
Ed io scorgo ciò ch'io sento, no, e salda rimango
Con ragion che porto il novo erro.
Che mercé l'angelo, e vidi l'opra, e mando,
Guitterra porse un gran colpo, e non accorto.
```

## Prompt: solo_et_pensoso

- Prompt text: `Solo et pensoso`

- Seed: `1340`

- Generated file: `outputs/generations/sonnet_control_historical_v2_xxl_v5_best/solo_et_pensoso.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: low

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: corrupted morphology and incompatible clauses overwhelm the familiar prompt's poetic register.

- Notes: The real Petrarch opening is preserved, but it does not lead to a coherent continuation. Forms such as `creaturaggia`, `rubellate e dissiporte`, and `Io s'ha Tovar` are strong evidence of lexical and syntactic degeneration, not controlled archaic spelling.

### Generated Text

```text
Solo et pensoso ch'i' vo' sfogarti
contendar del vostro antico fele,
sí di quel che di tal pria vive diede.
E m'm, che col viver m'è parta,
lui sia caro a veder quella sol maggior creaturaggia,
che la mia vita vol de la mia vita,
onde l'anima mia discenda agli occhi suoi.
Voi l'uno averete per morire,
ché 'l guiderdon che v'ha rubellate e dissiporte;
ma per lo vostro penser s'è d'acqua,
è om ch'al parer, tanto, per fuggir la vita
che 'l foco m'ha giá conquitto.
— Io s'ha Tovar la colpa mia,
crudo i. tenebre di paura; —
```

## Prompt: line_start

- Prompt text: `Amor, `

- Seed: `1341`

- Generated file: `outputs/generations/sonnet_control_historical_v2_xxl_v5_best/line_start.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: low

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: dialogue-like punctuation and poetic fragments are assembled without stable grammar or semantic continuity.

- Notes: The sample uses several plausible-looking poetic fragments, but the speaker, referents, and syntax keep changing. The 14-line result is enforced by decoding and does not demonstrate a learned octave, sestet, rhyme pattern, or volta.

### Generated Text

```text
Amor, salendo e scalzo,
tornando tu su le preghiere; e agli occhi tuoi
quella dai lacci d'Amor, ch'ogni vestita alzando.
— O oggimai ch'i' so com'io godo,
al mondo di vittorie, siate voi sol cheto. —
- Oimè! voi, so chi vi piace,
tutt'ad una donna di quale e' non è dato
amore. - Non far, Seradiva, che è, egual:
che vi spirto di dolor per la gran gelanza
a chiamar, perché 'l ciel t'è insegnato;
non furor onterv'esser al collare d'Amore,
che non m'ha di compagni ver la donna donzina.
– altro, e sèguidare e saper dir dove
questi a perseguire ciò ch'han voluto,
```
