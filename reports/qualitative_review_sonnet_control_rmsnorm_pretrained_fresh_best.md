# Qualitative Generation Review

Generation directory: `outputs/generations/sonnet_control_rmsnorm_pretrained_fresh_best`

## Review Instructions

- Fill in each `TODO` field after reading the generated text.

- Use `low`, `medium`, or `high` consistently within this report.

- Judge the generated text as model output, not as a polished poem.

- Keep weak and failed samples in the report.

## Prompt: amor

- Prompt text: `Amor`

- Seed: `1337`

- Generated file: `outputs/generations/sonnet_control_rmsnorm_pretrained_fresh_best/amor.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: medium

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: locally plausible fragments fail to form a coherent situation or argument.

- Notes: The opening has historical-literary texture, but malformed or opaque forms such as `plo`, `Eransmar`, `aneggia`, and `baame` prevent a readable poetic continuation.

### Generated Text

```text
Amor, che di mille son sentite. - Male, - soggiunse il Magnifico,
la qual parea che ella fosse assai?
Mentre che 'l morir si levasse
che entrasse in plo: per li spettanti gradi
ch'ebber tratto da quei di lui, in gran pena,
e miseramente il ritenesse i lumi Eransmar et lui.
Lí aventurosa il ben ritenesse
che per dar l'anima sua potesse in guisa
aneggia, per la qual di creatura humile
non verrett'a quel dì, ch'una potenza facci
argomento che trovò un golfo sí che 'l duol rubitro.
In tal caso non sa che si respira
leggier che la dolce pianger cena
di baame lagrimose tigre et di penno:
```

## Prompt: donna

- Prompt text: `Donna`

- Seed: `1338`

- Generated file: `outputs/generations/sonnet_control_rmsnorm_pretrained_fresh_best/donna.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: medium

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: recognizable poetic vocabulary is connected by unstable grammatical relations.

- Notes: Several clauses and images are Italian-like, but phrases such as `cupido de chetar pomo` and `gran cerchi fallir non po gelando` do not establish a stable speaker or meaning.

### Generated Text

```text
Donna, non sa che si sia,
quanta pietà et nemica e pensiertina
ne fanno dolce arbitrio.
Vedete, frondi, mio onore,
ché 'l cupido de chetar pomo, et quel che a santar po vivere;
e 'l mercenaio è tale, che l'uomo e 'l mio veneno
dessure; et l'amorose speme infredda et cupida.
Allor ch'è sempre intendente po 'l contrario.
Amor è tanto dismisura adopera,
ch'a la memoria d'un pietoso odore;
qual quaderno ha ben tu fia messo l'odoro,
come la natura stessa nel paradiso,
ch'a sé nel suo male stringe: ch'io
gran cerchi fallir non po gelando,
```

## Prompt: io_son

- Prompt text: `Io son`

- Seed: `1339`

- Generated file: `outputs/generations/sonnet_control_rmsnorm_pretrained_fresh_best/io_son.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: medium

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: dialogue-like syntax and poetic vocabulary do not resolve into a consistent voice or proposition.

- Notes: This sample contains comparatively fluent short stretches, but malformed constructions such as `che se tu suggiassi cotal male la denti` and abrupt speaker changes keep it from being coherent.

### Generated Text

```text
Io son anzi mercennaio una d'un giorno,
credendo tutt'i fosse già 'n gran tempesta.
– Anzi hai gran vanto: i' t'avvien, madonna,
ed io ti provego che far ponga por doglio.
— Or s'i' trovi, core che tu sii sí forte
che io credo ch'ella sia ciascun dì;
che se tu suggiassi cotal male la denti:
fa' che di bene, ch'io t'avrò' femina. —
E tu canteraine di lungi in debole
perdono del mio volto, ch'e' mi convien far venire
a bene e 'n danno e 'ntraggio.
Deo, com'hai ben condizione e peggio,
com'ell'è buono, t'agravando come m'hai.
Ché non potrì in forse partirsi;
```

## Prompt: solo_et_pensoso

- Prompt text: `Solo et pensoso`

- Seed: `1340`

- Generated file: `outputs/generations/sonnet_control_rmsnorm_pretrained_fresh_best/solo_et_pensoso.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: medium

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: prompt-adjacent archaizing language breaks into malformed words and disconnected clauses.

- Notes: The continuation keeps an archaic register but includes forms such as `lassorimanga` and repeated `intelletto`; it does not become a Petrarch-like reflection despite the prompt.

### Generated Text

```text
Solo et pensoso intelletto;
e trovo dì sempre amar amare,
lassorimanga et perdono.
E tal ch'io non trovo intelletto:
poi terra mia, et lascerò via;
s'eo mi scorgo il menar netto;
o non me n'è sviluppare alcun dono,
ch'i' non ardisco star seco in quella città
de li occhi in signoria.
– Or vi voglio addurr'in fine di morte;
incontrati siete dolci e sconsolati,
ch'anchor v'altr'oppresso 'l credo si ritrano,
en vo ne desiderio in vita,
d'ora in anno e in infinito, e mancando
```

## Prompt: line_start

- Prompt text: `Amor, `

- Seed: `1341`

- Generated file: `outputs/generations/sonnet_control_rmsnorm_pretrained_fresh_best/line_start.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: medium

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: locally plausible imagery is joined by malformed syntax and invented-looking word forms.

- Notes: The sample has recognizable words and literary imagery, but `destabili`, `tremarzo`, and shifting referents prevent a coherent line of thought. The low copying score supports treating this as generation rather than a surface reproduction of a training poem.

### Generated Text

```text
Amor, è, tra troppo alto amante?
O cui manto piangi, or mi destabili pur il mio
effetti che di quel ciel sono saliti,
che vi debbo guardare, per obbidire
de la mia mente, onde morta siena.
Sí cara ogn'uom cerco e di cor festi,
ch'è come luce e 'l dì ed e' suo' iniqua et
fidar nozze di quella scorta lo meo disdegno
l'ho sì imperialmente spento
dove lo specchio e 'l tabernacolo:
Sì mi disprego, i' sí dolci tremarzo,
onde m'è 'l mio buon mercato allegrezza.
Allor non è scoppiata la parola
anzi di lui.
```
