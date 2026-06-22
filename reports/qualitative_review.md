# Qualitative Generation Review

Generation directory: `outputs/generations/transformer_context768_scaled_001`

## Review Instructions

- Use `low`, `medium`, or `high` consistently within this report.

- For positive fields, `high` means the desired property is clearly present.

- For problem fields, `high` means the problem is severe.

- Judge the generated text as model output, not as a polished poem.

- Keep weak and failed samples in the report.

## Prompt: amor

- Prompt text: `Amor`

- Seed: `1337`

- Generated file: `outputs/generations/transformer_context768_scaled_001/amor.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: low

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: malformed words and malformed special-token text.

- Notes: The prompt is not preserved cleanly because `Amor` is immediately fused into `Amorziavta`. The sample has line breaks and punctuation, but most words are not interpretable Italian or plausible archaic Italian. It emits both a corrupted separator-like string and one correct `<|poem_end|>`, so the model has partially learned the separator pattern without controlling it reliably.

### Generated Text

```text
Amorziavta?
Nè digrse aze siú so vesí cel viave si,
Orbive e sú begreco en colene smenttri.


<|poemoem_ensend|>
Sa aio i, t mentra vi'o
Io a dovami ' gntaissteto
ma dasseori mi, gla Dindengr pol dieme
i i tundorse ono glo i doserengio
n gi ch'uer que idomocispua straial funa,
erttar bemilingga che pe fo mio ior scer laduno sure
'altavato diù t'imo lo, la nctono
che stririnque a l ve d'uon solinte,
dagi'Ampesala Lono me ce dinaverto
cona ur'u'usonome er, cchetonoradi io,
ma, eriu rnta lam'an 'a r,
stame viné setenudo iame sco fuchi
che pròco ché itenolmmeti san misosso
vore che pi moror toriste;
er a rluteorav'l ste,
pi Aduola pe l dero, orevilltezi
cher pa de l mome te.


<|poem_end|>

Lai m' è ter ben pasoiunoia per ronzoricombei de
em'ami, ntosme, si cchezaso losalcre,
esse po o se, ro goimei mi o ssgo ireti duese.
Se cionncheta diantellerei di ie be:
Eunala peve derdoi mpi nia ir ceri bi
```

## Prompt: donna

- Prompt text: `Donna`

- Seed: `1338`

- Generated file: `outputs/generations/transformer_context768_scaled_001/donna.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: low

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: incoherent character-level word formation.

- Notes: The prompt is preserved as its own first line, and the output contains punctuation and line breaks. However, most generated words are malformed. The output also contains a corrupted separator fragment before a correct `<|poem_end|>`, which indicates weak control of the end-of-poem token.

### Generated Text

```text
Donna
maca s'ese di aipedo eneru t sude
putrti'el i cr c'ich'è lo aro sssscoco.


<|pxavell vi frgi ch' m_e gla
d|>
Ami gigianior lo na, ch' tton spineto,
quala trorvigo me, egre, qunen no m'la 'è de,
cover e, riontuamolisa da, i
a mbagl sí puda piolire chemautore 'a sta.
Ch'ua di non so rstonzal fe:
E lasos'nor man de i quan l belena e d date,
Il derir sunoglti sí stialla, de,
Chelid'no n be cte sampia del tr agrdo,
coi'lavache 'on e poima diàmi con to;
a vose seccon te e ncoveral r a,
dera talarr sí enn Mlexta nta brlo.


<|poem_end|>

Que calchessa sose amon dilenine scheze,
dr berte ch'è d ettorina lu veza, 'aral quevo,
nsezimerno tor e la i pr, gna de,
noino darara tera, ali gere di cilone
sì, de si'n ch'onere como deme
chertondertoi cheche famolito am' fie,
ener fienognerto n quave, te comiane.
Orute di selte stososongieco ccolle de tocondo,
mare chezio pe male susí fenon si necreli sen
```

## Prompt: io_son

- Prompt text: `Io son`

- Seed: `1339`

- Generated file: `outputs/generations/transformer_context768_scaled_001/io_son.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: low

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: no stable lexical or poem-boundary control.

- Notes: The opening prompt is absorbed into a malformed first word. The sample has many line breaks and some old-Italian-looking character patterns, including accented vowels and apostrophes, but it does not maintain readable syntax. It produces a corrupted separator-like sequence rather than the correct `<|poem_end|>`.

### Generated Text

```text
Io sonovi, meschi, vetrarechia l pi r nta.
Lundo sentaco stefa cimelo, ffichala;
chetarute to Amiarte fri pa;
cheni pamo'encun glme mer na sbe
magi' vedi dare 'è deacoia:
chè che vuspe i oghi e no un 'iselasunze,
E de ssen usra n se noschei nenfi,
ché strr'è ssinde tene, me vo pfapo,
no sovetorati qual di 'u sponzielci;
i cathe sa lamegre squrerettatogni
ta che dino ttuiche, no fualute.


<|po cem_end|>

S'no miude cone miú dona lllile
chen fa lami rsa terilescrsa
eri: alanterna dime Go carrïe;
ovontr pent'ate
pià benzarn ver suro ndene — da,
Opi me i sse vedeonne
de mol s'andesananndemi sci, he cotato meniso
d'a m'massarerenia me. piorrezzzi:
Dene char r, fogià e valal' ladebanetorta.
E son vo moi que sci pe si
peramei l ssí d|pembbegè fo
ccoegre
oe, n soch' der 'ondeta, vi dio pinin e,
vi, mi timami vom'imor, minda, chi,
io t anoch' cie uca bberr mer fadi.
Pia e gio st mo matetomi l chentumo
```

## Prompt: solo_et_pensoso

- Prompt text: `Solo et pensoso`

- Seed: `1340`

- Generated file: `outputs/generations/transformer_context768_scaled_001/solo_et_pensoso.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: medium

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: local style imitation without semantic coherence.

- Notes: This is the strongest sample stylistically because the prompt is a real Petrarch opening and the model continues with line breaks, apostrophes, accents, and a correct `<|poem_end|>`. It still does not form a coherent sonnet: the line count is too high, the syntax is mostly broken, and the continuation is not semantically readable.

### Generated Text

```text
Solo et pensoso,
e mpi con dirtoneda trad'è i do,co
aacheo quentade sa e n eta n ntrst è fi,
prtai, ch' t benamegoinarsontime:
Or ta miti pià ed' quer or queravovuite dilo,
n mich'amio qund'n da detren cage; atose,
ché mare 'ucuo la l'à esatr i 'ar de piredennnte
a, fe ttesseglessi suntrn mi da nn bisso,
vi a fa dolt tet' ver svi ócqua me pen del aro.
Ch'a'al i pa sssí iame di axadiovun fio,
osu m' sesicrisacia l m ' glur ile
d'l quodom'imaci vi eni doghe iamel sial co;
farunorora cirtua na vipe nonder mo
atil be rmintalll fison fo.


<|poem_end|>

e a vasa t fano ia me soma?
Ieso n po moresafiav'a e
pertuatal d oiari vo ine scar,
eragi chion, fr chera a è sanguon frera
donu brdetentarce, l e,
petetuse sui non so che gone sstrze,
en n e'a l tonoinamol'le dosen pa 'l ne:
no chr li pemomemo è perennnocesener s'a
erara cco, derven do n sil echi dala torma
l afegisoné d micusi.
— ma nde for De, pr cholia p
```

## Prompt: line_start

- Prompt text: `Amor, `

- Seed: `1341`

- Generated file: `outputs/generations/transformer_context768_scaled_001/line_start.txt`

### Human Review

- Sonnet-like structure: low

- Language/style plausibility: low

- Coherence: low

- Repetition problems: medium

- Memorization concern: low

- Strongest failure mode: malformed token and word generation.

- Notes: The initial prompt is preserved, and the sample contains line breaks and a correct `<|poem_end|>` later in the output. The output also contains a corrupted separator-like fragment and many malformed words. The sample does not maintain a 14-line structure or a coherent theme.

### Generated Text

```text
Amor, mania pialita?
E seni de'l r, men mioma ore
vi medo ste cessis'ano soc leremono.

<|> d gnvivem_end|>

Io mi — cove - — hedi felivannomorel'è Ami ta diento
e da ddona 'e mecod ntra'elco:
pe pos'ndil ccareme ce qua a spri.
E rll leco m', tral a a lchosta cinggr to,
che digandai aloli, fe a tesoma proveme
della dteminemave folosci se nettre
do pe be viarse o pi bbrciemasire.
"stitovi pateri i prriori scin stora.
Nônd'è nie spil'ai vo i suralaisentorte
cita quacua, er: te etesi,
Mel riteghanse è sero rage pola:
(tuletro pornospela da fese, sa;
e me vor che le mino stt e fo ciania?
n Co pe diera mar 'è ma pel raiodespe,
pr i'or 'iò vie pientumava irnte.


<|poem_end|>


to tttrad' lamoran ti dudessatevama miori 'ellmito
ce la fio, n sia spranerabbe
eganerò r pudre,
In pil etrrur mpimalo, mianormame,
n, ei·chena ege io t lion ango be bien ma:
losgh'Ame per, conzzitoll pel ar lta di,
ce peliar
```
