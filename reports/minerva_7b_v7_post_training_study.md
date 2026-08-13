# Minerva 7B V7 Post-Training Study

## Scope

This study follows one pinned Minerva 7B Instruct model through a three-stage,
full-weight BF16 curriculum:

1. historical and literary Italian adaptation;
2. historical non-sonnet poetry adaptation;
3. V7 sonnet specialization.

It retains the midpoint and validation-selected endpoint of every stage and
compares model weights, selected token embeddings, hidden representations,
losses, and generated behavior. A later bounded rank-8 LoRA-DPO branch targets
two observed output failures: meta-text and incomplete terminal syntax.

The evidence is descriptive, except for the matched DPO adapter comparison.
Weight and activation differences locate change but do not by themselves prove
that a component caused a behavior.

## Training Dynamics

All three stages passed their predeclared adaptation and preservation gates.

| Stage | Selected update | Runtime | Selected target loss | Main observed change |
| --- | ---: | ---: | ---: | --- |
| Historical/general | 2,065 | 15,495 s | historical-general `2.8466` | Broadest and largest adaptation |
| Non-sonnet poetry | 760 | 7,547 s | poetry `2.8475` | Additional poetry-domain improvement |
| Sonnets | 120 of 135 | 3,115 s | sonnet `3.1103` | Small, targeted late specialization |

Stage 1 reduced historical-general loss by `0.0762`, historical-poetry loss by
`0.1702`, and sonnet loss by `0.2385` between its first validation and selected
endpoint. Stage 2 further reduced poetry loss by `0.0780` and sonnet loss by
`0.0908`. Stage 3 reduced sonnet loss by `0.0179` while historical, modern, and
instruction losses changed by less than `0.0051` in absolute value. Update 120
was retained because the terminal update 135 was effectively tied on sonnet
loss but worse on instruction loss.

## Where The Weights Changed

The global relative parameter displacement from the untouched parent to the
selected Stage-3 model is `0.03302`. Most movement occurred early:

| Comparison | Global relative parameter delta |
| --- | ---: |
| Parent to Stage-1 midpoint | `0.02994` |
| Stage-1 midpoint to selected | `0.00819` |
| Stage-1 selected to Stage-2 midpoint | `0.00728` |
| Stage-2 midpoint to selected | `0.00191` |
| Stage-2 selected to Stage-3 midpoint | `0.000533` |
| Stage-3 midpoint to selected | `0.0000971` |

This monotonic reduction is consistent with a large domain shift followed by
increasingly narrow specialization. It is not evidence that late stages were
unimportant: a small parameter movement can change behavior materially.

Across the parent-to-final comparison, selected embedding rows moved by mean
relative L2 `0.05624`, while their mean top-20-neighbor Jaccard remained
`0.9755`. The LM head moved less (`0.01140`, Jaccard `0.9876`). During the late
half of Stage 3, the corresponding embedding and LM-head movements were only
`0.0000883` and `0.0000203`, with neighbor Jaccard `1.0`. The curriculum altered
vectors measurably without wholesale reorganization of the inspected local
neighborhoods.

## What Happened To Representations

On 48 fixed probes spanning historical general text, non-sonnet poetry,
sonnets, and modern instruction text, parent-to-final mean hidden-state drift
was `0.2394`; the standard-sonnet probes drifted most (`0.3177`) and modern
instruction probes least (`0.1599`). The minimum linear CKA remained `0.9219`,
so the representation space changed substantially but did not lose its broad
geometry. Mean top-20 next-token overlap fell to `0.4994`, and mean logit
entropy decreased by `0.2102`.

Again, most change was early. The late Stage-3 half had mean relative drift
`0.00371`, minimum CKA `0.999997`, and top-20 logit overlap `0.9668`. These
results support a curriculum in which Stage 1 performs the principal domain
adaptation, Stage 2 reinforces poetic behavior, and Stage 3 makes a small
sonnet-specific adjustment. They do not identify a single causal layer.

## Generation Findings Before DPO

The seven-state study used a 504-output confirmatory grid and a 20,160-output
high-volume validation grid. Every state and recipe was rated under frozen
automatic and blinded protocols. No high-risk memorization output was found.

The final Stage-3 state reliably preserved the supplied opening and decoder-
controlled line count, but strict literary quality remained unreliable. In the
504-row blinded high-volume sample, Stage 3 supplied 11/72 moderate-clean
outputs and 0 strict-good outputs; creative decoding supplied 8/24 of those
moderate outputs. A separate 120-row blinded prompt experiment found only 3
moderate-clean and 1 strict-good output across all arms.

A prompt that explicitly prohibited labels and prose improved the automatic
meta-text-free rate by 6.25 percentage points (95% prompt-cluster interval
`+2.08` to `+10.31`). An explicit 4+4+3+3 structural prompt sharply worsened
meta-text, and deterministic retries did not improve the strict screen. The
no-label/prose prompt with creative decoding was therefore selected for the
bounded DPO validation and final protocol.

## AI-Judged DPO

The preference branch used 534 training-only pairs and three blinded AI votes
per pair. It must not be called human-calibrated or human-aligned: AI-majority
preferences agreed with the user's separate 20-pair review on only 12/20
(60%). The 482/52 prompt-disjoint DPO split trained for 61 optimizer updates.

On 960 matched validation outputs, DPO increased the automatic surface-screen
rate from 13.96% to 18.96% (paired `+5.00` points, 95% interval `+0.63` to
`+9.38`). In the frozen 80-output validation review, genuine terminal
completion rose from 12/40 to 20/40. Grammar and form were approximately
unchanged, and neither system produced a strict-good reviewed output.
Preservation losses changed by at most `+0.01334`, with the largest change on
instruction validation.

## One-Time V7 Test

The final system, comparator, prompt, decoder, stopping rule, metrics, and blind
sample were hash-frozen before opening the V7 test. The one-time run covers all
1,244 test openings, two seeds, and both systems: 4,976 outputs. No retuning or
rerun is allowed after this access.

| Test metric | Stage 3 | DPO | Paired DPO change (95% interval) |
| --- | ---: | ---: | ---: |
| Opening preserved | 100.00% | 100.00% | `0.00` |
| Fourteen lines | 99.96% | 99.92% | `-0.04` points (`-0.20`, `+0.08`) |
| Meta-text free | 86.33% | 87.78% | `+1.45` points (`-0.28`, `+3.18`) |
| Terminal punctuation | 17.60% | 20.46% | `+2.85` points (`+0.72`, `+4.86`) |
| Automatic surface screen | 15.07% | 17.60% | `+2.53` points (`+0.52`, `+4.50`) |
| High memorization risk | 0/2,488 | 0/2,488 | `0` |

These results replicate a modest surface/completion improvement on the sealed
test. They do not establish acceptable sonnet quality.

The preregistered blind review sampled 100 matched test prompts, one frozen
seed per prompt, and both systems. An AI qualitative analyst scored all 200
outputs before identities were revealed. DPO-minus-Stage-3 prompt-paired
results were:

| Blind literary measure | Stage 3 | DPO | Paired DPO change (95% interval) |
| --- | ---: | ---: | ---: |
| Grammar mean (1--5) | `2.93` | `2.92` | `-0.01` (`-0.20`, `+0.18`) |
| Historical-register mean | `2.88` | `3.09` | `+0.21` (`+0.04`, `+0.38`) |
| Poetic-quality mean | `2.63` | `2.69` | `+0.06` (`-0.12`, `+0.23`) |
| Sonnet/form mean | `1.69` | `1.78` | `+0.09` (`-0.03`, `+0.22`) |
| Volta/argument mean | `2.63` | `2.63` | `0.00` (`-0.21`, `+0.21`) |
| Visibly complete | 35/100 | 41/100 | `+6` points (`-9`, `+21`) |
| Moderate-clean | 0/100 | 3/100 | `+3` points (`0`, `+7`) |
| Strict-good | 0/100 | 0/100 | `0` |

Only the historical-register interval excluded zero. DPO did not demonstrate
a reliable grammar, poetic-quality, form, volta, or genuine-completion gain.
The three moderate-clean DPO outputs are a descriptive positive signal, not
evidence of consistently acceptable sonnets. Neither system produced a
strict-good output; sonnet/form was the weakest dimension for both.

One of the three moderate-clean DPO examples began `Io ritorno pur, lasso, al
loco amato`. The reviewer found recognizable 4+4+3+3 progression, a turn marked
by `Indi`, historical register, and a complete final volition, while explicitly
noting weak rhyme. Its matched Stage-3 output ended with the drafting label
`Secondo verso:` and repeated `al paese`. This pair illustrates the targeted
gain without overstating it: DPO can remove a visible meta-text/completion
failure on some prompts, but the retained poem still lacks a strong rhyme plan
and received only 3/5 for grammar, poetry, form, and volta.

## Conclusions And Limitations

- Full-weight historical adaptation produced most of the parameter and
  representation movement.
- Poetry adaptation added meaningful domain improvement with smaller movement.
- Sonnet specialization was narrowly targeted and preserved earlier domains,
  but reliable rhyme, grammar, volta, and closure did not emerge.
- Prompt wording reduced meta-text without solving literary structure.
- AI-judged DPO produced small, reproducible completion/surface gains, not a
  transformation into a consistently good sonnet generator.
- Final blind review found a bounded historical-register gain but no reliable
  broad literary-quality improvement; strict-good yield was 0/100 for both.
- Fourteen-line stopping is decoder control, not learned rhyme, metre, or stanza
  organization.
- The selected token registry and 48-probe set are bounded samples.
- CKA, drift, and weight deltas are descriptive; no layer-restoration or
  ablation result licenses a causal claim here.
- The AI judges are correlated and failed the human-calibration threshold.
- Surface memorization checks cannot detect every type of recall or unknown
  overlap in the external parent pretraining corpus.

## Reproducibility

Public code/configuration includes the state registry, weight/embedding/
representation analyzers, high-volume generator, prompt intervention, DPO
trainer, matched validation, preservation evaluation, and one-time final-test
analyzer. Full BF16 checkpoints, adapters, raw generations, private mappings,
and annotations remain in the verified machine-local archive because of size
and data-lineage restrictions.
