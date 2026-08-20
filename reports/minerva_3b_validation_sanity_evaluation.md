# Minerva 3B Validation Sanity Audit

## Question

The fixed Minerva test showed that QLoRA substantially improved 14-line task
behavior but produced unexpectedly ungrammatical and collapsed continuations.
This validation-only audit tested whether that result came from an overly strict
final-test rubric, weak Base prompting, excessive adapter strength, or the
selected checkpoint.

## Protocol

- Model: `sapienzanlp/Minerva-3B-base-v1.0`, revision
  `129ae5366bae3611a1c9f8c68606c38b7de8b055`.
- Data: eight V5 validation openings from eight authors and five centuries.
- Final-test isolation: no fixed test opening or output participated.
- Decoding: seed 4242, temperature 0.8, top-k 50, 512-token ceiling, and
  decoder-enforced stopping after thirteen generated continuation lines.
- Conditions: raw Base, explicitly instructed Base, four epoch-3 adapter
  strengths, and the epoch-6 full-strength overfitting contrast.
- Review: all 56 outputs were judged behind hashed identifiers before the
  condition mapping or automatic report was opened.

The fixed blinded decisions are in
[`minerva_3b_validation_sanity_blinded_judgments.md`](minerva_3b_validation_sanity_blinded_judgments.md).
The complete samples and prompt-level review are retained locally and excluded
from the public tree.

## Results

| Condition | Role | Form | Grammatical | Seven-line topic | Severe collapse | Mean repetition |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Base, raw opening | diagnostic | 7/8 | 5/8 | 7/8 | 3/8 | 0.3892 |
| Base, explicit instruction | diagnostic | 6/8 | 3/8 | 8/8 | 5/8 | 0.2917 |
| Epoch 3, scale 0.25 | eligible | 8/8 | 0/8 | 8/8 | 5/8 | 0.3537 |
| Epoch 3, scale 0.50 | eligible | 8/8 | 0/8 | 8/8 | 1/8 | 0.1902 |
| Epoch 3, scale 0.75 | eligible | 8/8 | 0/8 | 7/8 | 0/8 | 0.1895 |
| Epoch 3, scale 1.00 | eligible | 8/8 | 1/8 | 8/8 | 0/8 | 0.1693 |
| Epoch 6, scale 1.00 | diagnostic | 8/8 | 0/8 | 8/8 | 0/8 | 0.1579 |

The Base conditions often generated grammatical modern prose, but commonly
ignored the historical-sonnet task. Examples became essays about automobiles,
artemisia, death, or sonnet definitions. The explicit instruction did not
improve this Base checkpoint: it reduced form completion and increased severe
collapse, consistent with evaluating a base language model rather than an
instruction-tuned chat model.

The adapters made length and line production reliable and usually preserved a
recognizable topic. They also shifted the text toward archaic-looking diction.
That surface adaptation did not preserve syntax. Across the four selectable
epoch-3 strengths, only one of 32 outputs was generally grammatical. Lowering
adapter strength therefore did not recover Base fluency; scale 0.25 instead
reintroduced Base-like repetition while retaining malformed pseudo-archaic
language.

The lower automatic repetition at larger scales is not evidence of better
language. Epoch 6 has the lowest mean repetition and zero severe loops, yet none
of its eight outputs is generally grammatical. This is why the predeclared
human criterion was necessary.

## Decision

**No epoch-3 adapter strength qualifies. No additional final-test rerun will be
performed.**

Each selectable condition needed at least 5/8 generally grammatical outputs.
Their counts were 0, 0, 0, and 1. This margin is too large to attribute the
failure to borderline judgments or an overly strict `12/20` final threshold.
The original QLoRA failure remains the project result.

The diagnostic identifies a real tradeoff in this recipe:

- untouched Base retains more ordinary Italian fluency but does not reliably
  perform the requested sonnet continuation;
- QLoRA learns the visible form, historical lexical cues, and topic persistence
  while degrading grammatical composition.

This does not establish that Minerva 3B cannot generate acceptable sonnets
under every possible post-training method. It establishes that the project's
fixed rank-16, all-attention-and-MLP, opening-line-continuation QLoRA recipe does
not do so, and that post-hoc adapter scaling does not rescue it.

The next project checkpoint returns to the separately approved untouched-Minerva
judge-validation gate. That gate must independently establish whether this Base
checkpoint can rank quality reliably before DPO or GRPO is permitted.
