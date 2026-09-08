# Minerva 7B V7 Memorization Selection Correction

Date: 2026-09-08

The V7 nearest-reference selector previously ranked a match by character
40-gram containment before its longest common substring. Because the high-risk
rule uses either threshold, a medium-risk match with higher containment could
hide a high-risk match with a substring of at least 160 characters.

The corrected selector ranks risk first, then containment, then longest common
substring length. It uses the reference identity as a stable final tie-breaker.
Regression tests cover the former OR-threshold failure and prove that reversing
the reference order does not change the result.

The saved one-time final outputs were scored again against the local,
hash-verified V7 sonnet training reference. No poem text or generated text was
written to this report. The frozen generation and analysis artifacts were not
changed.

| System | Outputs | Low risk | Medium risk | High risk | Changed risk labels | Changed nearest references |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage 3 | 2,488 | 2,488 | 0 | 0 | 0 | 0 |
| DPO | 2,488 | 2,488 | 0 | 0 | 0 | 0 |

The published result remains unchanged: **0 high-risk outputs in 4,976 saved
final outputs**.

Verification identities:

- final completion manifest SHA-256:
  `943da8b647d3d29b5e1457b70cfe15c934b84d71444fccf8e141e19fb489670a`
- memorization reference manifest SHA-256:
  `5a4223d00fd6e09604340ebbe8d24f2f90588dfd1aa86c7abb2165a8215d6ad8`
