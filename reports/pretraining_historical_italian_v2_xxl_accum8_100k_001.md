# Pretraining Run: pretraining_historical_italian_v2_xxl_accum8_100k_001

This report records a from-scratch broader-Italian-corpus pretraining run. Raw corpus files, interval checkpoints, and the final checkpoint are intentionally local-only; the configuration and observed results are preserved here.

## Configuration

| Setting | Value |
| --- | --- |
| Device | cuda:0 |
| Vocabulary size | 16,000 |
| Training tokens | 17,891,995 |
| Validation tokens | 180,745 |
| Context length | 512 |
| Batch size | 1 |
| Completed steps | 100,000 |
| Learning rate | 3.0e-04 |
| Learning-rate schedule | warmup_cosine |
| Warmup steps | 1,000 |
| Minimum learning rate | 3.0e-05 |
| Evaluation | every 1,000 steps; all 353 sequential windows |
| Interval checkpoints | every 5,000 steps |
| Parameters | 234,839,008 |
| Embedding dimension | 1024 |
| Transformer layers | 16 |
| Attention heads | 16 |
| Head dimension | 64 |
| Feed-forward dimension | 2731 |
| Normalization | layer_norm |
| Normalization epsilon | 1.0e-05 |
| Position encoding | learned_absolute |
| RoPE theta | 10000 |
| Feed-forward type | swiglu |
| Tied token embeddings | False |

## Loss Summary

| Measurement | Step | Training loss | Validation loss |
| --- | ---: | ---: | ---: |
| First recorded evaluation | 25,001 | 2.0763 | 2.7224 |
| Best validation evaluation | 18,000 | not retained | 2.6694 |
| Final evaluation | 100,000 | 0.1601 | 5.0911 |

## Saved Local Artifacts

- Interval checkpoints: 0
- Final checkpoint size: 2944.8 MiB
- Loss-history records: 76

## Final Sample Excerpt

```text
Nel qual riedificazione venne l’ultimo detto Hppolise con grande esercito in Asia e la discesa l'anno quadragesimo detto.
È in questo luoco a passar un vitello, un uomo solo chiamato Dront, cioè Macomettore, cosí chiamato, che era signor di Tripoli e di Soria; e vi era un uomo di fortezza portoghese, i quali vi tenevano delle navi, cioè quelle che di prima erano restassino casali, caravelle, perle, per sorate, caravaniche, acqua o panni che se ne manda, quali n'avevano fatto baratti dell'India e i nobili di Romani che riseggono e avevano offerta, con promissione di molte cose. Per il che, quando lo vedevano entrare nel porto e commissione, lui non poteva piú vedere e sapeva, perché non vi sapeva, che andavano ora Barletta e in Rodi, che mai non viddero nuova terra in
```

## Interpretation

The loss fell substantially from the first recorded evaluation, and the sample has learned historical Italian prose-like texture. It is not sonnet-specialized: that is the intended role of the next fine-tuning stage. Each validation result covers the complete fixed holdout through non-overlapping sequential windows, so the best-validation checkpoint is a deterministic selection within this run. Validation was lower at step 18,000 than at the final step, so downstream generation and fine-tuning should use `best_validation.pt`, not `model.pt`.

## Full Loss History

| Step | Training loss | Validation loss | Learning rate |
| ---: | ---: | ---: | ---: |
| 25,001 | 2.0763 | 2.7224 | 2.63e-04 |
| 26,000 | 1.9331 | 2.7452 | 2.60e-04 |
| 27,000 | 1.7784 | 2.7544 | 2.57e-04 |
| 28,000 | 1.9045 | 2.7821 | 2.53e-04 |
| 29,000 | 1.5150 | 2.8072 | 2.50e-04 |
| 30,000 | 1.7116 | 2.8094 | 2.47e-04 |
| 31,000 | 1.5348 | 2.8299 | 2.43e-04 |
| 32,000 | 1.3658 | 2.8553 | 2.40e-04 |
| 33,000 | 1.5085 | 2.8919 | 2.36e-04 |
| 34,000 | 1.3081 | 2.9289 | 2.32e-04 |
| 35,000 | 1.3808 | 2.9552 | 2.29e-04 |
| 36,000 | 1.2158 | 2.9929 | 2.25e-04 |
| 37,000 | 1.3951 | 3.0091 | 2.21e-04 |
| 38,000 | 1.6595 | 3.0552 | 2.17e-04 |
| 39,000 | 1.3628 | 3.0349 | 2.13e-04 |
| 40,000 | 1.0082 | 3.1204 | 2.09e-04 |
| 41,000 | 1.0646 | 3.1448 | 2.05e-04 |
| 42,000 | 1.3945 | 3.1742 | 2.01e-04 |
| 43,000 | 0.9997 | 3.2295 | 1.97e-04 |
| 44,000 | 1.0398 | 3.2838 | 1.93e-04 |
| 45,000 | 0.9675 | 3.3509 | 1.88e-04 |
| 46,000 | 0.9159 | 3.3580 | 1.84e-04 |
| 47,000 | 0.9598 | 3.3974 | 1.80e-04 |
| 48,000 | 0.8219 | 3.4262 | 1.76e-04 |
| 49,000 | 0.8804 | 3.4769 | 1.71e-04 |
| 50,000 | 0.9609 | 3.5211 | 1.67e-04 |
| 51,000 | 0.9643 | 3.5352 | 1.63e-04 |
| 52,000 | 0.7826 | 3.5907 | 1.59e-04 |
| 53,000 | 0.7455 | 3.5922 | 1.54e-04 |
| 54,000 | 0.7910 | 3.6356 | 1.50e-04 |
| 55,000 | 0.9097 | 3.6785 | 1.46e-04 |
| 56,000 | 0.6595 | 3.7287 | 1.42e-04 |
| 57,000 | 0.8004 | 3.7575 | 1.37e-04 |
| 58,000 | 0.7210 | 3.8084 | 1.33e-04 |
| 59,000 | 0.5121 | 3.8545 | 1.29e-04 |
| 60,000 | 0.6779 | 3.8785 | 1.25e-04 |
| 61,000 | 0.6366 | 3.9198 | 1.21e-04 |
| 62,000 | 0.4952 | 3.9714 | 1.17e-04 |
| 63,000 | 0.5162 | 4.0205 | 1.13e-04 |
| 64,000 | 0.6341 | 4.0700 | 1.09e-04 |
| 65,000 | 0.4312 | 4.1045 | 1.05e-04 |
| 66,000 | 0.4761 | 4.1516 | 1.01e-04 |
| 67,000 | 0.3502 | 4.1994 | 9.75e-05 |
| 68,000 | 0.2871 | 4.2472 | 9.38e-05 |
| 69,000 | 0.3358 | 4.2876 | 9.02e-05 |
| 70,000 | 0.2830 | 4.3374 | 8.67e-05 |
| 71,000 | 0.2651 | 4.3652 | 8.32e-05 |
| 72,000 | 0.2608 | 4.3927 | 7.99e-05 |
| 73,000 | 0.2791 | 4.4395 | 7.66e-05 |
| 74,000 | 0.2314 | 4.4735 | 7.34e-05 |
| 75,000 | 0.2064 | 4.5019 | 7.03e-05 |
| 76,000 | 0.2467 | 4.5563 | 6.73e-05 |
| 77,000 | 0.1992 | 4.5939 | 6.44e-05 |
| 78,000 | 0.2225 | 4.6153 | 6.16e-05 |
| 79,000 | 0.2771 | 4.6552 | 5.89e-05 |
| 80,000 | 0.2353 | 4.6899 | 5.63e-05 |
| 81,000 | 0.1995 | 4.7296 | 5.38e-05 |
| 82,000 | 0.2453 | 4.7522 | 5.14e-05 |
| 83,000 | 0.2668 | 4.7854 | 4.92e-05 |
| 84,000 | 0.1786 | 4.8009 | 4.70e-05 |
| 85,000 | 0.2312 | 4.8372 | 4.50e-05 |
| 86,000 | 0.1793 | 4.8542 | 4.31e-05 |
| 87,000 | 0.1774 | 4.8649 | 4.13e-05 |
| 88,000 | 0.1795 | 4.8957 | 3.97e-05 |
| 89,000 | 0.1590 | 4.9089 | 3.81e-05 |
| 90,000 | 0.1543 | 4.9322 | 3.67e-05 |
| 91,000 | 0.1684 | 4.9559 | 3.55e-05 |
| 92,000 | 0.1404 | 4.9723 | 3.43e-05 |
| 93,000 | 0.1748 | 4.9900 | 3.33e-05 |
| 94,000 | 0.1415 | 5.0073 | 3.24e-05 |
| 95,000 | 0.1719 | 5.0184 | 3.17e-05 |
| 96,000 | 0.1713 | 5.0340 | 3.11e-05 |
| 97,000 | 0.1454 | 5.0444 | 3.06e-05 |
| 98,000 | 0.1449 | 5.0674 | 3.03e-05 |
| 99,000 | 0.1749 | 5.0794 | 3.01e-05 |
| 100,000 | 0.1601 | 5.0911 | 3.00e-05 |
