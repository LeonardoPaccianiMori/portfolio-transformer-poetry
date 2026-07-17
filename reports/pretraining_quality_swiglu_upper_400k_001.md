# Pretraining Run: pretraining_quality_swiglu_upper_400k_001

This report records a from-scratch broader-Italian-corpus pretraining run. Raw corpus files, interval checkpoints, and the final checkpoint are intentionally local-only; the configuration and observed results are preserved here.

## Configuration

| Setting | Value |
| --- | --- |
| Device | cuda:0 |
| Vocabulary size | 8,000 |
| Training tokens | 17,121,479 |
| Validation tokens | 172,961 |
| Context length | 512 |
| Batch size | 1 |
| Completed steps | 400,000 |
| Learning rate | 3.0e-04 |
| Learning-rate schedule | warmup_cosine |
| Warmup steps | 2,000 |
| Minimum learning rate | 3.0e-05 |
| Evaluation | every 5,000 steps; all 337 sequential windows |
| Interval checkpoints | every 25,000 steps |
| Parameters | 59,807,900 |
| Embedding dimension | 640 |
| Transformer layers | 10 |
| Attention heads | 10 |
| Head dimension | 64 |
| Feed-forward dimension | 1707 |
| Normalization | layer_norm |
| Normalization epsilon | 1.0e-05 |
| Position encoding | learned_absolute |
| RoPE theta | 10000 |
| Feed-forward type | swiglu |
| Tied token embeddings | False |

## Loss Summary

| Measurement | Step | Training loss | Validation loss |
| --- | ---: | ---: | ---: |
| First recorded evaluation | 1 | 9.1869 | 9.1825 |
| Best validation evaluation | 180,000 | 2.1168 | 2.4983 |
| Final evaluation | 400,000 | 1.1551 | 2.7760 |

## Saved Local Artifacts

- Interval checkpoints: 16
- Final checkpoint size: 784.9 MiB
- Loss-history records: 81

## Final Sample Excerpt

```text
Nel Santo Siena, poi che papa Benedetto si disfece e comperò, la città ne divenne molto più bella. E dopo la sua morte si fece generale marchese il vescovo e 'l suo fratel carnale del Patrimonio arcivescovo di Baviera, i più messi e maggiori cherici dali co·lloro ricchezze, molto favorati, e brivilegiadori, i quali non furono in accordo de la coronazione reale, quando furono a lo 'mperadore tre tra grande e ripresi.
CLXXXVII
Come il Comune di Firenze cominciò guerra in mare e in terra.
Nell'anno MCCCVIII, maestrato a' due anni di Cristo MCCLXXXXV, per le lunghe piove, discendeano in grande parte i pianeve dentro al segno del Leone per l'Ombria, e per lo scedessono e recati al solinletto, e del mese di... si faceano innanzi e
```

## Interpretation

The loss fell substantially from the first recorded evaluation, and the sample has learned historical Italian prose-like texture. It is not sonnet-specialized: that is the intended role of the next fine-tuning stage. Each validation result covers the complete fixed holdout through non-overlapping sequential windows, so the best-validation checkpoint is a deterministic selection within this run. Validation was lower at step 180,000 than at the final step, so downstream generation and fine-tuning should use `best_validation.pt`, not `model.pt`.

## Full Loss History

| Step | Training loss | Validation loss | Learning rate |
| ---: | ---: | ---: | ---: |
| 1 | 9.1869 | 9.1825 | 1.50e-07 |
| 5,000 | 3.7914 | 3.9621 | 3.00e-04 |
| 10,000 | 3.5185 | 3.7809 | 3.00e-04 |
| 15,000 | 3.5235 | 3.6870 | 2.99e-04 |
| 20,000 | 3.4172 | 3.4076 | 2.99e-04 |
| 25,000 | 3.0402 | 3.1831 | 2.98e-04 |
| 30,000 | 2.7149 | 3.0592 | 2.97e-04 |
| 35,000 | 2.9378 | 2.9801 | 2.95e-04 |
| 40,000 | 2.5068 | 2.9210 | 2.94e-04 |
| 45,000 | 2.8110 | 2.8541 | 2.92e-04 |
| 50,000 | 2.4238 | 2.8194 | 2.90e-04 |
| 55,000 | 3.0767 | 2.7692 | 2.88e-04 |
| 60,000 | 2.2693 | 2.7444 | 2.86e-04 |
| 65,000 | 2.4202 | 2.7280 | 2.84e-04 |
| 70,000 | 2.3741 | 2.6953 | 2.81e-04 |
| 75,000 | 2.2427 | 2.6676 | 2.78e-04 |
| 80,000 | 2.5539 | 2.6567 | 2.75e-04 |
| 85,000 | 2.4283 | 2.6433 | 2.72e-04 |
| 90,000 | 1.6739 | 2.6229 | 2.69e-04 |
| 95,000 | 2.1051 | 2.6034 | 2.65e-04 |
| 100,000 | 2.0396 | 2.5987 | 2.62e-04 |
| 105,000 | 1.9622 | 2.5903 | 2.58e-04 |
| 110,000 | 1.9837 | 2.5745 | 2.54e-04 |
| 115,000 | 2.2412 | 2.5739 | 2.50e-04 |
| 120,000 | 2.2471 | 2.5522 | 2.46e-04 |
| 125,000 | 2.4319 | 2.5456 | 2.41e-04 |
| 130,000 | 2.3758 | 2.5438 | 2.37e-04 |
| 135,000 | 1.8084 | 2.5504 | 2.32e-04 |
| 140,000 | 1.7214 | 2.5322 | 2.28e-04 |
| 145,000 | 2.5149 | 2.5334 | 2.23e-04 |
| 150,000 | 2.0525 | 2.5277 | 2.18e-04 |
| 155,000 | 1.9839 | 2.5241 | 2.13e-04 |
| 160,000 | 2.2656 | 2.5201 | 2.08e-04 |
| 165,000 | 2.5451 | 2.5076 | 2.03e-04 |
| 170,000 | 2.2066 | 2.5029 | 1.98e-04 |
| 175,000 | 2.2868 | 2.5074 | 1.93e-04 |
| 180,000 | 2.1168 | 2.4983 | 1.87e-04 |
| 185,000 | 1.5994 | 2.5124 | 1.82e-04 |
| 190,000 | 2.0621 | 2.5038 | 1.77e-04 |
| 195,000 | 1.7655 | 2.5146 | 1.71e-04 |
| 200,000 | 1.7773 | 2.5115 | 1.66e-04 |
| 205,000 | 2.1351 | 2.5097 | 1.61e-04 |
| 210,000 | 2.0641 | 2.5098 | 1.55e-04 |
| 215,000 | 1.5800 | 2.5080 | 1.50e-04 |
| 220,000 | 2.0494 | 2.5255 | 1.45e-04 |
| 225,000 | 1.5238 | 2.5156 | 1.40e-04 |
| 230,000 | 1.5105 | 2.5240 | 1.34e-04 |
| 235,000 | 2.1778 | 2.5324 | 1.29e-04 |
| 240,000 | 1.3420 | 2.5306 | 1.24e-04 |
| 245,000 | 2.0315 | 2.5371 | 1.19e-04 |
| 250,000 | 1.9446 | 2.5464 | 1.14e-04 |
| 255,000 | 1.7027 | 2.5427 | 1.09e-04 |
| 260,000 | 1.3174 | 2.5519 | 1.04e-04 |
| 265,000 | 2.1742 | 2.5609 | 9.97e-05 |
| 270,000 | 1.7939 | 2.5704 | 9.51e-05 |
| 275,000 | 2.1114 | 2.5756 | 9.06e-05 |
| 280,000 | 1.7094 | 2.5836 | 8.62e-05 |
| 285,000 | 1.3809 | 2.5815 | 8.19e-05 |
| 290,000 | 1.3236 | 2.5963 | 7.78e-05 |
| 295,000 | 1.6363 | 2.5956 | 7.38e-05 |
| 300,000 | 1.5440 | 2.6114 | 6.99e-05 |
| 305,000 | 1.8743 | 2.6203 | 6.62e-05 |
| 310,000 | 2.0877 | 2.6314 | 6.27e-05 |
| 315,000 | 1.1921 | 2.6368 | 5.93e-05 |
| 320,000 | 1.8325 | 2.6365 | 5.60e-05 |
| 325,000 | 1.5696 | 2.6453 | 5.30e-05 |
| 330,000 | 1.1175 | 2.6639 | 5.01e-05 |
| 335,000 | 1.2777 | 2.6626 | 4.74e-05 |
| 340,000 | 1.1392 | 2.6760 | 4.49e-05 |
| 345,000 | 1.5624 | 2.6898 | 4.25e-05 |
| 350,000 | 1.3295 | 2.6918 | 4.04e-05 |
| 355,000 | 2.0080 | 2.7003 | 3.84e-05 |
| 360,000 | 1.7842 | 2.7102 | 3.67e-05 |
| 365,000 | 1.6385 | 2.7158 | 3.51e-05 |
| 370,000 | 1.3449 | 2.7361 | 3.38e-05 |
| 375,000 | 1.3110 | 2.7367 | 3.26e-05 |
| 380,000 | 1.4309 | 2.7442 | 3.17e-05 |
| 385,000 | 1.8047 | 2.7526 | 3.09e-05 |
| 390,000 | 1.2675 | 2.7565 | 3.04e-05 |
| 395,000 | 1.5435 | 2.7712 | 3.01e-05 |
| 400,000 | 1.1551 | 2.7760 | 3.00e-05 |
