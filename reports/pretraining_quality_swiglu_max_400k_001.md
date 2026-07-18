# Pretraining Run: pretraining_quality_swiglu_max_400k_001

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
| Parameters | 97,729,856 |
| Embedding dimension | 768 |
| Transformer layers | 12 |
| Attention heads | 12 |
| Head dimension | 64 |
| Feed-forward dimension | 2048 |
| Normalization | layer_norm |
| Normalization epsilon | 1.0e-05 |
| Position encoding | learned_absolute |
| RoPE theta | 10000 |
| Feed-forward type | swiglu |
| Tied token embeddings | False |

## Loss Summary

| Measurement | Step | Training loss | Validation loss |
| --- | ---: | ---: | ---: |
| First recorded evaluation | 1 | 9.0885 | 9.0914 |
| Best validation evaluation | 180,000 | 1.6357 | 2.5164 |
| Final evaluation | 400,000 | 1.2559 | 2.8853 |

## Saved Local Artifacts

- Interval checkpoints: 16
- Final checkpoint size: 1263.1 MiB
- Loss-history records: 81

## Final Sample Excerpt

```text
Nel Santo lor Giano doveano por le lor mento accese infino a dì X di febraio; e notari, che andava la mattina a piè, che andava in sul crocifisso dallato a' tambuchi e scalzo; e quella mattina anzi un garzone digrizia, che per cotanto andarsi scuotono, leggier pane e vino e l'onore e 'l passatempo con allegrezza e con gran festa, non possendo sì altri sofficienti avere il beneficio, il fecero chiamare calamitoso e crudele e insanguinoso e incommutabile peccato, come in Cristo dinanzi al 30Fiumile stato figurato abbiamo. Morto tempo tranquillo clò il grano, e a Parigi fatti, doltre, e infino alla fine di nostra Donna in questo capitolo, e ora Gismini si fece doge, e fecelivi mettere, innanzi alla
```

## Interpretation

The loss fell substantially from the first recorded evaluation, and the sample has learned historical Italian prose-like texture. It is not sonnet-specialized: that is the intended role of the next fine-tuning stage. Each validation result covers the complete fixed holdout through non-overlapping sequential windows, so the best-validation checkpoint is a deterministic selection within this run. Validation was lower at step 180,000 than at the final step, so downstream generation and fine-tuning should use `best_validation.pt`, not `model.pt`.

## Full Loss History

| Step | Training loss | Validation loss | Learning rate |
| ---: | ---: | ---: | ---: |
| 1 | 9.0885 | 9.0914 | 1.50e-07 |
| 5,000 | 3.5918 | 3.9484 | 3.00e-04 |
| 10,000 | 3.9348 | 3.8286 | 3.00e-04 |
| 15,000 | 3.5407 | 3.7587 | 2.99e-04 |
| 20,000 | 3.3926 | 3.6322 | 2.99e-04 |
| 25,000 | 3.0216 | 3.4020 | 2.98e-04 |
| 30,000 | 3.1720 | 3.2242 | 2.97e-04 |
| 35,000 | 2.7249 | 3.1108 | 2.95e-04 |
| 40,000 | 2.5568 | 3.0173 | 2.94e-04 |
| 45,000 | 2.8343 | 2.9449 | 2.92e-04 |
| 50,000 | 2.5176 | 2.9026 | 2.90e-04 |
| 55,000 | 2.6899 | 2.8386 | 2.88e-04 |
| 60,000 | 2.7671 | 2.8015 | 2.86e-04 |
| 65,000 | 2.4507 | 2.7900 | 2.84e-04 |
| 70,000 | 2.4091 | 2.7483 | 2.81e-04 |
| 75,000 | 2.6288 | 2.7158 | 2.78e-04 |
| 80,000 | 2.6279 | 2.6998 | 2.75e-04 |
| 85,000 | 2.2902 | 2.6705 | 2.72e-04 |
| 90,000 | 2.1009 | 2.6676 | 2.69e-04 |
| 95,000 | 2.0698 | 2.6287 | 2.65e-04 |
| 100,000 | 2.3731 | 2.6148 | 2.62e-04 |
| 105,000 | 2.7516 | 2.6141 | 2.58e-04 |
| 110,000 | 2.5240 | 2.5989 | 2.54e-04 |
| 115,000 | 2.2277 | 2.5911 | 2.50e-04 |
| 120,000 | 2.3077 | 2.5753 | 2.46e-04 |
| 125,000 | 2.3166 | 2.5720 | 2.41e-04 |
| 130,000 | 1.9700 | 2.5557 | 2.37e-04 |
| 135,000 | 2.0727 | 2.5586 | 2.32e-04 |
| 140,000 | 2.1184 | 2.5528 | 2.28e-04 |
| 145,000 | 1.7639 | 2.5435 | 2.23e-04 |
| 150,000 | 2.0075 | 2.5394 | 2.18e-04 |
| 155,000 | 2.0064 | 2.5426 | 2.13e-04 |
| 160,000 | 1.6026 | 2.5363 | 2.08e-04 |
| 165,000 | 2.3075 | 2.5249 | 2.03e-04 |
| 170,000 | 2.1190 | 2.5380 | 1.98e-04 |
| 175,000 | 2.4856 | 2.5206 | 1.93e-04 |
| 180,000 | 1.6357 | 2.5164 | 1.87e-04 |
| 185,000 | 1.3520 | 2.5290 | 1.82e-04 |
| 190,000 | 2.5303 | 2.5352 | 1.77e-04 |
| 195,000 | 2.1011 | 2.5330 | 1.71e-04 |
| 200,000 | 2.0165 | 2.5231 | 1.66e-04 |
| 205,000 | 1.9568 | 2.5335 | 1.61e-04 |
| 210,000 | 1.9603 | 2.5421 | 1.55e-04 |
| 215,000 | 1.8696 | 2.5448 | 1.50e-04 |
| 220,000 | 1.8338 | 2.5534 | 1.45e-04 |
| 225,000 | 1.5946 | 2.5524 | 1.40e-04 |
| 230,000 | 1.7950 | 2.5530 | 1.34e-04 |
| 235,000 | 1.6358 | 2.5665 | 1.29e-04 |
| 240,000 | 1.3223 | 2.5644 | 1.24e-04 |
| 245,000 | 1.5979 | 2.5790 | 1.19e-04 |
| 250,000 | 1.3314 | 2.5762 | 1.14e-04 |
| 255,000 | 1.8484 | 2.5851 | 1.09e-04 |
| 260,000 | 1.5771 | 2.5956 | 1.04e-04 |
| 265,000 | 2.3866 | 2.5990 | 9.97e-05 |
| 270,000 | 1.1577 | 2.6058 | 9.51e-05 |
| 275,000 | 1.3668 | 2.6196 | 9.06e-05 |
| 280,000 | 1.7985 | 2.6356 | 8.62e-05 |
| 285,000 | 1.3984 | 2.6472 | 8.19e-05 |
| 290,000 | 1.7111 | 2.6506 | 7.78e-05 |
| 295,000 | 1.4499 | 2.6684 | 7.38e-05 |
| 300,000 | 1.4607 | 2.6727 | 6.99e-05 |
| 305,000 | 1.1640 | 2.7022 | 6.62e-05 |
| 310,000 | 1.7782 | 2.6969 | 6.27e-05 |
| 315,000 | 1.2641 | 2.7090 | 5.93e-05 |
| 320,000 | 1.0795 | 2.7080 | 5.60e-05 |
| 325,000 | 0.8494 | 2.7327 | 5.30e-05 |
| 330,000 | 1.3302 | 2.7318 | 5.01e-05 |
| 335,000 | 1.7882 | 2.7504 | 4.74e-05 |
| 340,000 | 1.0432 | 2.7578 | 4.49e-05 |
| 345,000 | 1.9513 | 2.7667 | 4.25e-05 |
| 350,000 | 1.5031 | 2.7821 | 4.04e-05 |
| 355,000 | 1.1609 | 2.7878 | 3.84e-05 |
| 360,000 | 0.9416 | 2.8008 | 3.67e-05 |
| 365,000 | 1.5706 | 2.8089 | 3.51e-05 |
| 370,000 | 1.5809 | 2.8231 | 3.38e-05 |
| 375,000 | 0.9356 | 2.8368 | 3.26e-05 |
| 380,000 | 1.2624 | 2.8418 | 3.17e-05 |
| 385,000 | 0.8921 | 2.8558 | 3.09e-05 |
| 390,000 | 0.7842 | 2.8624 | 3.04e-05 |
| 395,000 | 1.4256 | 2.8763 | 3.01e-05 |
| 400,000 | 1.2559 | 2.8853 | 3.00e-05 |
