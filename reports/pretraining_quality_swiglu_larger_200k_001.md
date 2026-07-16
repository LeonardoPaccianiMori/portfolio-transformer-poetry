# Pretraining Run: pretraining_quality_swiglu_larger_200k_001

This report records a from-scratch broader-Italian-corpus pretraining run. Raw corpus files, interval checkpoints, and the final checkpoint are intentionally local-only; the configuration and observed results are preserved here.

## Configuration

| Setting | Value |
| --- | --- |
| Device | cuda:0 |
| Vocabulary size | 8,000 |
| Training tokens | 17,121,479 |
| Validation tokens | 172,961 |
| Context length | 512 |
| Batch size | 2 |
| Completed steps | 200,000 |
| Learning rate | 3.0e-04 |
| Learning-rate schedule | warmup_cosine |
| Warmup steps | 2,000 |
| Minimum learning rate | 3.0e-05 |
| Evaluation | every 5,000 steps; all 337 sequential windows |
| Interval checkpoints | every 25,000 steps |
| Parameters | 33,671,312 |
| Embedding dimension | 512 |
| Transformer layers | 8 |
| Attention heads | 8 |
| Head dimension | 64 |
| Feed-forward dimension | 1365 |
| Normalization | layer_norm |
| Normalization epsilon | 1.0e-05 |
| Position encoding | learned_absolute |
| RoPE theta | 10000 |
| Feed-forward type | swiglu |
| Tied token embeddings | False |

## Loss Summary

| Measurement | Step | Training loss | Validation loss |
| --- | ---: | ---: | ---: |
| First recorded evaluation | 1 | 9.0710 | 9.0621 |
| Best validation evaluation | 110,000 | 2.0768 | 2.5001 |
| Final evaluation | 200,000 | 1.9949 | 2.5769 |

## Saved Local Artifacts

- Interval checkpoints: 8
- Final checkpoint size: 449.7 MiB
- Loss-history records: 41

## Final Sample Excerpt

```text
Nel Santo Gesù Cristo con Santa Maria Madalena e Santa Carità per la chiesa e collocatansi Paulo dalle Marie e Guido Scimuo de' Medici. Et oltre alle tutte fece in un diversi quadri, grande quanto il naturale della detta sepoltura di papa Nicolaio. Et intorno a questo è molto celebre, e nella volta è in un arco una San Rocco in fresco, in modo che si vede che tre lune è avanti che il tramezzo manifestissimamente si storce per non esser veduto: mentre essi cantano col motto dell'arte dell'arte per tale azzione ch'essi adoperano quella parte donde egli passava e dove nel fondo si faceva. Dentro poi dentro è un paese grandissimo per tutto, di che egli si dilettò tanto di servitù, che egli fu chiamato alla Signoria della terra di Prato;
```

## Interpretation

The loss fell substantially from the first recorded evaluation, and the sample has learned historical Italian prose-like texture. It is not sonnet-specialized: that is the intended role of the next fine-tuning stage. Each validation result covers the complete fixed holdout through non-overlapping sequential windows, so the best-validation checkpoint is a deterministic selection within this run. Validation was lower at step 110,000 than at the final step, so downstream generation and fine-tuning should use `best_validation.pt`, not `model.pt`.

## Full Loss History

| Step | Training loss | Validation loss | Learning rate |
| ---: | ---: | ---: | ---: |
| 1 | 9.0710 | 9.0621 | 1.50e-07 |
| 5,000 | 4.1249 | 3.8192 | 3.00e-04 |
| 10,000 | 3.2336 | 3.2765 | 2.99e-04 |
| 15,000 | 2.7479 | 3.0567 | 2.97e-04 |
| 20,000 | 2.6698 | 2.9196 | 2.95e-04 |
| 25,000 | 2.5903 | 2.8539 | 2.91e-04 |
| 30,000 | 2.5940 | 2.7791 | 2.87e-04 |
| 35,000 | 2.2593 | 2.7079 | 2.82e-04 |
| 40,000 | 2.4242 | 2.6742 | 2.76e-04 |
| 45,000 | 2.3524 | 2.6381 | 2.70e-04 |
| 50,000 | 2.0166 | 2.6143 | 2.63e-04 |
| 55,000 | 2.3767 | 2.5822 | 2.55e-04 |
| 60,000 | 2.2621 | 2.5733 | 2.47e-04 |
| 65,000 | 2.1518 | 2.5541 | 2.38e-04 |
| 70,000 | 2.2502 | 2.5505 | 2.29e-04 |
| 75,000 | 2.4865 | 2.5306 | 2.19e-04 |
| 80,000 | 2.2244 | 2.5225 | 2.09e-04 |
| 85,000 | 1.9816 | 2.5175 | 1.99e-04 |
| 90,000 | 2.1748 | 2.5156 | 1.88e-04 |
| 95,000 | 1.9524 | 2.5002 | 1.78e-04 |
| 100,000 | 2.0405 | 2.5035 | 1.67e-04 |
| 105,000 | 1.9220 | 2.5008 | 1.56e-04 |
| 110,000 | 2.0768 | 2.5001 | 1.46e-04 |
| 115,000 | 1.8147 | 2.5009 | 1.35e-04 |
| 120,000 | 1.7767 | 2.5046 | 1.25e-04 |
| 125,000 | 1.8223 | 2.5029 | 1.15e-04 |
| 130,000 | 1.9680 | 2.5043 | 1.05e-04 |
| 135,000 | 1.6424 | 2.5067 | 9.57e-05 |
| 140,000 | 1.5855 | 2.5135 | 8.67e-05 |
| 145,000 | 1.8349 | 2.5154 | 7.82e-05 |
| 150,000 | 1.7826 | 2.5162 | 7.03e-05 |
| 155,000 | 1.7988 | 2.5255 | 6.30e-05 |
| 160,000 | 1.7725 | 2.5325 | 5.63e-05 |
| 165,000 | 1.9543 | 2.5357 | 5.03e-05 |
| 170,000 | 1.4636 | 2.5408 | 4.50e-05 |
| 175,000 | 1.6829 | 2.5511 | 4.05e-05 |
| 180,000 | 1.6950 | 2.5552 | 3.67e-05 |
| 185,000 | 1.8066 | 2.5598 | 3.38e-05 |
| 190,000 | 1.5864 | 2.5653 | 3.17e-05 |
| 195,000 | 1.4148 | 2.5729 | 3.04e-05 |
| 200,000 | 1.9949 | 2.5769 | 3.00e-05 |
