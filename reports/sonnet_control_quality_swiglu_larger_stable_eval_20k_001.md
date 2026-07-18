# Fine-Tuning Run: sonnet_control_quality_swiglu_larger_stable_eval_20k_001

This report records sonnet specialization from the broader-Italian pretraining checkpoint. Checkpoints remain local-only; this report preserves the configuration, validation evidence, and selection rule.

## Configuration

| Setting | Value |
| --- | --- |
| Parent checkpoint step | 110,000 |
| Parent vocabulary | 8,000 |
| Fine-tuning vocabulary | 8,003 |
| Added vocabulary entries | 3 (literal strings not recorded) |
| Context length | 512 |
| Batch size | 2 |
| Learning rate | 3.0e-05 |
| Train steps | 3,000 |
| Evaluation | every 250 steps; all 43 fixed sequential windows |

## Checkpoint Selection

| Measurement | Step | Validation loss |
| --- | ---: | ---: |
| Best recorded validation | 1,000 | 2.9771 |
| Selected saved checkpoint | 1,000 | 2.9771 |

The selected saved checkpoint is the exact checkpoint from the best recorded validation step. It is selected for evaluation instead of the final checkpoint because validation worsened after the best step.

## Overfitting Evidence

| Measurement | Step | Training loss | Validation loss |
| --- | ---: | ---: | ---: |
| First recorded evaluation | 1 | 3.9096 | 3.8528 |
| Final evaluation | 3,000 | 1.6186 | 3.1615 |

## Full Loss History

| Step | Training loss | Validation loss |
| ---: | ---: | ---: |
| 1 | 3.9096 | 3.8528 |
| 250 | 3.2018 | 3.1434 |
| 500 | 2.7366 | 3.0453 |
| 750 | 2.9815 | 3.0013 |
| 1,000 | 2.5758 | 2.9771 |
| 1,250 | 2.6042 | 2.9702 |
| 1,500 | 2.4681 | 2.9751 |
| 1,750 | 2.3867 | 2.9902 |
| 2,000 | 2.0673 | 3.0154 |
| 2,250 | 2.3460 | 3.0334 |
| 2,500 | 1.8557 | 3.0723 |
| 2,750 | 1.8266 | 3.1089 |
| 3,000 | 1.6186 | 3.1615 |
