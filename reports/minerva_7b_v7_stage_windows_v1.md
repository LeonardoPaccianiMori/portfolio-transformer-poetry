# Minerva 7B V7 Deterministic Stage Windows

Checkpoint 8D materializes local deterministic training, validation, and test
window indexes over the independently reproduced checkpoint-8C token pools.
The individual indexes and token IDs remain local; this report publishes only
the frozen policy, hashes, counts, and aggregate cap evidence.

## Training curriculum

| Stage | Windows | Target tokens |
| --- | ---: | ---: |
| stage_1_historical_general | 33,040 | 67,665,920 |
| stage_2_non_sonnet_poetry | 12,160 | 24,903,680 |
| stage_3_sonnets | 2,160 | 4,423,680 |
| **Total** | **47,360** | **96,993,280** |

Every source span contains 2,049 tokens and advances by 2,048 target
tokens. Documents are concatenated only within one role/split pool using
their encoded EOS separators. Cross-document windows retain exact
per-document target contributions, and cross-shard windows retain every
physical shard slice.

## Concentration evidence

| Stage / component | Group | Maximum | Ceiling | Pass |
| --- | --- | ---: | ---: | --- |
| stage_1_historical_general / historical_general | author_key | 9.0097% | 20.00% | yes |
| stage_1_historical_general / historical_general | work_key | 5.2902% | 15.00% | yes |
| stage_1_historical_general / nineteenth_century_bridge | author_key | 7.3843% | 20.00% | yes |
| stage_1_historical_general / nineteenth_century_bridge | work_key | 1.6041% | 15.00% | yes |
| stage_2_non_sonnet_poetry / historical_non_sonnet_poetry | author_key | 7.2345% | 20.00% | yes |
| stage_2_non_sonnet_poetry / historical_non_sonnet_poetry | work_key | 2.8566% | 15.00% | yes |
| stage_2_non_sonnet_poetry / stage_1_historical_replay | author_key | 4.9313% | 20.00% | yes |
| stage_2_non_sonnet_poetry / stage_1_historical_replay | work_key | 1.7270% | 15.00% | yes |
| stage_3_sonnets / standard_sonnets_v7_train | author_key | 4.6176% | 5.00% | yes |
| stage_3_sonnets / standard_sonnets_v7_train | epoch_key | 29.9996% | 30.00% | yes |
| stage_3_sonnets / stage_2_historical_replay | author_key | 4.6296% | 20.00% | yes |
| stage_3_sonnets / stage_2_historical_replay | work_key | 1.5432% | 15.00% | yes |

## Fixed held-out windows

| Split | Windows | Target tokens |
| --- | ---: | ---: |
| validation | 959 | 1,964,032 |
| test | 106 | 217,088 |

Held-out targets are sequential and non-overlapping. Their final incomplete
tails are dropped without padding. V7 validation/test and broader validation
pools never enter a training index.

## Reproduction and boundaries

- Window-index content identity: `e821e3afdc3bd7aa6874180509ba756f942e651980f6455469722c13f8f7424c`.
- Independent reproduction matches: `true`.
- Conditioned and protected V6 material included: `false`.
- GPU work started: `false`.
- Local caches deleted: `false`.
