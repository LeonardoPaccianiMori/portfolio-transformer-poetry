# Minerva 7B Historical FP16 LoRA Result

Run completed: 2026-08-09.

## Outcome

Stage A completed 4,000 of at most 6,762 updates and stopped through the
predeclared patience rule. It produced qualifying historical adapters, so the
V6 sonnet-specialization stage may proceed.

| Measurement | Stage zero | Selected step 4,000 |
|---|---:|---:|
| Historical validation loss | 3.262937 | 3.187692 |
| Modern-Italian loss | 2.174054 | 2.135842 |
| Instruction-response loss | 1.552938 | 1.671779 |

The historical loss improved by `0.075245`. Modern-Italian loss improved.
Instruction-response loss increased by 7.65 percent, within the declared
10-percent ceiling. Both preservation gates therefore pass.

## Selected Parent

- Update: `4000`
- Epoch: `2`
- Adapter file: `checkpoints/adapter_step_004000.pt`
- SHA-256: `acfad4d442ac8ea7349dcb1bd379c9b41859027ab45daac54c6b6aa35e0bbc63`
- Base revision: `d1fc0f0e589ae879c5ac763e0e4206a4d14a3f6d`
- Weight loading: unquantized FP16
- Trainable parameters: rank-8 attention LoRA adapters only

The run's original `result.json` named step 3,000 because its implementation
required another `0.005` improvement before replacing the stored best row.
That threshold belongs to patience decisions, while the protocol separately
requires selection of the absolute lowest qualifying historical loss. Steps
3,381 and 4,000 both passed every gate, and step 4,000 had the lowest loss.
Selection was therefore made independently from the complete retained history.
The implementation now performs this final protocol-based selection explicitly.

The complete 366 MiB run directory was copied from the rented instance. Local
and remote SHA-256 values match for both the selected adapter and `result.json`.
