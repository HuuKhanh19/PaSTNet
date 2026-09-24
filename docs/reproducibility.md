# Reproducibility

## Reference scope

The release freezes the current E0 PaSTNet architecture: a 128-channel SchNet
atom encoder, 32-channel path states, 16-channel relational states, three
separately parameterized Geo-IH/PaST stages, two tensor layers per PaST block,
and a spatial readout bypass. The model has 667,548 trainable parameters.

ESOL, FreeSolv, BACE, and Lipo recipes were extracted from the existing E0
training configurations. BBBP and single-endpoint ClinTox extend the same
binary-classification recipe used for BACE. The ClinTox endpoint is `CT_TOX`;
this release does not implement the two-endpoint ClinTox benchmark.

The inputs are the six refined CSV files listed in the README. Their exact
SHA-256 fingerprints and the ordered SMILES/target fingerprints for every
partition are in `pastnet/reference/data.json`.
The latter normalize numeric formatting so `0` and `0.0` describe the same label.
The release does not include the raw inputs. Changing their contents defines
a different input set and is rejected by the reference preparation command.

The upstream splitter's whole-scaffold allocation gives approximately
81/9/10 train/validation/test proportions. Empty-scaffold molecules form one
group; in the reference ESOL partitions all 314 are in training. Five Lipo
and thirteen ClinTox molecules with failed conformer generation are excluded
after the original split. This preserves retained row order and scaffold
membership. Unexpected conformer failures stop preparation for inspection.

## Release checks

The following checks were performed on 2026-09-23. Machine-readable outcomes
are in `docs/verification.json`.

| Check | Outcome |
|---|---|
| Raw preprocessing and pinned splitting | All 54 train/validation/test partitions across six datasets and three seeds match the reference ordered SMILES and targets |
| Post-conformer filtering | The five Lipo and thirteen ClinTox exclusions match the fixed manifests |
| Model extraction | All state tensors and the Torch RNG state match the research model after initialization at seed 42 |
| CPU numerical parity | Feature tensors, predictions, and parameter gradients match the research model exactly on the checked mixed-conformer batch |
| Conformer generation | Three removed FreeSolv cache entries regenerate with identical coordinates and metadata using two workers |
| Real-data smoke training | All six datasets complete two CPU epochs at split seed 0, checkpoint selection, and one test evaluation on small debug subsets |
| Checkpoint inference | All nine available E0 checkpoints for ESOL, FreeSolv, and BACE load strictly into the extracted model and evaluate successfully |
| Independent installation | A fresh Python 3.10 environment installs the pinned dependencies and runs without DGL |

The automated tests cover mixed-conformer batching, conformer permutation,
proper rigid motions, finite gradients, deterministic conformer generation,
scaffold separation, canonical atom-order calibration, regression and
classification training, test-evaluation guards, sample-standard-deviation
aggregation, and interrupted-epoch recovery. On CPU, resumed training produces
exactly the same final weights and test predictions as uninterrupted training.
CI runs these tests on Linux and Windows without benchmark data.

## Numerical limits

The three BACE checkpoint replays reproduce the stored ROC-AUC exactly.
The maximum absolute test-RMSE differences from stored runs are approximately
`0.00035` for ESOL and `0.00510` for FreeSolv. Both use BF16 inference.
Repeated evaluation of the original research implementation on the same CUDA
device also exhibited varying predictions; a fixed random seed does not make
these CUDA reductions bitwise deterministic. The full per-seed measurements
are retained in `verification.json` rather than substituted into paper tables.

Validation metrics saved by the release are the measurements that selected
the checkpoint. The test set is evaluated only after loading that checkpoint.
Checkpoint resume restores Python, NumPy, Torch, CUDA, and loader RNG states
at the last completed epoch. GPU resume remains subject to the numerical
limits above.

**The release checks did not retrain all 18 full E0 runs.** The supplied commands
produce those runs from raw inputs; smoke checks and checkpoint replay are
functional and compatibility validation, not a new set of paper results.
