# Table 2: controlled conformer-interaction ablations

The Table 2 entry point uses the shared PaSTNet data pipeline and trainer, with
five matched random-scaffold partitions (seeds 0, 1, 2, 3, 4), model seed 42,
and the same cached ensembles of at most five conformers. Each variant is
trained independently from initialization. No pretrained weights or experiment
outputs are required or distributed.

| CLI variant | Table 2 operator | Parameters |
|---|---|---:|
| `no_past` | Identity (no PaST) | 410,892 |
| `same_path_attention` | Same-path self-attention | 667,710 |
| `centered_attention` | Singleton-centered attention | 667,710 |
| `no_pair_context` | Pair-local PaST | 618,396 |
| `pair_local_matched` | Capacity-matched pair-local PaST | 667,548 |
| `no_singleton_subtraction` | PaST without singleton subtraction | 667,548 |
| `ign15` | Full 15-map equivariant tensor | 666,972 |
| `full_past` | Full PaST | 667,548 |

## Run

Install the package as described in the main README and place the refined ESOL,
FreeSolv and BACE CSV inputs in `data/raw/`. The raw-input fingerprints and ordered
split fingerprints for all five seeds are checked during preparation.

```bash
python -m pastnet.table2 prepare --dataset all --workers 4
python -m pastnet.table2 train --dataset all --variant all --device cuda
python -m pastnet.table2 summarize
```

Or run preparation, training and aggregation together:

```bash
python -m pastnet.table2 run --dataset all --variant all --workers 4 --device cuda --resume
```

This schedules **120 sequential runs** (8 operators x 3 datasets x 5 splits).
To select one operator and dataset:

```bash
python -m pastnet.table2 train --dataset esol --variant centered_attention --seeds 0 1 2 3 4 --device cuda
```

All operators use Adam at learning rate 0.001, no weight decay or scheduler,
logical batch size 16, gradient clipping at 5, at most 300 epochs, and patience
40. ESOL and FreeSolv use train-only target standardization and MSE, with BF16
training and evaluation. BACE uses unweighted BCE on raw logits, BF16 training,
FP32 evaluation, and microbatches of four with sample-weighted accumulation.
The training partitions alone determine the relative bond scale. Checkpoint
selection uses minimum validation RMSE or maximum validation ROC-AUC; test
evaluation occurs once after selection.

The original initialization of retained modules and the post-construction RNG
state are preserved across controls. The pair-local variant retains the final
pair-to-conformer contraction; it removes contextual maps inside the tensor
layers. Capacity matching widens the existing pair-local FFNs only. The centered
attention and 15-map controls evaluate their complete shared block on both the
ensemble and singleton inputs. Removing singleton subtraction changes only the
PaST cross-conformer message.

## Outputs and checks

Runs are saved under
`results/table2/<variant>/<dataset>/seed_<seed>/`. Use `--results-dir` to change
the root, `--data-dir` for prepared data, and `--resume` for checkpoint recovery.
Run identities include the variant and recipe, preventing results from different
operators from being mixed. The E0 commands retain their existing output paths.

`results/table2/summary.csv` and `summary.json` report means and sample standard
deviations across the requested seeds. BACE ROC-AUC is converted to **percent** in
these summaries; individual run metrics remain on the 0-1 scale. Missing,
mismatched and smoke-test results are rejected. Selecting fewer than five seeds
produces a partial summary with the actual number of seeds recorded.

After preparation, a functional CPU check can be run with:

```bash
python -m pastnet.table2 train --dataset esol --variant all --seeds 0 --device cpu --smoke
python -m pytest -q
```

Smoke runs use two FP32 epochs on up to eight molecules per partition and are
stored in a separate `smoke/` subdirectory. Tests cover operator capacities,
retained initialization, permutation and masking behavior, singleton behavior,
gradients, and regression/classification training with checkpoint recovery.
These checks do not constitute full benchmark retraining or numerical results
for the paper.
