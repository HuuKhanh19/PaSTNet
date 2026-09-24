# PaSTNet

Implementation of **Pathwise Relational Conformer Tensors for Molecular Ensemble Learning**.

PaSTNet learns across aligned atoms, bonds, angles, and torsions in molecular conformer ensembles. It combines a SchNet spatial encoder with three geometry-conditioned path propagation and Pathwise State Tensor (PaST) stages, followed by learned pathwise conformer pooling. This repository contains the complete PaSTNet model and the E0 training pipeline for six molecular property benchmarks.

## Installation

Use Python 3.10 and a CUDA GPU with BF16 support for the E0 experiments. Dependencies are pinned in `pyproject.toml`. From the root of the downloaded repository, run:

```bash
conda create -n pastnet python=3.10 -y
conda activate pastnet
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
python -m pip install -e ".[test]"
```

The reference environment uses PyTorch 2.8.0, PyG 2.7.0, RDKit 2026.3.2, and NumPy 2.2.6. CUDA wheel alternatives are listed in the [PyTorch installation archive](https://pytorch.org/get-started/previous-versions/#v280). For CPU tests, replace `cu129` with `cpu`. DGL and optional PyG extension packages are not required.

## Data

Place the six **refined CSV inputs** in `data/raw/`. Use the filenames and columns below. These are the inputs used by the reference experiments; an unprocessed MoleculeNet download is not interchangeable with them. Raw data, generated conformers, and model checkpoints are excluded from Git.

| Dataset | Filename | SMILES column | Target column | Retained molecules | Metric |
|---|---|---|---|---:|---|
| ESOL | `refined_ESOL.csv` | `smiles` | `measured` | 1,115 | RMSE ↓ |
| FreeSolv | `refined_FreeSolv.csv` | `smiles` | `measured` | 635 | RMSE ↓ |
| Lipophilicity | `refined_Lipophilicity.csv` | `smiles` | `measured` | 4,093 | RMSE ↓ |
| BACE | `refined_BACE.csv` | `SMILES` | `class` | 1,454 | ROC-AUC ↑ |
| BBBP | `refined_BBBP.csv` | `SMILES` | `class` | 1,753 | ROC-AUC ↑ |
| ClinTox | `refined_ClinTox.csv` | `smiles` | `CT_TOX` | 1,336 | ROC-AUC ↑ |

ClinTox uses the **single CT_TOX endpoint**. The preparation step validates raw-file fingerprints and all ordered split memberships against `pastnet/reference/data.json`.

## Reproduce E0

One command prepares all data, trains the three split seeds for each dataset, and summarizes the results:

```bash
python -m pastnet run --dataset all --workers 4 --device cuda
```

Alternatively, run the stages separately:

```bash
python -m pastnet prepare --dataset all --workers 4
python -m pastnet train --dataset esol --seeds 0 1 2 --device cuda
python -m pastnet summarize --dataset esol
```

Dataset identifiers are `esol`, `freesolv`, `lipo`, `bace`, `bbbp`, and `clintox`. Use `--data-dir` and `--results-dir` to change artifact locations. Each training process uses one GPU; `--device cuda:1` selects a second visible GPU. `--workers` controls CPU conformer generation only.

Preparation reuses validated cache entries. To resume training from the last completed epoch and skip completed runs:

```bash
python -m pastnet run --dataset all --workers 4 --device cuda --resume
```

## Experimental protocol

- **Splits:** the random scaffold splitter is included in `pastnet/_vendor/schnet_gp_splitter.py`. It groups chirality-aware Bemis–Murcko scaffolds, allocates test groups first, then validation groups. The requested fractions are 10% test and 10% of the remainder for validation, approximately **81/9/10**, subject to whole-scaffold allocation. Split seeds are 0, 1, and 2; initialization and training-loader seed are fixed at 42.
- **Conformers:** ETKDGv3 generates up to 20 candidates. MMFF94s optimization, with UFF fallback, is followed by lowest-energy initialization and greedy max–min heavy-atom RMSD selection at 0.5 Å, retaining up to five conformers. Explicit hydrogens and atom correspondence are preserved. Five fixed Lipo failures and thirteen fixed ClinTox failures are removed **after splitting**, without reshuffling. IDs and reasons are saved in the preparation manifests.
- **Training:** Adam, learning rate `1e-3`, no weight decay or scheduler, batch size 16, gradient clipping at 5, at most 300 epochs, and patience 40. Regression uses training-only target z-scores and MSE; classification uses unweighted binary cross-entropy on logits. Validation RMSE or ROC-AUC selects the checkpoint; the selected model is evaluated on test once.
- **Precision:** training uses BF16. ESOL and FreeSolv also evaluate with BF16; Lipo and classification tasks evaluate in FP32. Lipo and classification tasks accumulate gradients over microbatches of four molecules. The full model has **667,548 parameters**. Dataset recipes are in `pastnet/configs/`.

## Outputs and verification

Each `results/<dataset>/seed_<seed>/` contains the resolved configuration, training history, `best.pt`, resumable `last.pt`, test predictions, and metrics. `summary.csv` and `summary.json` report the unweighted mean and **sample standard deviation** across seeds. ROC-AUC is stored on the 0–1 scale; multiply by 100 for percentage reporting.

Run the tests without benchmark data:

```bash
python -m pytest -q
```

After data preparation, check a small training run:

```bash
python -m pastnet train --dataset esol --seeds 0 --device cpu --smoke
```

Smoke runs use two FP32 epochs on up to eight rows per split, write to a separate `smoke/` directory, and are excluded from benchmark summaries. Numerical results can vary across GPU architectures and software builds; the pinned recipes and data fingerprints define the reference protocol. Release checks and their scope are recorded in `docs/reproducibility.md`.

## Acknowledgments

PaSTNet builds on SchNet-GP and the Path Complex Neural Network atom/bond encodings. Source attribution and adaptation details are in `THIRD_PARTY_NOTICES.md`. The code is distributed under the MIT license in `LICENSE`.
