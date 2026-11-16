"""SchNet-GP preprocessing with English comments and docstrings."""

from typing import List
import pandas as pd

def validate_smiles(smiles: str) -> bool:
    """Check if a SMILES string is valid using RDKit."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False
    finally:
        RDLogger.EnableLog('rdApp.*')

def target_column_names(n_tasks: int) -> List[str]:
    """Return label column names for the preprocessed DataFrame.

    - 1 task  -> ['target']            (preserves legacy naming and cache compatibility)
    - >1 task -> ['target_0', ...]     (multi-label)
    """
    if n_tasks == 1:
        return ['target']
    return [f'target_{i}' for i in range(n_tasks)]

def preprocess_dataframe(
    df: pd.DataFrame,
    smiles_column: str,
    label_columns: List[str],
    task_type: str,
) -> pd.DataFrame:
    """Clean a DataFrame: validate SMILES, remove duplicates, standardize label columns.

    Labels are renamed to 'target' (single-task) or 'target_0', etc. (multi-label).
    - single-task (regression/classification): drop rows with missing labels.
    - multi-label: keep missing labels as NaN for masking in the loss; drop only
      rows with no available labels.
    """
    df = df.copy()
    initial = len(df)
    n_tasks = len(label_columns)

    # Convert labels to numeric values (empty cells become NaN).
    for c in label_columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    if task_type == 'multilabel':
        # Drop rows with missing SMILES or no available labels.
        df = df.dropna(subset=[smiles_column])
        df = df[df[label_columns].notna().any(axis=1)]
    else:
        df = df.dropna(subset=[smiles_column] + label_columns)
    dropped = initial - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows with NaN")

    valid_mask = df[smiles_column].apply(validate_smiles)
    invalid = (~valid_mask).sum()
    if invalid:
        df = df[valid_mask]
        print(f"  Removed {invalid} invalid SMILES")

    dups = df.duplicated(subset=[smiles_column], keep='first').sum()
    if dups:
        df = df.drop_duplicates(subset=[smiles_column], keep='first')
        print(f"  Removed {dups} duplicate SMILES")

    out_cols = target_column_names(n_tasks)
    df = df[[smiles_column] + label_columns].copy()
    df.columns = ['smiles'] + out_cols

    # Keep classification labels as floats (0.0/1.0, NaN for missing values)
    # for masked BCEWithLogits; regression targets are floats as well.
    for c in out_cols:
        df[c] = df[c].astype(float)

    return df.reset_index(drop=True)
