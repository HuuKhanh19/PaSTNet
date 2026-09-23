"""Preprocessing functions extracted unchanged from pinned SchNet-GP."""

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
    """Tên cột nhãn trong DataFrame đã tiền xử lý.

    - 1 task  -> ['target']            (giữ nguyên convention cũ, cache tương thích)
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
    """Clean DataFrame: validate SMILES, remove duplicates, chuẩn hoá cột nhãn.

    Nhãn được đổi tên thành 'target' (single-task) hoặc 'target_0..' (multi-label).
    - single-task (regression/classification): bỏ hàng thiếu nhãn.
    - multi-label: GIỮ ô trống (= nhãn thiếu, xử lý bằng mask ở loss); chỉ bỏ hàng
      không có nhãn nào.
    """
    df = df.copy()
    initial = len(df)
    n_tasks = len(label_columns)

    # Ép nhãn về số (ô trống -> NaN).
    for c in label_columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    if task_type == 'multilabel':
        # Bỏ hàng thiếu SMILES hoặc không có bất kỳ nhãn nào.
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

    # Nhãn classification giữ dạng float (0.0/1.0, NaN cho thiếu) để dùng
    # BCEWithLogits + mask; regression cũng float.
    for c in out_cols:
        df[c] = df[c].astype(float)

    return df.reset_index(drop=True)
