"""Strict readers for prepared single-target molecular splits."""

import csv
import hashlib
import math
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

from rdkit import Chem
from torch.utils.data import Dataset


DEFAULT_DATA_ROOT = (
    Path("data/processed/esol/splits")
)
# Supplied train/valid/test CSVs use these column names.
DEFAULT_SMILES_COL = "smiles"
DEFAULT_TARGET_COL = "target"


@dataclass(frozen=True)
class MoleculeConfig:
    """data_root contains the supplied seed_0 through seed_2 directories."""

    data_root: Path = DEFAULT_DATA_ROOT
    split_seed: int = 0
    smiles_col: str = DEFAULT_SMILES_COL
    target_col: str = DEFAULT_TARGET_COL
    include_test: bool = True


@dataclass(frozen=True)
class MoleculeRecord:
    smiles: str
    canonical_smiles: str
    mol_id: str
    target: float
    row_number: int
    mol: Chem.Mol = field(repr=False, compare=False)


class MoleculeDataset(Dataset):
    """One CSV split, eagerly validated, with no filtering or deduplication.

    Use load_splits to also enforce disjointness across train/val/test.
    Each item contains the original SMILES, canonical isomeric SMILES, a stable
    ID, an unnormalized float target, and a parsed RDKit Mol without generated
    coordinates. row_number is the source CSV line number (header is line 1).
    """

    def __init__(
        self,
        csv_path,
        *,
        smiles_col=DEFAULT_SMILES_COL,
        target_col=DEFAULT_TARGET_COL,
    ):
        self.csv_path = Path(csv_path)
        self.smiles_col = smiles_col
        self.target_col = target_col
        records = []
        parser_params = Chem.SmilesParserParams()
        parser_params.parseName = False

        with self.csv_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            headers = reader.fieldnames or []
            missing = {smiles_col, target_col} - set(headers)
            if missing:
                raise ValueError(
                    f"{self.csv_path}: missing required columns {sorted(missing)}; "
                    f"available columns: {headers}"
                )
            if len(headers) != len(set(headers)):
                raise ValueError(f"{self.csv_path}: duplicate CSV column names")

            try:
                for row in reader:
                    location = f"{self.csv_path}: row {reader.line_num}"
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError(f"{location}: row does not match CSV header")

                    raw_target = row[target_col].strip()
                    if not raw_target:
                        raise ValueError(f"{location}: missing target in {target_col!r}")
                    try:
                        target = float(raw_target)
                    except ValueError as exc:
                        raise ValueError(
                            f"{location}: invalid target {raw_target!r} in {target_col!r}"
                        ) from exc
                    if not math.isfinite(target):
                        raise ValueError(
                            f"{location}: non-finite target {raw_target!r} in {target_col!r}"
                        )

                    smiles = row[smiles_col].strip()
                    if not smiles:
                        raise ValueError(f"{location}: missing SMILES in {smiles_col!r}")
                    mol = Chem.MolFromSmiles(smiles, parser_params)
                    if mol is None or mol.GetNumAtoms() == 0:
                        raise ValueError(f"{location}: invalid SMILES {smiles!r}")
                    canonical = Chem.MolToSmiles(
                        mol, canonical=True, isomericSmiles=True
                    )
                    mol_id = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]
                    records.append(
                        MoleculeRecord(
                            smiles=smiles,
                            canonical_smiles=canonical,
                            mol_id=mol_id,
                            target=target,
                            row_number=reader.line_num,
                            mol=mol,
                        )
                    )
            except csv.Error as exc:
                raise ValueError(
                    f"{self.csv_path}: row {reader.line_num}: malformed CSV: {exc}"
                ) from exc

        if not records:
            raise ValueError(f"{self.csv_path}: empty Molecule split")
        self.records = tuple(records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


@dataclass(frozen=True)
class MoleculeSplits:
    """The supplied valid.csv is exposed as val; no split is recomputed."""

    train: MoleculeDataset
    val: MoleculeDataset
    test: MoleculeDataset = None

    def validate_disjoint(self):
        """Raise with source locations if canonical SMILES or mol_ids overlap."""
        splits = tuple((name, split) for name, split in
                       (("train", self.train), ("val", self.val), ("test", self.test))
                       if split is not None)
        for (left_name, left), (right_name, right) in combinations(splits, 2):
            for attribute in ("canonical_smiles", "mol_id"):
                left_records = {getattr(item, attribute): item for item in left}
                right_records = {getattr(item, attribute): item for item in right}
                overlap = left_records.keys() & right_records.keys()
                if overlap:
                    example = sorted(overlap)[0]
                    raise ValueError(
                        f"Molecule splits {left_name}/{right_name} overlap by {attribute}: "
                        f"{len(overlap)} shared value(s), e.g. {example!r}; "
                        f"{left.csv_path}: row {left_records[example].row_number} and "
                        f"{right.csv_path}: row {right_records[example].row_number}"
                    )


def load_splits(config=None):
    """Load and validate existing CSVs for one seed using an MoleculeConfig.

    Example: load_splits(MoleculeConfig(split_seed=0, target_col="target")).
    No scaffold splitting, shuffling, target normalization, or disk writes occur.
    """
    config = config if config is not None else MoleculeConfig()
    if config.split_seed not in (0, 1, 2):
        raise ValueError("Molecule split_seed must be one of 0, 1, 2")
    directory = Path(config.data_root) / f"seed_{config.split_seed}"
    names = ("train", "valid", "test") if config.include_test else ("train", "valid")
    paths = [directory / f"{split}.csv" for split in names]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required pre-split Molecule CSV does not exist: {path}")
    splits = MoleculeSplits(
        *[
            MoleculeDataset(
                path, smiles_col=config.smiles_col, target_col=config.target_col
            )
            for path in paths
        ]
    )
    splits.validate_disjoint()
    return splits
