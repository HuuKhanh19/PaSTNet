"""Dataset adapter for immutable, cached molecular geometry."""

import torch
from torch.utils.data import Dataset

from pastnet.geometry.features import hrge_from_cache


class FeatureDataset(Dataset):
    def __init__(self, records, cache):
        self.records = tuple(records)
        self.cache = cache
        self.features = {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        if record.mol_id not in self.features:
            entry = self.cache.load(record.mol_id, expected_smiles=record.canonical_smiles)
            self.features[record.mol_id] = hrge_from_cache(entry, record.mol_id)
        return torch.tensor(record.target, dtype=torch.float32), self.features[record.mol_id]


def identity(items):
    return items


def smoke_records(records, classification=False, count=8):
    """Small debug subset; include both classes when they are available."""
    selected = list(records[:count])
    if classification:
        for label in (0.0, 1.0):
            if not any(r.target == label for r in selected):
                match = next((r for r in records if r.target == label), None)
                if match is not None:
                    selected[-1] = match
    return tuple(selected)
