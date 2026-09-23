import hashlib
from importlib.resources import files

import pandas as pd
import pytest

from pastnet._vendor.schnet_gp_splitter import generate_scaffold, random_scaffold_split
from pastnet.config import DATASETS, load_config, reference_data
from pastnet.data.dataset import MoleculeConfig, load_splits
from pastnet.data.prepare import split_raw


def test_splitter_matches_pinned_source():
    source = files("pastnet").joinpath("_vendor/schnet_gp_splitter.py").read_bytes()
    assert hashlib.sha256(source).hexdigest() == "5297c60a3343c95b6cac98d3bfbc3c36965c5499472d410d0f63d5efb61335af"


def test_scaffold_disjointness_and_reproducibility():
    smiles = ["C", "CC", "CCC", "c1ccccc1", "Cc1ccccc1", "c1ccncc1", "C1CCCCC1", "C1CCCC1", "c1ccoc1", "c1ccsc1"]
    frame = pd.DataFrame(dict(smiles=smiles, target=range(len(smiles))))
    a = random_scaffold_split(frame, frame.smiles, random_seed=0, ratio_test=.3, ration_valid=.3, dataframe=True)
    b = random_scaffold_split(frame, frame.smiles, random_seed=0, ratio_test=.3, ration_valid=.3, dataframe=True)
    for left, right in zip(a, b):
        pd.testing.assert_frame_equal(left, right)
    scaffolds = [{generate_scaffold(s, True) for s in part.smiles} for part in a]
    assert all(not scaffolds[i] & scaffolds[j] for i, j in ((0, 1), (0, 2), (1, 2)))
    assert sorted(i for part in a for i in part.index) == list(range(len(frame)))


def test_six_recipes_and_clintox_endpoint():
    ref = reference_data()
    assert set(ref) == set(DATASETS)
    assert load_config("clintox")["target_column"] == "CT_TOX"
    assert len(ref["clintox"]["excluded_conformer_ids"]) == 13
    assert len(ref["lipo"]["excluded_conformer_ids"]) == 5
    for name in DATASETS:
        cfg = load_config(name)
        assert cfg["model_seed"] == 42 and cfg["split_seeds"] == [0, 1, 2]


def test_modified_raw_data_fails_before_splitting(tmp_path):
    (tmp_path / "refined_ESOL.csv").write_text("smiles,measured\nC,1\n")
    with pytest.raises(ValueError, match="fingerprint"):
        split_raw("esol", tmp_path, tmp_path / "processed")


def test_canonical_overlap_rejected(tiny_data, tmp_path):
    directory = tmp_path / "seed_0"
    directory.mkdir()
    for part in ("train", "valid", "test"):
        (directory / f"{part}.csv").write_text("smiles,target\nCCO,1\n")
    with pytest.raises(ValueError, match="overlap"):
        load_splits(MoleculeConfig(data_root=tmp_path))


def test_bond_calibration_uses_cached_canonical_atom_order(tiny_data):
    from dataclasses import replace
    from types import SimpleNamespace
    from pastnet.data.cache import ConformerCache
    from pastnet.geometry.relative import fit_training_bond_scale
    splits = load_splits(MoleculeConfig(data_root=tiny_data / "esol/splits"))
    ethanol = next(r for r in splits.train if r.canonical_smiles == "CCO")
    alternative = replace(ethanol, smiles="OCC")
    cache = ConformerCache(tiny_data / "esol/conformers_etkdg_mmff")
    first = fit_training_bond_scale(SimpleNamespace(train=[ethanol]), cache)
    second = fit_training_bond_scale(SimpleNamespace(train=[alternative]), cache)
    assert first.value == second.value
