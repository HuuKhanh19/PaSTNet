import pandas as pd
import pytest
import torch

from pastnet.data.cache import initialize_cache
from pastnet.data.conformers import ConformerConfig, generate_conformers
from pastnet.data.prepare import molecule_identity
from pastnet.io import sha256, write_json


@pytest.fixture(scope="session", autouse=True)
def single_thread():
    torch.set_num_threads(1)


@pytest.fixture(scope="session")
def tiny_data(tmp_path_factory):
    root = tmp_path_factory.mktemp("molecules")
    # Synthetic test chemistry; not a subset of any benchmark.
    splits = dict(train=["C", "CC", "CCC", "CCCC", "CCO", "CCN", "CCF", "CCCl"],
                  valid=["CO", "CN", "CF", "CCl", "COC", "CNC", "CCBr", "CCS"],
                  test=["CBr", "CS", "CC=O", "CC#N", "CCCO", "CCCN", "CCCF", "CCCCl"])
    import shutil
    cache = initialize_cache(root / "esol" / "conformers_etkdg_mmff", ConformerConfig())
    for smiles in sum(splits.values(), []):
        canonical, _ = molecule_identity(smiles)
        coords, metadata = generate_conformers(canonical)
        cache.store(coords, metadata)
    shutil.copytree(cache.directory, root / "bace" / "conformers_etkdg_mmff")
    for dataset in ("esol", "bace"):
        directory = root / dataset / "splits" / "seed_0"
        directory.mkdir(parents=True)
        manifest = dict(cache_fingerprint=cache.fingerprint, cache_index_sha256=sha256(cache.directory / "index.jsonl"),
                        splits={"0": {}})
        for name, smiles in splits.items():
            targets = [float(i % 2) if dataset == "bace" else i/3-1 for i in range(len(smiles))]
            path = directory / f"{name}.csv"
            pd.DataFrame(dict(smiles=smiles, target=targets)).to_csv(path, index=False)
            manifest["splits"]["0"][name] = dict(sha256=sha256(path))
        write_json(directory / "manifest.json", manifest)
    return root
