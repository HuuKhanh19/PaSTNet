from dataclasses import replace

import numpy as np
import torch

from pastnet.data.cache import ConformerCache
from pastnet.data.conformers import generate_conformers
from pastnet.data.molecule import molecule_from_smiles
from pastnet.geometry.features import build_hrge_features, hrge_from_cache
from pastnet.model import PaSTNet


def test_batch_independence_and_backward(tiny_data):
    cache = ConformerCache(tiny_data / "esol" / "conformers_etkdg_mmff")
    ids = sorted(cache.index)[:3]
    inputs = [hrge_from_cache(cache.load(key), key) for key in ids]
    torch.manual_seed(42)
    model = PaSTNet(bond_scale=.01)
    assert sum(p.numel() for p in model.parameters()) == 667548
    packed = model(inputs)
    individual = torch.stack([model(m) for m in inputs])
    torch.testing.assert_close(packed, individual, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(model(list(reversed(inputs))).flip(0), packed, atol=2e-6, rtol=2e-5)
    packed.square().sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum().item() for g in grads) > 0


def test_conformer_permutation_and_rigid_motion():
    smiles = "CCCCO"
    coords, _ = generate_conformers(smiles)
    # An ensemble with distinct conformers, generated without benchmark data.
    assert len(coords) > 1
    mol = molecule_from_smiles(smiles)
    original = build_hrge_features(mol, torch.tensor(coords, dtype=torch.float64))
    permutation = build_hrge_features(mol, torch.tensor(coords[::-1].copy(), dtype=torch.float64))
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    moved = build_hrge_features(mol, torch.tensor(coords @ rotation + 3.0, dtype=torch.float64))
    torch.manual_seed(42)
    model = PaSTNet(bond_scale=.01).eval()
    with torch.no_grad():
        expected = model(original)
        torch.testing.assert_close(model(permutation), expected, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(model(moved), expected, atol=2e-6, rtol=2e-5)


def test_cache_generation_is_deterministic(tiny_data):
    cache = ConformerCache(tiny_data / "esol" / "conformers_etkdg_mmff")
    entry = next(cache.load(k) for k, v in cache.index.items() if v["canonical_smiles"] == "CCCC")
    coords, metadata = generate_conformers(entry.graph_smiles)
    np.testing.assert_array_equal(coords, entry.coords)
    assert metadata["selected_conformer_ids"] == entry.metadata["selected_conformer_ids"]
