"""Deterministic OFFLINE conformer generation. Never import from a train loop."""

import hashlib
import math
from dataclasses import asdict, dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from pastnet.data.molecule import atom_order_signature, molecule_from_smiles


@dataclass(frozen=True)
class ConformerConfig:
    global_seed: int = 42
    num_candidates: int = 20
    max_conformers: int = 5
    rmsd_threshold: float = 0.5  # Angstrom, aligned heavy atoms, fixed atom IDs
    max_iterations: int = 1000
    mmff_variant: str = "MMFF94s"

    def __post_init__(self):
        for name in ("global_seed", "num_candidates", "max_conformers", "max_iterations"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.global_seed < 0:
            raise ValueError("global_seed must be nonnegative")
        if self.num_candidates < 1 or not 1 <= self.max_conformers <= self.num_candidates:
            raise ValueError("num_candidates must be positive; max_conformers is 1..num_candidates")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if not math.isfinite(self.rmsd_threshold) or self.rmsd_threshold <= 0:
            raise ValueError("rmsd_threshold must be finite and positive")
        if self.mmff_variant not in ("MMFF94s", "MMFF94"):
            raise ValueError("mmff_variant must be MMFF94s or MMFF94")


def molecule_seed(mol_id, global_seed):
    digest = hashlib.sha256(f"{global_seed}:{mol_id}".encode("utf-8")).digest()
    # Positive signed 32-bit seed; never RDKit's special unseeded value -1.
    return int.from_bytes(digest[:8], "big") % 2147483646 + 1


class ConformerGenerationError(ValueError):
    """A molecule with no usable conformers; metadata is persisted by the CLI."""

    def __init__(self, message, metadata):
        super().__init__(message)
        self.metadata = metadata


def select_diverse_conformers(mol, energies, max_conformers, rmsd_threshold):
    """Greedy max-min RMSD selection, seeded by the lowest-energy conformer.

    Alignment compares the SAME heavy atom i to i, without reflections or
    symmetry-based renumbering. GetAlignmentTransform does not mutate positions.
    Energy only orders candidates/ties; it is never a model input or weight.
    """
    ordered = sorted(energies, key=lambda cid: (energies[cid], cid))
    selected = [ordered.pop(0)]
    heavy = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    atom_map = [(i, i) for i in (heavy or list(range(mol.GetNumAtoms())))]
    minimum = {cid: float("inf") for cid in ordered}
    while ordered and len(selected) < max_conformers:
        latest = selected[-1]
        for cid in ordered:
            rmsd, _ = rdMolAlign.GetAlignmentTransform(
                mol, mol, prbCid=cid, refCid=latest, atomMap=atom_map, reflect=False
            )
            minimum[cid] = min(minimum[cid], float(rmsd))
        best = min(ordered, key=lambda cid: (-minimum[cid], energies[cid], cid))
        if minimum[best] < rmsd_threshold:
            break
        selected.append(best)
        ordered.remove(best)
    return selected


def generate_conformers(canonical_smiles, config=None):
    """Return (coords[K,N,3], metadata) using the exact PCNN atom order.

    Original PCNN includes explicit hydrogens, so none are removed. All
    conformers belong to one Mol and share its atom table throughout. This
    function only computes; cache writing is a separate offline operation.
    """
    config = config if config is not None else ConformerConfig()
    parsed = Chem.MolFromSmiles(canonical_smiles)
    if parsed is None or not parsed.GetNumAtoms():
        raise ValueError(f"Invalid SMILES: {canonical_smiles!r}")
    if Chem.MolToSmiles(parsed, canonical=True, isomericSmiles=True) != canonical_smiles:
        raise ValueError("Generation requires canonical isomeric SMILES")
    mol_id = hashlib.sha1(canonical_smiles.encode("utf-8")).hexdigest()[:16]
    seed = molecule_seed(mol_id, config.global_seed)
    mol = molecule_from_smiles(canonical_smiles)
    signature = atom_order_signature(mol)
    for atom in mol.GetAtoms():
        atom.SetIntProp("_pcnn_original_index", atom.GetIdx())
    metadata = {
        "mol_id": mol_id, "canonical_smiles": canonical_smiles,
        "graph_smiles": canonical_smiles, "seed": seed,
        "config": asdict(config), "embedding": "ETKDGv3", "num_threads": 1,
        "hydrogen_policy": "retain_explicit_H_as_in_original_PCNN",
        "atom_order": signature, "num_graph_atoms": mol.GetNumAtoms(),
        "num_heavy_atoms": mol.GetNumHeavyAtoms(), "coordinate_units": "angstrom",
        "energy_units": "kcal/mol", "num_requested": config.num_candidates,
        "num_generated": 0, "num_optimized": 0, "num_retained": 0,
        "optimizer": None, "fallbacks": [], "candidates": [],
        "selected_conformer_ids": [], "energies": [],
        "selection": "lowest_energy_then_max_min_fixed_atom_heavy_RMSD",
    }

    def fail(message):
        metadata.update(status="failed", error=message)
        raise ConformerGenerationError(f"{mol_id}: {message}", metadata)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 1
    params.pruneRmsThresh = -1.0  # Deduplicate AFTER force-field optimization.
    params.enforceChirality = True
    params.clearConfs = True
    try:
        conf_ids = list(AllChem.EmbedMultipleConfs(
            mol, numConfs=config.num_candidates, params=params
        ))
        if not conf_ids:
            metadata["fallbacks"].append("ETKDGv3 returned zero candidates; retry useRandomCoords=True")
            params.useRandomCoords = True
            conf_ids = list(AllChem.EmbedMultipleConfs(
                mol, numConfs=config.num_candidates, params=params
            ))
    except (ValueError, RuntimeError) as exc:
        fail(f"ETKDGv3 failed: {exc}")
    metadata["num_generated"] = len(conf_ids)
    if not conf_ids:
        fail("ETKDGv3 produced no conformers, including deterministic retry")

    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            optimizer = config.mmff_variant
            metadata["optimizer"] = optimizer
            results = AllChem.MMFFOptimizeMoleculeConfs(
                mol, numThreads=1, maxIters=config.max_iterations,
                mmffVariant=optimizer,
            )
        else:
            metadata["fallbacks"].append("MMFF parameters unavailable; use UFF")
            if not AllChem.UFFHasAllMoleculeParams(mol):
                fail("Neither MMFF nor UFF has parameters for every atom")
            optimizer = "UFF"
            metadata["optimizer"] = optimizer
            results = AllChem.UFFOptimizeMoleculeConfs(
                mol, numThreads=1, maxIters=config.max_iterations,
            )
        metadata["optimizer"] = optimizer
    except ConformerGenerationError:
        raise
    except (ValueError, RuntimeError) as exc:
        fail(f"Force-field optimization failed: {exc}")

    if len(results) != len(conf_ids):
        fail("Optimizer returned the wrong number of results")
    energies = {}
    for cid, (status, energy) in zip(conf_ids, results):
        coords = np.asarray(mol.GetConformer(cid).GetPositions(), dtype=np.float64)
        valid = status == 0 and math.isfinite(energy) and np.isfinite(coords).all()
        metadata["candidates"].append({
            "conformer_id": cid, "optimizer_status": int(status),
            "energy": float(energy) if math.isfinite(energy) else None,
            "usable": bool(valid),
        })
        if valid:
            energies[cid] = float(energy)
    metadata["num_optimized"] = len(energies)
    if not energies:
        fail("No converged finite conformers; failed/nonconverged candidates discarded")
    selected = select_diverse_conformers(
        mol, energies, config.max_conformers, config.rmsd_threshold
    )
    if atom_order_signature(mol) != signature or any(
        atom.GetIntProp("_pcnn_original_index") != atom.GetIdx()
        for atom in mol.GetAtoms()
    ):
        fail("PCNN atom order changed during conformer preprocessing")
    coords = np.stack([mol.GetConformer(cid).GetPositions() for cid in selected])
    if coords.shape != (len(selected), mol.GetNumAtoms(), 3):
        fail("Coordinates do not match [K, number_of_PCNN_graph_atoms, 3]")
    metadata.update(
        status="ok", num_retained=len(selected), selected_conformer_ids=selected,
        energies=[energies[cid] for cid in selected],
    )
    return coords, metadata
