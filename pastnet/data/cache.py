"""Conformer cache IO and validation; reading NEVER generates conformers."""

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase

from pastnet.data.molecule import atom_order_signature, molecule_from_smiles


DEFAULT_CACHE_DIR = (
    Path("data/processed/esol/conformers_etkdg_mmff")
)
SCHEMA_VERSION = 1


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _fingerprint(manifest):
    return hashlib.sha256(_json(manifest).encode("utf-8")).hexdigest()


def _coords_digest(coords):
    return hashlib.sha256(np.asarray(coords, dtype="<f8").tobytes(order="C")).hexdigest()


def _atomic_write(path, writer):
    """Write beside the destination, then atomically replace it (single writer)."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            writer(handle)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def initialize_cache(cache_dir, config):
    """Offline only: create a manifest or reject a conflicting existing cache."""
    directory = Path(cache_dir)
    expected = {
        "schema_version": SCHEMA_VERSION, "generator_version": 1,
        "rdkit_version": rdBase.rdkitVersion, "numpy_version": np.__version__,
        "atom_order_policy": "PCNN_MolFromSmiles_canonical_then_AddHs",
        "config": asdict(config),
    }
    manifest_path = directory / "config.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != expected:
            raise ValueError(
                f"{manifest_path}: cache configuration/version mismatch; "
                "use the original settings or a separate --cache-dir"
            )
    else:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(f"Refusing to initialize nonempty cache without config.json: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        _atomic_write(manifest_path, lambda handle: handle.write(_json(expected).encode("utf-8")))
    index_path = directory / "index.jsonl"
    if not index_path.exists():
        _atomic_write(index_path, lambda handle: handle.write(b""))
    return ConformerCache(directory)


@dataclass(frozen=True)
class CachedConformers:
    coords: np.ndarray
    metadata: dict

    @property
    def graph_smiles(self):
        return self.metadata["graph_smiles"]

    def molecule(self, conformer_index=0):
        """Build the PCNN molecule and attach ONE saved conformer, without embedding."""
        mol = molecule_from_smiles(self.graph_smiles)
        conf = Chem.Conformer(mol.GetNumAtoms())
        conf.Set3D(True)
        for index, position in enumerate(self.coords[conformer_index]):
            conf.SetAtomPosition(index, tuple(float(value) for value in position))
        mol.AddConformer(conf, assignId=True)
        return mol


class ConformerCache:
    """One cache shared by all split seeds; load has no generation fallback.

    Energies are audit metadata only. Use graph_smiles or molecule() to preserve
    the exact graph atom order; do not independently renumber atoms.
    """

    def __init__(self, cache_dir=DEFAULT_CACHE_DIR):
        self.directory = Path(cache_dir)
        self.manifest = json.loads((self.directory / "config.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported conformer cache schema")
        if (self.manifest.get("rdkit_version") != rdBase.rdkitVersion
                or self.manifest.get("numpy_version") != np.__version__):
            raise ValueError("RDKit/NumPy version differs from the conformer cache manifest")
        self.fingerprint = _fingerprint(self.manifest)
        self.index = {}
        for line in (self.directory / "index.jsonl").read_text(encoding="utf-8").splitlines():
            metadata = json.loads(line)
            mol_id = metadata["mol_id"]
            self.path(mol_id)  # Also validate IDs from disk before using them as paths.
            if mol_id in self.index:
                raise ValueError(f"Duplicate mol_id in cache index: {mol_id}")
            if metadata.get("cache_fingerprint") != self.fingerprint:
                raise ValueError(f"Cache fingerprint mismatch in index: {mol_id}")
            self.index[mol_id] = metadata

    def path(self, mol_id):
        if not re.fullmatch(r"[0-9a-f]{16}", mol_id):
            raise ValueError(f"Invalid mol_id: {mol_id!r}")
        return self.directory / f"{mol_id}.npz"

    def _update_index(self, metadata):
        self.index[metadata["mol_id"]] = metadata
        contents = "".join(_json(self.index[key]) + "\n" for key in sorted(self.index))
        _atomic_write(self.directory / "index.jsonl", lambda handle: handle.write(contents.encode("utf-8")))

    def validate_entry(self, mol_id, coords, atom_indices, metadata):
        """Check [K,N,3], ordered PCNN topology, checksum, and selection metadata."""
        if metadata.get("status") != "ok" or metadata.get("mol_id") != mol_id:
            raise ValueError(f"{mol_id}: entry is not a successful conformer cache record")
        if metadata.get("cache_fingerprint") != self.fingerprint:
            raise ValueError(f"{mol_id}: cache configuration fingerprint mismatch")
        if metadata.get("config") != self.manifest["config"]:
            raise ValueError(f"{mol_id}: conformer settings differ from manifest")
        smiles = metadata["canonical_smiles"]
        if hashlib.sha1(smiles.encode("utf-8")).hexdigest()[:16] != mol_id:
            raise ValueError(f"{mol_id}: SMILES identity does not match molecule ID")
        if metadata.get("graph_smiles") != smiles:
            raise ValueError(f"{mol_id}: graph SMILES differs from canonical cache identity")
        mol = molecule_from_smiles(smiles)
        if mol is None or atom_order_signature(mol) != metadata.get("atom_order"):
            raise ValueError(f"{mol_id}: cached atom order does not match PCNN")
        n_atoms = mol.GetNumAtoms()  # Exactly the g.add_nodes count in atom_to_graph.
        maximum = self.manifest["config"]["max_conformers"]
        if (coords.ndim != 3 or coords.shape[1:] != (n_atoms, 3)
                or not 1 <= coords.shape[0] <= maximum):
            raise ValueError(f"{mol_id}: expected [K,{n_atoms},3], 1 <= K <= {maximum}; got {coords.shape}")
        if not np.array_equal(atom_indices, np.arange(n_atoms)):
            raise ValueError(f"{mol_id}: atom indices are not in PCNN order")
        if not np.isfinite(coords).all() or _coords_digest(coords) != metadata.get("coords_sha256"):
            raise ValueError(f"{mol_id}: non-finite or corrupted coordinates")
        kept = coords.shape[0]
        if metadata.get("num_graph_atoms") != n_atoms or metadata.get("num_retained") != kept:
            raise ValueError(f"{mol_id}: atom/conformer counts disagree with coordinates")
        if not kept <= metadata["num_optimized"] <= metadata["num_generated"] <= metadata["num_requested"]:
            raise ValueError(f"{mol_id}: inconsistent generated/optimized/retained counts")
        selected = metadata["selected_conformer_ids"]
        energies = np.asarray(metadata["energies"], dtype=float)
        if len(selected) != kept or len(set(selected)) != kept or energies.shape != (kept,):
            raise ValueError(f"{mol_id}: inconsistent selected conformer IDs/energies")
        if not np.isfinite(energies).all():
            raise ValueError(f"{mol_id}: non-finite retained energies")
        candidates = {item["conformer_id"]: item for item in metadata["candidates"]}
        for cid, energy in zip(selected, energies):
            candidate = candidates.get(cid, {})
            if (candidate.get("optimizer_status") != 0 or not candidate.get("usable")
                    or candidate.get("energy") != energy):
                raise ValueError(f"{mol_id}: retained a failed or inconsistent candidate")

    def load(self, mol_id, *, expected_smiles=None, check_index=True):
        """Read an existing entry or fail. No ETKDG/MMFF/UFF is called here."""
        path = self.path(mol_id)
        if not path.is_file():
            raise FileNotFoundError(f"Missing conformer cache: {path}; run offline preprocessing first")
        with np.load(path, allow_pickle=False) as archive:
            coords = archive["coords"]
            atom_indices = archive["atom_indices"]
            metadata = json.loads(archive["metadata"].item())
        self.validate_entry(mol_id, coords, atom_indices, metadata)
        if check_index and self.index.get(mol_id) != metadata:
            raise ValueError(f"{mol_id}: NPZ metadata does not match index.jsonl")
        if expected_smiles is not None:
            parsed = Chem.MolFromSmiles(expected_smiles)
            if parsed is None or Chem.MolToSmiles(parsed, canonical=True, isomericSmiles=True) != metadata["canonical_smiles"]:
                raise ValueError(f"{mol_id}: requested molecule identity differs from cache")
            expected_mol = molecule_from_smiles(expected_smiles)
            if expected_mol is None or atom_order_signature(expected_mol) != metadata["atom_order"]:
                raise ValueError(f"{mol_id}: requested graph atom order differs; use cached graph_smiles")
        coords.setflags(write=False)
        return CachedConformers(coords=coords, metadata=metadata)

    def store(self, coords, metadata):
        """Offline writer; existing entries cannot be overwritten accidentally."""
        metadata = dict(metadata, cache_fingerprint=self.fingerprint, coords_sha256=_coords_digest(coords))
        mol_id = metadata["mol_id"]
        path = self.path(mol_id)
        if path.exists():
            raise FileExistsError(f"Conformer cache already exists: {path}")
        atom_indices = np.arange(coords.shape[1], dtype=np.int64)
        self.validate_entry(mol_id, coords, atom_indices, metadata)
        _atomic_write(path, lambda handle: np.savez_compressed(
            handle, coords=np.asarray(coords, dtype=np.float64), atom_indices=atom_indices,
            metadata=np.array(_json(metadata)),
        ))
        self._update_index(metadata)

    def record_failure(self, metadata):
        if metadata.get("status") != "failed" or self.path(metadata["mol_id"]).exists():
            raise ValueError("Cannot replace a successful entry with failure metadata")
        self._update_index(dict(metadata, cache_fingerprint=self.fingerprint))

    def recover_entry(self, mol_id):
        """Offline recovery after an interrupted NPZ-write/index-update pair."""
        entry = self.load(mol_id, check_index=False)
        self._update_index(entry.metadata)
        return entry
