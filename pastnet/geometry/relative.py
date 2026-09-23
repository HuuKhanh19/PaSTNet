"""Relative geometry and training-only bond calibration for PaSTNet.

The pair convention is ordered (query, key), including diagonal pairs. This
literal full K x K convention is recorded in calibration metadata because the
architecture does not separately prescribe a quantile sampling convention.
No conformers are padded, duplicated or discarded. Calibration writes no files.
"""

from dataclasses import asdict, dataclass
import math

import torch


RELATIVE_GEOMETRY_DIMS = ((0, 0), (8, 8), (3, 2), (4, 3))
RELATIVE_BOND_EPS = 1e-8
BOND_PAIR_CONVENTION = "all_ordered_including_diagonal"


def relative_geometry(order, hidden, *, values=None, valid=None, bond_scale=None):
    """Return even/odd geometry [K,K,g] for one fixed path and hidden [K,d].

    P0 ignores every supplied geometry argument. For P1, values are bond
    lengths in angstrom and bond_scale is the fixed training Q0.95 estimate.
    For P2/P3, values are radians and valid is a boolean [K] mask. Invalid
    values are masked BEFORE trigonometry, so NaN placeholders are harmless.

    The relative-bond epsilon is explicitly chosen as 1e-8: the architecture
    specifies a stabilizer but no separate numeric value for this denominator.
    Geometry arithmetic uses at least float32; outputs match hidden dtype/device.
    """
    if isinstance(order, bool) or not isinstance(order, int) or order not in range(4):
        raise ValueError("PaST path order must be 0, 1, 2 or 3")
    if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 2
            or hidden.shape[1] < 1 or not 1 <= hidden.shape[0] <= 10
            or not hidden.is_floating_point()):
        raise ValueError("PaST hidden states must be floating [K,d] with 1<=K<=10")
    k = hidden.shape[0]
    if order == 0:
        return hidden.new_empty(k, k, 0), hidden.new_empty(k, k, 0)

    calculation_dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
    if values is None:
        raise ValueError(f"P{order} requires geometry values [K]")
    values = torch.as_tensor(values, dtype=calculation_dtype, device=hidden.device)
    if values.shape != (k,):
        raise ValueError("Relative geometry values must have shape [K]")

    if order == 1:
        if not torch.isfinite(values).all() or (values < 0).any():
            raise ValueError("Bond lengths must be finite and nonnegative")
        if bond_scale is None:
            raise ValueError("P1 requires a fitted training-only bond scale")
        scale = torch.as_tensor(bond_scale, dtype=calculation_dtype, device=hidden.device)
        if scale.ndim != 0 or not torch.isfinite(scale) or scale < 0:
            raise ValueError("Bond scale must be a finite nonnegative scalar")
        if scale.requires_grad:
            raise ValueError("Training-fitted bond scale must be fixed, not trainable")
        delta = values[:, None] - values[None, :]
        signed_u = delta / (scale + RELATIVE_BOND_EPS)
        centers = torch.arange(8, dtype=calculation_dtype, device=hidden.device) * (2.0 / 7.0)
        even = torch.exp(-((7.0 / 2.0) ** 2)
                         * (signed_u.abs().unsqueeze(-1) - centers).square())
        odd = signed_u.unsqueeze(-1) * even
        return even.to(hidden.dtype), odd.to(hidden.dtype)

    if valid is None:
        raise ValueError("Angular geometry requires a boolean validity mask [K]")
    valid = torch.as_tensor(valid, device=hidden.device)
    if valid.dtype != torch.bool or valid.shape != (k,):
        raise ValueError("Angular validity must be boolean [K]")
    if not torch.isfinite(values[valid]).all():
        raise ValueError("Valid angular geometry must be finite")
    safe_values = torch.where(valid, values, torch.zeros_like(values))
    delta = safe_values[:, None] - safe_values[None, :]
    pair_valid = valid[:, None] & valid[None, :]
    harmonics = torch.arange(1, order + 1, dtype=calculation_dtype, device=hidden.device)
    phase = delta.unsqueeze(-1) * harmonics
    mask = pair_valid.unsqueeze(-1)
    even_harmonics = torch.where(mask, phase.cos(), torch.zeros_like(phase))
    odd = torch.where(mask, phase.sin(), torch.zeros_like(phase))
    even = torch.cat((even_harmonics, mask.to(calculation_dtype)), dim=-1)
    return even.to(hidden.dtype), odd.to(hidden.dtype)


@dataclass(frozen=True)
class BondScaleEstimate:
    """Immutable, serializable training provenance for a reusable scalar scale."""

    value: float
    num_molecules: int
    num_paths: int
    num_samples: int
    training_ids: tuple = ()
    training_source: str = None
    cache_fingerprint: str = None
    quantile: float = 0.95
    interpolation: str = "linear"
    pair_convention: str = BOND_PAIR_CONVENTION
    calculation_dtype: str = "float64"
    epsilon: float = RELATIVE_BOND_EPS
    fitted_on: str = "train"

    def metadata(self):
        """Return JSON-compatible metadata for future config/checkpoint wiring."""
        result = asdict(self)
        result["training_ids"] = list(self.training_ids)
        return result


def fit_bond_scale(training_lengths, *, training_ids=(), training_source=None,
                   cache_fingerprint=None):
    """Fit Q0.95 over same-bond |length_c-length_c'| from TRAINING molecules.

    Each iterable element is [K,P1] for one molecule, with actual K in [1,10].
    Every ordered conformer pair including self is used for every bond. Thus
    molecules with K=1 contribute zero deltas; no synthetic conformers are made.
    Values from distinct bonds or molecules are never subtracted from each other.
    Empty bond orders contribute no samples. An entirely empty sample set raises.
    Quantiles are computed with float64 linear interpolation and no autograd.
    """
    differences = []
    molecule_count = path_count = sample_count = 0
    for lengths in training_lengths:
        lengths = torch.as_tensor(lengths, dtype=torch.float64, device="cpu").detach()
        if lengths.ndim != 2 or not 1 <= lengths.shape[0] <= 10:
            raise ValueError("Training bond lengths must have shape [K,P1], 1<=K<=10")
        if not torch.isfinite(lengths).all() or (lengths < 0).any():
            raise ValueError("Training bond lengths must be finite and nonnegative")
        molecule_count += 1
        path_count += lengths.shape[1]
        delta = (lengths[:, None, :] - lengths[None, :, :]).abs().reshape(-1)
        sample_count += delta.numel()
        if delta.numel():
            differences.append(delta)
    if not differences:
        raise ValueError("Cannot fit relative-bond scale without training bond samples")
    ids = tuple(training_ids)
    if ids and len(ids) != molecule_count:
        raise ValueError("Training molecule IDs must match the supplied length tensors")
    scale = float(torch.quantile(torch.cat(differences), 0.95, interpolation="linear"))
    if not math.isfinite(scale):
        raise ValueError("Training relative-bond quantile is not finite")
    return BondScaleEstimate(
        value=scale, num_molecules=molecule_count, num_paths=path_count,
        num_samples=sample_count, training_ids=ids,
        training_source=str(training_source) if training_source is not None else None,
        cache_fingerprint=cache_fingerprint,
    )


def fit_training_bond_scale(splits, cache):
    """Read ONLY splits.train and its existing conformer caches to fit the scale.

    This wrapper deliberately never accesses splits.val or splits.test. Cache
    coordinates and canonical bond IDs determine float64 bond lengths directly;
    neither targets nor HRGE's float32 feature conversion enter the statistic.
    No cache generation, dataset mutation, config writes or training occurs here.
    """
    from pastnet.geometry.paths import build_covalent_path_complex
    from pastnet.data.molecule import atom_order_signature, molecule_from_smiles

    train = splits.train
    ids = []

    def training_lengths():
        for molecule in train:
            entry = cache.load(molecule.mol_id, expected_smiles=molecule.canonical_smiles)
            mol = molecule_from_smiles(entry.graph_smiles)
            if mol is None or atom_order_signature(mol) != entry.metadata.get("atom_order"):
                raise ValueError("Cached atom order does not match the PCNN graph molecule")
            topology = build_covalent_path_complex(mol)
            coords = torch.tensor(entry.coords, dtype=torch.float64)
            if (coords.ndim != 3 or coords.shape[1:] != (mol.GetNumAtoms(), 3)
                    or not 1 <= coords.shape[0] <= 10 or not torch.isfinite(coords).all()):
                raise ValueError("Cached coordinates must be finite [K,N,3], 1<=K<=10")
            bonds = topology.indices[1]
            lengths = torch.linalg.vector_norm(
                coords[:, bonds[:, 1]] - coords[:, bonds[:, 0]], dim=-1,
            )
            ids.append(molecule.mol_id)
            yield lengths

    return fit_bond_scale(
        training_lengths(), training_ids=ids,
        training_source=getattr(train, "csv_path", None),
        cache_fingerprint=getattr(cache, "fingerprint", None),
    )
