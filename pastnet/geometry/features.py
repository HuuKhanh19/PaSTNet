"""Geometry features on the shared covalent path complex.

Build cached spatial and path inputs for PaSTNet.
Coordinates use angstrom, angles use radians, and path IDs follow the supplied
atom order. Cache inputs retain PCNN's explicit hydrogens and actual K.
"""

from dataclasses import dataclass, field, replace
import math

import torch

from pastnet.geometry.encodings import BondRBF
from pastnet.geometry.encodings import encode_bond_14, get_node_attributes
from pastnet.geometry.paths import CovalentPathComplex, build_covalent_path_complex, compute_path_geometry
from pastnet.data.molecule import atom_order_signature, molecule_from_smiles


HRGE_DIMS = (92, 37, 13, 12)
EPSILON_GEO = 1e-8


@dataclass(frozen=True)
class HRGEFeatures:
    """One molecule: shared indices plus float32 [K,P_r,F_r] features.

    Scalar geometry is retained separately from the encodings. Invalid angles
    have finite placeholders and explicit bool masks, never missing conformers.
    Treat these tensors and the shared topology as read-only.
    """

    topology: CovalentPathComplex
    features: tuple
    bond_lengths: torch.Tensor
    angles: torch.Tensor
    torsions: torch.Tensor
    angle_valid: torch.Tensor
    torsion_valid: torch.Tensor
    mol_id: str = "synthetic"
    coords: torch.Tensor = None
    atomic_numbers: torch.Tensor = None
    spatial_edges: torch.Tensor = None
    spatial_rbf: torch.Tensor = None
    spatial_envelope: torch.Tensor = None
    relation_indices: tuple = None
    _relative_cache: dict = field(default_factory=dict, repr=False, compare=False)
    _device_cache: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def counts(self):
        return tuple(len(indices) for indices in self.topology.indices)

    @property
    def num_conformers(self):
        return self.features[0].shape[0]

    def to(self, device):
        device = torch.device(device)
        if device == self.features[0].device:
            return self
        key = str(device)
        if key in self._device_cache:
            return self._device_cache[key]
        topology = replace(
            self.topology,
            indices=tuple(index.to(device) for index in self.topology.indices),
            faces=tuple(faces.to(device) for faces in self.topology.faces),
        )
        result = replace(
            self, topology=topology,
            features=tuple(value.to(device) for value in self.features),
            bond_lengths=self.bond_lengths.to(device), angles=self.angles.to(device),
            torsions=self.torsions.to(device), angle_valid=self.angle_valid.to(device),
            torsion_valid=self.torsion_valid.to(device),
            coords=self.coords.to(device) if self.coords is not None else None,
            atomic_numbers=self.atomic_numbers.to(device) if self.atomic_numbers is not None else None,
            spatial_edges=self.spatial_edges.to(device) if self.spatial_edges is not None else None,
            spatial_rbf=self.spatial_rbf.to(device) if self.spatial_rbf is not None else None,
            spatial_envelope=self.spatial_envelope.to(device) if self.spatial_envelope is not None else None,
            relation_indices=tuple(tuple(x.to(device) for x in relation)
                                   for relation in self.relation_indices) if self.relation_indices is not None else None,
            _relative_cache={},
            _device_cache={},
        )
        self._device_cache[key] = result
        return result


def build_hrge_features(mol, coords, *, topology=None, mol_id="synthetic"):
    """Build the fixed 92/37/13/12D PaSTNet representation, preserving every K row.

    ``mol`` already owns the desired atom order; no hydrogens or conformers are
    added here. Optional topology allows reuse of exactly the same path IDs.
    The unchanged path geometry primitive supplies measured lengths, conventional
    bond angles, signed torsions, centroid distances, areas and volumes.

    Validity follows that primitive with epsilon_geo=1e-8: angle edges must
    exceed epsilon; torsion edges must also exceed epsilon and plane normal
    norms must exceed epsilon times the adjacent edge lengths' product.
    Collinear nonzero bonds still define a valid angle, but not a torsion.
    """
    if getattr(coords, "ndim", None) != 3 or not 1 <= coords.shape[0] <= 5:
        raise ValueError("PaSTNet requires coordinates [K,N,3] with 1<=K<=5")
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("PaSTNet requires a nonempty molecule")
    if topology is None:
        topology = build_covalent_path_complex(mol)
    elif (topology.num_atoms != mol.GetNumAtoms()
          or topology.atom_order != atom_order_signature(mol)):
        raise ValueError("Shared topology does not match the molecule's atom order")
    geometry = compute_path_geometry(topology, coords, encode_two_path="dim_8", eps=EPSILON_GEO)
    k = coords.shape[0]
    device = geometry.p1.device
    if any(value.device != device for value in (*topology.indices, *topology.faces)):
        topology = replace(topology,
            indices=tuple(value.to(device) for value in topology.indices),
            faces=tuple(value.to(device) for value in topology.faces))

    atoms = torch.tensor(
        [get_node_attributes(atom.GetSymbol(), atom_features="cgcnn") for atom in mol.GetAtoms()],
        dtype=torch.float32, device=device,
    )
    bonds = torch.tensor(
        [encode_bond_14(mol.GetBondBetweenAtoms(int(i), int(j))) for i, j in topology.indices[1]],
        dtype=torch.float32, device=device,
    ).reshape(-1, 21)
    lengths = geometry.p1[..., 0].float()
    # Reuse the fixed 16-RBF encoding (0..4 angstrom).
    bond_rbf = BondRBF().to(device)
    f0 = atoms.unsqueeze(0).expand(k, -1, -1)
    f1 = torch.cat((bonds.unsqueeze(0).expand(k, -1, -1), bond_rbf(lengths)), dim=-1)

    theta = geometry.p2[..., 0]
    e_theta = torch.stack((theta / math.pi, theta.cos(), theta.sin(),
                           (2 * theta).cos(), (2 * theta).sin(), torch.ones_like(theta)), dim=-1)
    e_theta = torch.where(geometry.bond_angle_valid.unsqueeze(-1), e_theta, torch.zeros_like(e_theta))
    # Path geometry p2 columns 1:7 are d_i,d_j,d_k,l_a,l_b,l_ab; its last is area.
    # Its legacy turning-angle columns are deliberately not part of PaSTNet geometry.
    f2 = torch.cat((e_theta, geometry.p2[..., 1:7], geometry.p2[..., -1:]), dim=-1).float()

    phi = geometry.torsions
    e_phi = torch.stack((phi.cos(), phi.sin(), (2 * phi).cos(), (2 * phi).sin(),
                        (3 * phi).cos(), (3 * phi).sin(), torch.ones_like(phi)), dim=-1)
    e_phi = torch.where(geometry.torsion_valid.unsqueeze(-1), e_phi, torch.zeros_like(e_phi))
    # Preserve V, d_il, A4, l_a+l_b, l_b+l_c; replace the original cos/sin by E_phi.
    f3 = torch.cat((geometry.p3[..., :1], e_phi, geometry.p3[..., 3:]), dim=-1).float()
    features = (f0, f1, f2, f3)
    for order, (values, width) in enumerate(zip(features, HRGE_DIMS)):
        if values.shape != (k, len(topology.indices[order]), width) or not torch.isfinite(values).all():
            raise ValueError(f"{mol_id}: invalid PaSTNet geometry features at P{order}")
    from pastnet.model.schnet import spatial_graph
    xyz = (coords.detach().clone().to(device=device, dtype=torch.float32)
           if isinstance(coords, torch.Tensor) else torch.tensor(coords, device=device, dtype=torch.float32))
    edges, rbf, envelope = spatial_graph(xyz)
    relations = []
    for order, faces in enumerate(topology.faces):
        source, target = faces.reshape(-1), faces.flip(-1).reshape(-1)
        coface = torch.arange(faces.shape[0], device=faces.device).repeat_interleave(2)
        degree = torch.bincount(target, minlength=topology.indices[order].shape[0]).clamp_min(1)
        relations.append((source, target, coface, degree))
    return HRGEFeatures(
        topology=topology, features=features, bond_lengths=lengths,
        angles=theta.float(), torsions=phi.float(), angle_valid=geometry.bond_angle_valid,
        torsion_valid=geometry.torsion_valid, mol_id=mol_id,
        coords=xyz,
        atomic_numbers=torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()],
                                    dtype=torch.long, device=device),
        spatial_edges=edges, spatial_rbf=rbf, spatial_envelope=envelope,
        relation_indices=tuple(relations),
    )


def hrge_from_cache(entry, mol_id="synthetic"):
    """Read an already loaded cache entry without generation or atom renumbering."""
    mol = molecule_from_smiles(entry.graph_smiles)
    if mol is None or atom_order_signature(mol) != entry.metadata.get("atom_order"):
        raise ValueError("Cached atom order does not match the PCNN graph molecule")
    return build_hrge_features(mol, entry.coords, mol_id=mol_id)


def prepare_relative_geometry(hrge, bond_scale):
    """Memoize fixed relative geometry per immutable molecule and fitted scale.

    The content argument is used only for shape/device; no trainable output is
    cached. Device transfer intentionally creates a fresh cache.
    """
    from pastnet.model.past_batched import batched_relative_geometry
    key = float(bond_scale)
    if key not in hrge._relative_cache:
        arguments = ({}, dict(values=hrge.bond_lengths, bond_scale=key),
                     dict(values=hrge.angles, valid=hrge.angle_valid),
                     dict(values=hrge.torsions, valid=hrge.torsion_valid))
        hrge._relative_cache[key] = tuple(
            batched_relative_geometry(order, hrge.features[order], **arguments[order])
            for order in range(4))
    return hrge._relative_cache[key]
