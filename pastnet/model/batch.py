"""PaSTNet mixed-K batching: valid spatial graphs and masked canonical paths."""

from dataclasses import dataclass
import torch

from pastnet.geometry.features import HRGEFeatures, prepare_relative_geometry
from pastnet.geometry.paths import CovalentPathComplex

K_MAX = 5


@dataclass(frozen=True)
class PaSTNetBatch:
    topology: CovalentPathComplex
    features: tuple
    valid_masks: tuple
    path_batch: tuple
    relative_geometry: tuple
    relation_indices: tuple
    atomic_numbers: torch.Tensor
    spatial_edges: torch.Tensor
    spatial_rbf: torch.Tensor
    spatial_envelope: torch.Tensor
    atom_scatter: torch.Tensor
    atom_graph: torch.Tensor
    conformer_molecule: torch.Tensor
    atom_counts: torch.Tensor
    conformer_counts: torch.Tensor
    num_molecules: int

    @property
    def counts(self):
        return tuple(index.shape[0] for index in self.topology.indices)


def _validate_molecules(molecules):
    if not isinstance(molecules, (tuple, list)) or not molecules:
        raise ValueError("PaSTNet expects a nonempty sequence of HRGE molecules")
    for molecule in molecules:
        if not isinstance(molecule, HRGEFeatures) or not 1 <= molecule.num_conformers <= K_MAX:
            raise ValueError("PaSTNet requires actual K in [1,5]")
        if molecule.coords is None or molecule.atomic_numbers is None or molecule.spatial_edges is None:
            raise ValueError("PaSTNet requires cached coordinates and atomic numbers")
        if molecule.features[0].device != molecules[0].features[0].device:
            raise ValueError("All molecules must be on the same device")


def pack_molecules(molecules, *, bond_scale):
    """No conformer copying: spatial atoms contain exactly sum(K_m*N_m)."""
    _validate_molecules(molecules)
    device = molecules[0].features[0].device
    indices, faces = [[] for _ in range(4)], [[] for _ in range(3)]
    features, masks, path_batch = ([[] for _ in range(4)] for _ in range(3))
    relative = [[[], []] for _ in range(4)]
    relations = [[[] for _ in range(4)] for _ in range(3)]
    numbers, edges, rbf, envelopes, scatter, atom_graph, conformer_molecule = ([] for _ in range(7))
    atom_counts, conformer_counts = [], []
    total_atoms = sum(m.topology.num_atoms for m in molecules)
    atom_offset = spatial_offset = graph_offset = 0
    path_offsets = [0] * 4
    atom_order = dict(atoms=[], bonds=[])
    for molecule_id, molecule in enumerate(molecules):
        k, n = molecule.num_conformers, molecule.topology.num_atoms
        static_relative = prepare_relative_geometry(molecule, bond_scale)
        for order in range(4):
            count = molecule.counts[order]
            indices[order].append(molecule.topology.indices[order] + atom_offset)
            raw_features = molecule.features[order]
            padded_features = raw_features.new_zeros(K_MAX, count, raw_features.shape[-1])
            padded_features[:k] = raw_features
            features[order].append(padded_features)
            masks[order].append((torch.arange(K_MAX, device=device) < k)[:, None].expand(-1, count))
            path_batch[order].append(torch.full((count,), molecule_id, dtype=torch.long, device=device))
            for parity in range(2):
                raw = static_relative[order][parity]
                padded = raw.new_zeros(count, K_MAX, K_MAX, raw.shape[-1])
                padded[:, :k, :k] = raw
                relative[order][parity].append(padded)
            if order < 3:
                faces[order].append(molecule.topology.faces[order] + path_offsets[order])
                source, target, coface, degree = molecule.relation_indices[order]
                for parts, value in zip(relations[order],
                        (source + path_offsets[order], target + path_offsets[order],
                         coface + path_offsets[order+1], degree)):
                    parts.append(value)
        numbers.append(molecule.atomic_numbers.repeat(k))
        edges.append(molecule.spatial_edges + spatial_offset)
        rbf.append(molecule.spatial_rbf)
        envelopes.append(molecule.spatial_envelope)
        scatter.append((torch.arange(k, device=device)[:, None] * total_atoms
                        + atom_offset + torch.arange(n, device=device)[None]).reshape(-1))
        atom_graph.append(torch.arange(graph_offset, graph_offset+k, device=device).repeat_interleave(n))
        conformer_molecule.append(torch.full((k,), molecule_id, dtype=torch.long, device=device))
        atom_counts.extend([n] * k)
        conformer_counts.append(k)
        atom_order['atoms'].extend(molecule.topology.atom_order['atoms'])
        atom_order['bonds'].extend([[a+atom_offset,b+atom_offset] for a,b in molecule.topology.atom_order['bonds']])
        atom_offset += n
        spatial_offset += k*n
        graph_offset += k
        path_offsets = [a+b for a,b in zip(path_offsets,molecule.counts)]
    topology = CovalentPathComplex(total_atoms, atom_order,
        tuple(torch.cat(parts) for parts in indices), tuple(torch.cat(parts) for parts in faces))
    return PaSTNetBatch(
        topology=topology, features=tuple(torch.cat(parts, 1) for parts in features),
        valid_masks=tuple(torch.cat(parts, 1) for parts in masks),
        path_batch=tuple(torch.cat(parts) for parts in path_batch),
        relative_geometry=tuple(tuple(torch.cat(parts) for parts in pair) for pair in relative),
        relation_indices=tuple(tuple(torch.cat(parts) for parts in relation) for relation in relations),
        atomic_numbers=torch.cat(numbers),
        spatial_edges=torch.cat(edges, 1), spatial_rbf=torch.cat(rbf), spatial_envelope=torch.cat(envelopes),
        atom_scatter=torch.cat(scatter), atom_graph=torch.cat(atom_graph),
        conformer_molecule=torch.cat(conformer_molecule),
        atom_counts=torch.tensor(atom_counts, device=device),
        conformer_counts=torch.tensor(conformer_counts, device=device), num_molecules=len(molecules))


def spatial_to_paths(atoms, batch):
    packed = atoms.new_zeros(K_MAX * batch.topology.num_atoms, 128)
    return packed.index_copy(0, batch.atom_scatter, atoms).reshape(K_MAX, -1, 128)


def spatial_bypass(atoms, batch):
    """Mean atoms within each actual conformer, then uniform conformer mean."""
    dtype = torch.float64 if atoms.dtype == torch.float64 else torch.float32
    atoms = atoms.to(dtype)
    conformers = atoms.new_zeros(batch.atom_counts.shape[0], 128).index_add(0, batch.atom_graph, atoms)
    conformers = conformers / batch.atom_counts[:, None]
    molecules = atoms.new_zeros(batch.num_molecules, 128).index_add(0, batch.conformer_molecule, conformers)
    return molecules / batch.conformer_counts[:, None]


def forward_molecules(model, molecules, *, return_debug=False):
    batch = pack_molecules(molecules, bond_scale=model.backbone.bond_scale_value)
    return model.forward_packed(batch, return_debug=return_debug)
