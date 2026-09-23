"""One shared covalent topology and SE(3)-invariant geometry for K conformers.

No embedding, dynamic contacts, DGL construction, attention, or baseline edits.
Coordinate units are angstrom and angles are radians. Explicit H follow PCNN.
"""

from dataclasses import dataclass

import torch

from pastnet.data.molecule import atom_order_signature, molecule_from_smiles


P1_FEATURES = ("bond_length",)
P2_INTRINSIC_FEATURES = (
    "centroid_distance_i", "centroid_distance_j", "centroid_distance_k",
    "distance_ij", "distance_jk", "distance_ik",
)
P3_FEATURES = (
    "tetrahedron_volume", "cos_torsion", "sin_torsion", "distance_il",
    "quadrilateral_area", "distance_ij_plus_jk", "distance_jk_plus_kl",
)


def _canonical_path(path):
    return min(path, path[::-1])


@dataclass(frozen=True)
class CovalentPathComplex:
    """indices[r]: [P_r,r+1]; faces[r-1]: [P_r,2] prefix/suffix path IDs.

    Paths are simple and unique modulo reversal, ordered lexicographically.
    Face IDs refer to canonical representatives; an orientation is not implied.
    Tensor storage is shared across conformers and must be treated as read-only.
    """

    num_atoms: int
    atom_order: dict
    indices: tuple
    faces: tuple

    def indices_for_conformers(self, num_conformers):
        """Zero-copy [K,P_r,r+1] views; no rebuild or separate atom mapping."""
        if num_conformers < 1:
            raise ValueError("num_conformers must be positive")
        return tuple(index.unsqueeze(0).expand(num_conformers, -1, -1)
                     for index in self.indices)


def build_covalent_path_complex(mol):
    """Enumerate P0/P1/P2/P3 once using RDKit bonds and existing atom IDs.

    Input is already in PCNN atom order (use molecule_from_smiles for caches).
    No hydrogens are added/removed here. Reversal duplicates in legacy directed
    DGL graphs are represented by one logical path; legacy functions are intact.
    """
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("A nonempty RDKit molecule is required")
    n_atoms = mol.GetNumAtoms()
    neighbors = [set() for _ in range(n_atoms)]
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        neighbors[i].add(j)
        neighbors[j].add(i)
    paths = [{(i,) for i in range(n_atoms)}, set(), set(), set()]

    def extend(path):
        order = len(path) - 1
        paths[order].add(_canonical_path(path))
        if order == 3:
            return
        for neighbor in sorted(neighbors[path[-1]]):
            if neighbor not in path:
                extend(path + (neighbor,))

    for i in range(n_atoms):
        extend((i,))
    ordered = [sorted(level) for level in paths]
    indices = tuple(torch.tensor(level, dtype=torch.long).reshape(-1, order + 1)
                    for order, level in enumerate(ordered))
    faces = []
    for order in range(1, 4):
        lower_ids = {path: i for i, path in enumerate(ordered[order - 1])}
        pairs = [(lower_ids[_canonical_path(path[:-1])],
                  lower_ids[_canonical_path(path[1:])]) for path in ordered[order]]
        faces.append(torch.tensor(pairs, dtype=torch.long).reshape(-1, 2))
    return CovalentPathComplex(n_atoms, atom_order_signature(mol), indices, tuple(faces))


@dataclass(frozen=True)
class PathGeometry:
    topology: CovalentPathComplex
    p1: torch.Tensor
    p2: torch.Tensor
    p3: torch.Tensor
    torsions: torch.Tensor  # Diagnostic angles, not the P3 representation.
    bond_angle_valid: torch.Tensor
    torsion_valid: torch.Tensor
    p2_feature_names: tuple


def _norm(vector):
    return torch.linalg.vector_norm(vector, dim=-1)


def _cross(first, second):
    return torch.linalg.cross(first, second, dim=-1)


def _angle(first, second, eps):
    first_length, second_length = _norm(first), _norm(second)
    valid = (first_length > eps) & (second_length > eps)
    denominator = (first_length * second_length).clamp_min(eps * eps)
    cosine = ((first * second).sum(-1) / denominator).clamp(-1, 1)
    sine = _norm(_cross(first, second)) / denominator
    angle = torch.atan2(torch.where(valid, sine, torch.zeros_like(sine)),
                        torch.where(valid, cosine, torch.ones_like(cosine)))
    return angle, valid


def compute_path_geometry(topology, coords, *, encode_two_path="dim_8", eps=1e-8):
    """Compute [K,P_r,d_r] tensors, vectorizing over all K on one topology.

    P2 starts with the conventional bond angle then retains the original PCNN
    dim_8 (9 values) or dim_10 (10 values) intrinsic triangle features. Legacy
    angles are supplementary turning angles. Only centroid DISTANCES are used.
    P3 keeps PCNN's 7 internal quantities but uses the SIGNED sine of torsion.
    Undefined/collinear torsions are masked false with phi=0, (cos,sin)=(1,0).
    Invalid bond angles are zero and masked false. No NaN/Inf is silently filled.
    """
    if encode_two_path not in ("dim_8", "dim_10"):
        raise ValueError("encode_two_path must be dim_8 or dim_10")
    if not 0 < eps < 1:
        raise ValueError("eps must lie in (0,1)")
    # Copy numpy input: cache arrays are read-only; preserve float precision.
    xyz = coords if isinstance(coords, torch.Tensor) else torch.tensor(coords)
    if not xyz.is_floating_point():
        xyz = xyz.to(torch.float64)
    if (xyz.ndim != 3 or xyz.shape[0] < 1
            or xyz.shape[1:] != (topology.num_atoms, 3)):
        raise ValueError(f"Expected [K,{topology.num_atoms},3] with K >= 1; got {tuple(xyz.shape)}")
    if not torch.isfinite(xyz).all():
        raise ValueError("Coordinates contain NaN/Inf")
    xyz = xyz - xyz[:, :1, :]  # Reduce cancellation from a global offset.
    _, p1_index, p2_index, p3_index = (
        indices.to(xyz.device) for indices in topology.indices
    )
    bonds = xyz[:, p1_index]
    p1 = _norm(bonds[:, :, 1] - bonds[:, :, 0]).unsqueeze(-1)

    triangles = xyz[:, p2_index]
    a, b, c = triangles.unbind(dim=2)
    ab, bc, ac = b - a, c - b, c - a
    angle, angle_valid = _angle(-ab, bc, eps)
    turn, _ = _angle(ab, bc, eps)
    center = (a + b + c) / 3
    triangle_features = [
        angle, _norm(a - center), _norm(b - center), _norm(c - center),
        _norm(ab), _norm(bc), _norm(ac), turn,
    ]
    names = ("bond_angle",) + P2_INTRINSIC_FEATURES + ("pcnn_turn_j",)
    if encode_two_path == "dim_8":
        triangle_features.append(turn.square())
        names += ("pcnn_turn_j_squared",)
    else:
        turn_i, _ = _angle(-ab, ac, eps)
        turn_k, _ = _angle(bc, -ac, eps)
        triangle_features.extend((turn_i, turn_k))
        names += ("pcnn_turn_i", "pcnn_turn_k")
    triangle_features.append(0.5 * _norm(_cross(ab, ac)))
    names += ("triangle_area",)
    p2 = torch.stack(triangle_features, dim=-1)

    quadruples = xyz[:, p3_index]
    a, b, c, d = quadruples.unbind(dim=2)
    ab, bc, cd = b - a, c - b, d - c
    lengths = (_norm(ab), _norm(bc), _norm(cd))
    normal1, normal2 = _cross(ab, bc), _cross(bc, cd)
    norm1, norm2 = _norm(normal1), _norm(normal2)
    # Relative area tests flag nearly collinear planes independently of length.
    valid = ((lengths[0] > eps) & (lengths[1] > eps) & (lengths[2] > eps)
             & (norm1 > eps * lengths[0] * lengths[1])
             & (norm2 > eps * lengths[1] * lengths[2]))
    normal1 = normal1 / norm1.clamp_min(eps * eps).unsqueeze(-1)
    normal2 = normal2 / norm2.clamp_min(eps * eps).unsqueeze(-1)
    axis = bc / lengths[1].clamp_min(eps).unsqueeze(-1)
    cosine = (normal1 * normal2).sum(-1).clamp(-1, 1)
    sine = (_cross(normal1, normal2) * axis).sum(-1)
    phi = torch.atan2(torch.where(valid, sine, torch.zeros_like(sine)),
                      torch.where(valid, cosine, torch.ones_like(cosine)))
    volume = ((_cross(ab, c - a) * (d - a)).sum(-1)).abs() / 6
    area = 0.5 * _norm(_cross(b - d, c - a))
    p3 = torch.stack((volume, torch.cos(phi), torch.sin(phi), _norm(d - a),
                      area, lengths[0] + lengths[1], lengths[1] + lengths[2]), dim=-1)
    if not all(torch.isfinite(value).all() for value in (p1, p2, p3, phi)):
        raise ValueError("Non-finite path geometry; check coordinate scale")
    return PathGeometry(topology, p1, p2, p3, phi, angle_valid, valid, names)


def geometry_from_cache(entry, *, encode_two_path="dim_8"):
    """Reuse the exact PCNN atom order; build ONE complex for all saved conformers."""
    mol = molecule_from_smiles(entry.graph_smiles)
    if mol is None or atom_order_signature(mol) != entry.metadata.get("atom_order"):
        raise ValueError("Cached atom order does not match the PCNN graph molecule")
    topology = build_covalent_path_complex(mol)
    return compute_path_geometry(topology, entry.coords, encode_two_path=encode_two_path)
