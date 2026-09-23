"""PaSTNet spatial-to-path adapters with mandatory path-reversal symmetry."""

import torch
from torch import nn

from pastnet.geometry.features import HRGE_DIMS


HIDDEN_DIM = 32

# The channel action of reversing a whole path, not independently sorting
# endpoint scalars. P2 reverses centroid endpoints and adjacent bond lengths
# together; P3 exchanges the two terminal length sums. Full path reversal
# preserves signed torsion under compute_path_geometry's convention.
HRGE_REVERSAL_PERMUTATIONS = (
    tuple(range(92)),
    tuple(range(37)),
    (0, 1, 2, 3, 4, 5, 8, 7, 6, 10, 9, 11, 12),
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 10),
)


def reverse_hrge_features(values, order):
    """Return HRGE for the reversed representative of the same logical path.

    P1 chemical channels belong to the stored RDKit bond, not the path's
    orientation. Atom relabeling must transport those bond/stereo attributes
    consistently (as Chem.RenumberAtoms does); changing wedge encodings or
    regenerating chemical metadata is a different input transformation.
    """
    if type(order) is not int or not 0 <= order <= 3:
        raise ValueError("HRGE path order must be an integer from 0 to 3")
    if (not isinstance(values, torch.Tensor) or values.ndim < 1
            or values.shape[-1] != HRGE_DIMS[order]):
        raise ValueError(f"P{order} HRGE must have {HRGE_DIMS[order]} channels")
    if order < 2:
        return values
    return values[..., HRGE_REVERSAL_PERMUTATIONS[order]]


class SpatialToPathAdapter(nn.Module):
    """Reversal-consistent SchNet + unchanged HRGE adapters, width 32."""

    def __init__(self, *, reversal_invariant=True, hidden_dim=HIDDEN_DIM):
        super().__init__()
        if reversal_invariant is not True or hidden_dim != 32:
            raise ValueError("PaSTNet fixes reversal-consistent adapters and path width32")
        self.reversal_invariant = True
        self.hidden_dim = 32
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(width + spatial, 32), nn.SiLU(), nn.LayerNorm(32))
            for width, spatial in zip(HRGE_DIMS, (128, 256, 384, 512))
        ])

    def forward(self, features, atom_embeddings, topology, *, valid_masks=None):
        if len(features) != 4 or atom_embeddings.ndim != 3 or atom_embeddings.shape[-1] != 128:
            raise ValueError("Adapters require four HRGE orders and atom embeddings [K,N,128]")
        result = []
        for order, (values, projection) in enumerate(zip(features, self.projections)):
            if valid_masks is not None:
                values = torch.where(valid_masks[order][..., None], values, torch.zeros_like(values))
            atoms = atom_embeddings[:, topology.indices[order]]
            if order == 0:
                content = atoms[:, :, 0]
            elif order == 1:
                left, right = atoms.unbind(2)
                content = torch.cat((left + right, (left - right).abs()), -1)
            elif order == 2:
                left, center, right = atoms.unbind(2)
                content = torch.cat((center, left + right, (left - right).abs()), -1)
            else:
                left, inner_left, inner_right, right = atoms.unbind(2)
                content = torch.cat((left + right, (left - right).abs(),
                                     inner_left + inner_right, (inner_left - inner_right).abs()), -1)
            hidden = projection(torch.cat((content, values), -1))
            # Symmetrize the complete nonlinear encoder over the two HRGE
            # representatives. This preserves the existing channel definitions.
            if order >= 2:
                reversed_hidden = projection(torch.cat((content, reverse_hrge_features(values, order)), -1))
                hidden = 0.5 * (hidden + reversed_hidden)
            # Keep residual states consistent (including empty orders) while
            # autocast still accelerates the adapter's channel projections.
            hidden = hidden.to(atom_embeddings.dtype)
            if valid_masks is not None:
                hidden = torch.where(valid_masks[order][..., None], hidden, torch.zeros_like(hidden))
            result.append(hidden)
        return tuple(result)


# Import compatibility only: the HRGE-only architecture no longer exists.
HRGEProjection = SpatialToPathAdapter
