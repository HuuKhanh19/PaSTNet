"""PaSTNet masked conformer pooling, topology refinement and dual readout."""

import torch
from torch import nn
from torch.nn import functional as F

from pastnet.model.geo_ih import _incoming_softmax, relation_indices_from_faces


HIDDEN_DIM = 32
NUM_HEADS = 4
HEAD_DIM = 8
NEGATIVE_SLOPE = 0.01


def _stable(values):
    return values.float() if values.dtype in (torch.float16, torch.bfloat16) else values


class PathwiseConformerPool(nn.Module):
    """Learned softmax over valid conformers, independently for every path."""

    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scores = nn.ModuleList([nn.Linear(hidden_dim, 1, bias=False) for _ in range(4)])

    def forward(self, hidden, *, valid_masks=None, return_weights=False):
        if len(hidden) != 4:
            raise ValueError("PaSTNet conformer pooling requires P0/P1/P2/P3")
        first = hidden[0]
        if not isinstance(first, torch.Tensor) or first.ndim != 3:
            raise ValueError("PaSTNet pooling input must have shape [K,P_r,d]")
        k = first.shape[0]
        if not 1 <= k <= 5:
            raise ValueError("PaSTNet pooling requires conformer capacity in [1,5]")
        if valid_masks is None:
            valid_masks = tuple(torch.ones(value.shape[:2], dtype=torch.bool, device=value.device)
                                for value in hidden)
        if len(valid_masks) != 4:
            raise ValueError("Pooling requires one conformer validity mask per path order")
        result, weights = [], []
        for score, values, mask in zip(self.scores, hidden, valid_masks):
            if (not isinstance(values, torch.Tensor) or values.ndim != 3
                    or values.shape[0] != k or values.shape[-1] != self.hidden_dim
                    or not values.is_floating_point()
                    or values.dtype != first.dtype or values.device != first.device):
                raise ValueError("PaSTNet pooling requires consistent [K,P_r,d] tensors")
            if (mask.shape != values.shape[:2] or mask.dtype != torch.bool or mask.device != values.device
                    or not mask.any(dim=0).all()):
                raise ValueError("Each path needs at least one valid conformer in a boolean [K,P] mask")
            content = torch.where(mask[..., None], values, torch.zeros_like(values))
            if not torch.isfinite(content).all():
                raise ValueError("Valid conformer content must be finite")
            logits = _stable(score(content))
            weight = logits.masked_fill(~mask[..., None], -torch.inf).softmax(dim=0)
            result.append((weight * _stable(content)).sum(dim=0).to(values.dtype))
            weights.append(weight)
        result, weights = tuple(result), tuple(weights)
        return (result, weights) if return_weights else result


class TopologyOrderWeights(nn.Module):
    """Four 8-channel heads for a single content/topology relation order."""

    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        if hidden_dim != HIDDEN_DIM:
            raise ValueError("PaSTNet topology refinement requires four 8-channel heads")
        self.hidden_dim = hidden_dim
        self.num_heads = NUM_HEADS
        self.head_dim = HEAD_DIM
        self.source = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.target = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.coface = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.relation_bias = nn.Parameter(torch.zeros(NUM_HEADS, HEAD_DIM))
        self.attention_vector = nn.Parameter(torch.empty(NUM_HEADS, HEAD_DIM))
        nn.init.xavier_uniform_(self.attention_vector)
        self.value = nn.Linear(2 * hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x, e, faces, relation_indices=None):
        num_paths, num_cofaces = x.shape[0], e.shape[0]
        if num_cofaces == 0:
            return torch.zeros_like(x), torch.zeros_like(e)
        sources, destinations, cofaces, degree = (
            relation_indices_from_faces(faces, num_paths) if relation_indices is None else relation_indices)
        edge_content = e[cofaces]
        relation = (self.source(x)[sources] + self.target(x)[destinations]
                    + self.coface(e)[cofaces]).reshape(-1, NUM_HEADS, HEAD_DIM)
        z = F.leaky_relu(relation + self.relation_bias, negative_slope=NEGATIVE_SLOPE)
        scores = (z * self.attention_vector).sum(dim=-1)
        attention = _incoming_softmax(scores.unsqueeze(0), destinations, num_paths)[0]
        context = _stable(edge_content).new_zeros(num_paths, self.hidden_dim).index_add(
            0, destinations, _stable(edge_content))
        context = (context / degree[:, None]).to(x.dtype)
        base = self.value(torch.cat((x, context), dim=-1)).reshape(num_paths, NUM_HEADS, HEAD_DIM)
        weighted = attention.unsqueeze(-1) * base[sources]
        aggregate = weighted.new_zeros(num_paths, NUM_HEADS, HEAD_DIM).index_add(0, destinations, weighted)
        lower = self.output(F.leaky_relu(aggregate, negative_slope=NEGATIVE_SLOPE)
                            .reshape(num_paths, self.hidden_dim)).to(x.dtype)
        upper = (_stable(z).new_zeros(num_cofaces, NUM_HEADS, HEAD_DIM)
                 .index_add(0, cofaces, _stable(z)) / 2).reshape(num_cofaces, self.hidden_dim).to(e.dtype)
        return lower, upper


class TopologyOnlyRefinement(nn.Module):
    """One sequential P3 -> P2 -> P1 -> P0 sweep after conformer pooling.

    Both participating orders receive residual LayerNorm, retaining the original
    topology refinement convention. Geometry cannot enter this interface.
    """

    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.orders = nn.ModuleList([TopologyOrderWeights(hidden_dim) for _ in range(3)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(4)])

    def forward(self, pooled, topology, *, relation_indices=None):
        if len(pooled) != 4:
            raise ValueError("Final refinement requires all four pooled path orders")
        first = pooled[0]
        if not isinstance(first, torch.Tensor) or first.ndim != 2:
            raise ValueError("Final refinement requires [P_r,d] content")
        for order, values in enumerate(pooled):
            if (not isinstance(values, torch.Tensor)
                    or values.shape != (len(topology.indices[order]), self.hidden_dim)
                    or values.dtype != first.dtype or values.device != first.device
                    or not values.is_floating_point() or not torch.isfinite(values).all()):
                raise ValueError(f"P{order} pooled content must align with topology")
        result = list(pooled)
        for order in (2, 1, 0):
            faces = topology.faces[order]
            if (faces.shape != (result[order + 1].shape[0], 2) or faces.dtype != torch.long
                    or (faces.numel() and (int(faces.min()) < 0
                                          or int(faces.max()) >= result[order].shape[0]))):
                raise ValueError(f"Invalid P{order + 1} prefix/suffix incidence")
            faces = faces.to(device=first.device)
            lower, upper = result[order], result[order + 1]
            dl, du = self.orders[order](lower, upper, faces,
                                       None if relation_indices is None else relation_indices[order])
            result[order] = self.norms[order](lower + dl)
            result[order + 1] = self.norms[order + 1](upper + du)
        return tuple(result)


class FourOrderReadout(nn.Module):
    """Mean each path order to 32 channels; use zero for an empty order."""

    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, refined, *, path_batch=None, num_molecules=None):
        if len(refined) != 4:
            raise ValueError("Molecular readout requires P0/P1/P2/P3")
        for values in refined:
            if not isinstance(values, torch.Tensor) or values.ndim != 2 or values.shape[1] != self.hidden_dim:
                raise ValueError("Readout content must have shape [P_r,d]")
        if path_batch is None:
            means = tuple(_stable(values).mean(dim=0).to(values.dtype) if values.shape[0]
                          else values.new_zeros(self.hidden_dim) for values in refined)
            return torch.cat(means, dim=-1)
        if len(path_batch) != 4 or type(num_molecules) is not int or num_molecules < 1:
            raise ValueError("Packed readout requires four path batch indices and positive molecule count")
        means = []
        for values, batch in zip(refined, path_batch):
            if (batch.shape != values.shape[:1] or batch.dtype != torch.long or batch.device != values.device
                    or (batch.numel() and (int(batch.min()) < 0 or int(batch.max()) >= num_molecules))):
                raise ValueError("Readout path batch indices must identify each path's molecule")
            sums = _stable(values).new_zeros(num_molecules, self.hidden_dim).index_add(0, batch, _stable(values))
            counts = torch.bincount(batch, minlength=num_molecules).clamp_min(1)
            means.append((sums / counts[:, None]).to(values.dtype))
        return torch.cat(means, dim=-1)


class GeoPaSTReadout(nn.Module):
    """Learned path readout plus the uniform SchNet bypass -> molecular property prediction."""

    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.pooling = PathwiseConformerPool(hidden_dim)
        self.refinement = TopologyOnlyRefinement(hidden_dim)
        self.readout = FourOrderReadout(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1),
        )

    def forward(self, hidden, topology, spatial_bypass, *, valid_masks=None,
                path_batch=None, num_molecules=None, relation_indices=None, return_debug=False):
        pooled, weights = self.pooling(hidden, valid_masks=valid_masks, return_weights=True)
        refined = self.refinement(pooled, topology, relation_indices=relation_indices)
        z_path = self.readout(refined, path_batch=path_batch, num_molecules=num_molecules)
        if (not isinstance(spatial_bypass, torch.Tensor) or spatial_bypass.shape != z_path.shape
                or spatial_bypass.device != z_path.device or not torch.isfinite(spatial_bypass).all()):
            raise ValueError("Spatial bypass must be the finite uniform conformer mean, shape [...,128]")
        z_g = torch.cat((z_path, spatial_bypass), dim=-1)
        prediction = self.head(z_g).squeeze(-1)
        if return_debug:
            return prediction, dict(pool_weights=weights, pooled=pooled, refined=refined,
                                    z_path=z_path, z_spatial=spatial_bypass, zG=z_g)
        return prediction
