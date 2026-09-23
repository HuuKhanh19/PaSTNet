"""PaSTNet geometry-conditioned P3 -> P2 -> P1 -> P0 propagation.

Each order uses one 32-channel projection reshaped into four 8-channel heads.
Directed relations are static and may be supplied from the packed topology cache.
"""

import torch
from torch import nn
from torch.nn import functional as F

from pastnet.model.config import stage_drop_probability


HIDDEN_DIM = 32
NUM_HEADS = 4
HEAD_DIM = 8
GEOMETRY_DIM = 12
GEOMETRY_INPUT_DIMS = (16, 6, 7)
RMS_EPS = 1e-6
DROP_PROBABILITIES = (0.0, 0.0, 0.0)


def absolute_geometry_inputs(hrge):
    """View the already validity-masked RBF/E_theta/E_phi HRGE channels."""
    return (hrge.features[1][..., 21:], hrge.features[2][..., :6],
            hrge.features[3][..., 1:8])


class GeometryEncoderBank(nn.Module):
    """Three order-specific encoders shared across all configured stages.

    The geometry encoder's RMSNorm belongs to this shared bank (architecture
    section 11). Content RMSNorms are separate, stage-specific parameters.
    alpha_geo,r is a learned scalar per order, initialized to 0.1 and shared
    along with this bank; it has no stage/head index in the architecture.
    """

    def __init__(self, hidden_dim=HIDDEN_DIM, geometry_strength_init=0.1):
        super().__init__()
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(width, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, GEOMETRY_DIM),
                          nn.RMSNorm(GEOMETRY_DIM, eps=RMS_EPS))
            for width in GEOMETRY_INPUT_DIMS
        ])
        self.geometry_strength = nn.Parameter(torch.full((3,), geometry_strength_init))

    def forward(self, geometry_inputs):
        if len(geometry_inputs) != 3:
            raise ValueError("Geo-IH requires three absolute geometry tensors")
        for r, (values, width) in enumerate(zip(geometry_inputs, GEOMETRY_INPUT_DIMS)):
            if (not isinstance(values, torch.Tensor) or values.ndim != 3
                    or values.shape[-1] != width or not values.is_floating_point()
                    or not torch.isfinite(values).all()):
                raise ValueError(f"Geo-IH geometry order {r} must be finite [K,P,{width}]")
        return tuple(encoder(values) for encoder, values in zip(self.encoders, geometry_inputs))


def _incoming_softmax(scores, destinations, num_paths, valid=None):
    """Stable softmax over incoming relations, independently per K and head."""
    scores = scores.float() if scores.dtype in (torch.float16, torch.bfloat16) else scores
    if valid is not None:
        scores = scores.masked_fill(~valid[..., None], -torch.inf)
    indices = destinations.view(1, -1, 1).expand_as(scores)
    maxima = scores.new_full((scores.shape[0], num_paths, scores.shape[-1]), -torch.inf)
    maxima.scatter_reduce_(1, indices, scores.detach(), reduce="amax", include_self=True)
    maxima = torch.where(torch.isfinite(maxima), maxima, torch.zeros_like(maxima))
    exponentials = (scores - maxima.gather(1, indices)).exp()
    denominators = scores.new_zeros(maxima.shape).scatter_add(1, indices, exponentials)
    return exponentials / denominators.gather(1, indices).clamp_min(torch.finfo(scores.dtype).tiny)


def relation_indices_from_faces(faces, num_paths):
    """Precompute both directed incidences and a static lower-path degree."""
    source = faces.reshape(-1)
    target = faces.flip(-1).reshape(-1)
    coface = torch.arange(len(faces), device=faces.device).repeat_interleave(2)
    degree = torch.bincount(target, minlength=num_paths).clamp_min(1)
    return source, target, coface, degree


class GeoIHOrderWeights(nn.Module):
    """Four 8-channel heads with geometry-conditioned routing and values."""

    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        if hidden_dim != HIDDEN_DIM:
            raise ValueError("PaSTNet Geo-IH requires path width 32 and four 8-channel heads")
        self.hidden_dim = hidden_dim
        self.num_heads = NUM_HEADS
        self.head_dim = HEAD_DIM
        self.source = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.target = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.coface = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.relation_bias = nn.Parameter(torch.zeros(NUM_HEADS, HEAD_DIM))
        self.attention_vector = nn.Parameter(torch.empty(NUM_HEADS, HEAD_DIM))
        nn.init.xavier_uniform_(self.attention_vector)
        self.film = nn.Linear(GEOMETRY_DIM, 4 * self.hidden_dim)
        self.value = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

    def forward(self, x, e, faces, g, strength, return_debug=False,
                relation_indices=None, valid=None):
        k, num_paths, _ = x.shape
        num_cofaces = e.shape[1]
        if num_cofaces == 0:
            result = (torch.zeros_like(x), torch.zeros_like(e))
            if return_debug:
                return result, dict(routing=x.new_empty(k, 0, NUM_HEADS, HEAD_DIM),
                                    attention=x.new_empty(k, 0, NUM_HEADS),
                                    message=x.new_empty(k, 0, NUM_HEADS, HEAD_DIM))
            return result
        # Interleaved directions [face0->face1, face1->face0] per coface.
        sources, destinations, cofaces, degree = (
            relation_indices_from_faces(faces, num_paths) if relation_indices is None else relation_indices)
        edge_content = e[:, cofaces]
        relation = (self.source(x)[:, sources] + self.target(x)[:, destinations]
                    + self.coface(e)[:, cofaces]).reshape(k, -1, NUM_HEADS, HEAD_DIM)
        relation = relation + self.relation_bias
        modulation = strength * self.film(g).reshape(k, num_cofaces, NUM_HEADS, 4, HEAD_DIM).tanh()
        gamma, beta, lam, mu = modulation[:, cofaces].unbind(dim=-2)
        z = F.silu((1 + gamma) * relation + beta)
        scores = (z * self.attention_vector).sum(dim=-1)
        relation_valid = None if valid is None else valid[:, cofaces]
        attention = _incoming_softmax(scores, destinations, num_paths, relation_valid)

        # Mean incoming coface context for EACH SOURCE, not its destination.
        stable_edges = edge_content.float() if edge_content.dtype in (torch.float16, torch.bfloat16) else edge_content
        context = stable_edges.new_zeros(k, num_paths, self.hidden_dim).index_add(1, destinations, stable_edges)
        context = (context / degree.view(1, -1, 1)).to(x.dtype)
        base_value = self.value(torch.cat((x, context), dim=-1)).reshape(k, num_paths, NUM_HEADS, HEAD_DIM)
        message = (1 + lam) * base_value[:, sources] + mu
        weighted = attention.unsqueeze(-1) * message
        aggregated = weighted.new_zeros(k, num_paths, NUM_HEADS, HEAD_DIM).index_add(
            1, destinations, weighted)
        delta_lower = self.output(aggregated.reshape(k, num_paths, self.hidden_dim)).to(x.dtype)
        stable_z = z.float() if z.dtype in (torch.float16, torch.bfloat16) else z
        delta_upper = (stable_z.new_zeros(k, num_cofaces, NUM_HEADS, HEAD_DIM)
                       .index_add(1, cofaces, stable_z) / 2).reshape(k, num_cofaces, self.hidden_dim).to(e.dtype)
        result = (delta_lower, delta_upper)
        if return_debug:
            return result, dict(routing=z, attention=attention, message=message)
        return result


class GeoIHWeights(nn.Module):
    """One stage's propagation parameters, distinct across relation orders."""

    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.orders = nn.ModuleList([GeoIHOrderWeights(hidden_dim) for _ in range(3)])


class GeoIHStageState(nn.Module):
    """Content RMSNorm and per-channel LayerScale owned by one stage.

    P1/P2 use the same order's norm/scale when acting as lower or coface content
    in different substeps. No state or parameter is shared between stages here.
    """

    def __init__(self, hidden_dim=HIDDEN_DIM, layerscale_init=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.content_norms = nn.ModuleList([nn.RMSNorm(self.hidden_dim, eps=RMS_EPS) for _ in range(4)])
        self.layer_scales = nn.ParameterList([nn.Parameter(torch.full((self.hidden_dim,), layerscale_init))
                                             for _ in range(4)])


class GeoIHBlock(nn.Module):
    """One standalone Geo-IH stage: [K,P_r,d] -> [K,P_r,d], r=0..3.

    Pass the same GeometryEncoderBank to all stage calls, and the same
    GeoIHWeights for this stage. Only this block's content norms/LayerScales
    are registered here; heavy modules stay registered once with their owner.
    stage is 1-based and selects the specified stochastic-depth probability.
    One scalar drop decision is shared by the entire molecule/Geo-IH branch.
    Eval mode is deterministic. No post-residual normalization is performed.
    """

    def __init__(self, stage=1, *, hidden_dim=HIDDEN_DIM, num_stages=3,
                 stochastic_depth_max=0.0, layerscale_init=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.stage = stage
        self.drop_probability = stage_drop_probability(stage, num_stages, stochastic_depth_max)
        self.state = GeoIHStageState(hidden_dim, layerscale_init)

    def _validate(self, hidden, topology, geometry_inputs):
        if len(hidden) != 4 or len(geometry_inputs) != 3:
            raise ValueError("Geo-IH requires four content orders and three geometry orders")
        if not isinstance(hidden[0], torch.Tensor) or hidden[0].ndim != 3:
            raise ValueError("Geo-IH content must have shape [K,P_r,d]")
        first = hidden[0]
        k = first.shape[0]
        if not 1 <= k <= 5:
            raise ValueError("Geo-IH requires conformer capacity 1<=K<=5")
        for r, h in enumerate(hidden):
            if (not isinstance(h, torch.Tensor) or h.shape != (k, len(topology.indices[r]), self.hidden_dim)
                    or h.dtype != first.dtype or h.device != first.device
                    or not h.is_floating_point() or not torch.isfinite(h).all()):
                raise ValueError(f"P{r} must be finite [K,P_r,d] content with consistent dtype/device")
        for r, (raw, width) in enumerate(zip(geometry_inputs, GEOMETRY_INPUT_DIMS)):
            count = hidden[r + 1].shape[1]
            if (not isinstance(raw, torch.Tensor) or raw.shape != (k, count, width)
                    or not raw.is_floating_point() or raw.device != first.device
                    or not torch.isfinite(raw).all()):
                raise ValueError(f"Geometry input {r} must align with its coface content")
            faces = topology.faces[r]
            if (faces.shape != (count, 2) or faces.dtype != torch.long
                    or (faces.numel() and (int(faces.min()) < 0 or int(faces.max()) >= hidden[r].shape[1]))):
                raise ValueError(f"Invalid prefix/suffix faces for P{r}")

    def forward(self, hidden, topology, geometry_inputs, *, weights, geometry_encoders,
                geometry_embeddings=None, drop_multiplier=None, valid_masks=None,
                relation_indices=None):
        if valid_masks is not None:
            if len(valid_masks) != 4:
                raise ValueError("Geo-IH requires one conformer validity mask per order")
            for values, mask in zip(hidden, valid_masks):
                if mask.shape != values.shape[:2] or mask.dtype != torch.bool or mask.device != values.device:
                    raise ValueError("Geo-IH validity masks must be boolean [K,P] on the content device")
            hidden = tuple(torch.where(mask[..., None], value, torch.zeros_like(value))
                           for value, mask in zip(hidden, valid_masks))
            geometry_inputs = tuple(torch.where(mask[..., None], value, torch.zeros_like(value))
                                    for value, mask in zip(geometry_inputs, valid_masks[1:]))
        self._validate(hidden, topology, geometry_inputs)
        # Absolute geometry and encoder weights are identical in all stages.
        # Reusing this differentiable result preserves the sum of all gradients.
        embeddings = (geometry_encoders(geometry_inputs) if geometry_embeddings is None
                      else geometry_embeddings)
        if drop_multiplier is None:
            multiplier = hidden[0].new_ones(())
            if self.training and self.drop_probability:
                keep = 1 - self.drop_probability
                multiplier = multiplier.bernoulli_(keep) / keep
            multipliers = (multiplier,) * 4
        else:
            # Packed disjoint molecules retain one decision per molecule/branch;
            # its value is repeated over that molecule's paths, never conformers.
            if len(drop_multiplier) != 4:
                raise ValueError("Geo-IH packed multipliers require four path orders")
            multipliers = []
            for order, value in enumerate(drop_multiplier):
                if (value.shape != (hidden[order].shape[1],)
                        or value.dtype != hidden[order].dtype
                        or value.device != hidden[order].device):
                    raise ValueError("Geo-IH multiplier must align with each path order")
                multipliers.append(value.view(1, -1, 1))
        result = list(hidden)
        for r in (2, 1, 0):
            if result[r + 1].shape[1] == 0:
                continue
            x = self.state.content_norms[r](result[r])
            e = self.state.content_norms[r + 1](result[r + 1])
            faces = topology.faces[r].to(device=x.device)
            delta_lower, delta_upper = weights.orders[r](
                x, e, faces, embeddings[r], geometry_encoders.geometry_strength[r],
                relation_indices=None if relation_indices is None else relation_indices[r],
                valid=None if valid_masks is None else valid_masks[r + 1])
            # Both increments see the same pre-substep states. Subsequent r
            # receives the updated higher-order content immediately.
            result[r] = result[r] + multipliers[r] * self.state.layer_scales[r] * delta_lower
            result[r + 1] = result[r + 1] + multipliers[r + 1] * self.state.layer_scales[r + 1] * delta_upper
            if valid_masks is not None:
                result[r] = result[r].masked_fill(~valid_masks[r][..., None], 0)
                result[r + 1] = result[r + 1].masked_fill(~valid_masks[r + 1][..., None], 0)
        return tuple(result)
