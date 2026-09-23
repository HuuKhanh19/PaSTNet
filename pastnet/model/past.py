"""PaSTNet: 32-dimensional path states; pair width 16 (P0) or 32 (ablation).

Pair lifting -> tensor layers -> collapse -> exact full-minus-self
residualization -> gated residual. The self branch reruns the SAME tensor core
and collapse on each initial lifted diagonal, with K=1 and zero odd input.
All path and singleton batching uses the shared masked core in past_batched.
"""

import torch
from torch import nn

from pastnet.geometry.relative import relative_geometry, RELATIVE_GEOMETRY_DIMS
from pastnet.model.config import stage_drop_probability


HIDDEN_DIM = 32
PAIR_DIM = 16
RMS_EPS = 1e-6
TENSOR_LAYERS = 2
SYMMETRIC_INPUT_DIMS = (96, 104, 99, 100)
ODD_INPUT_DIMS = (32, 40, 34, 35)


def enforce_parity(symmetric, odd):
    """Project conformer axes onto exact symmetric/antisymmetric tensors."""
    return ((symmetric + symmetric.transpose(-3, -2)) * 0.5,
            (odd - odd.transpose(-3, -2)) * 0.5)


def symmetric_basis(symmetric):
    """The finalized nine symmetric 2->2 maps in their established order.

    Contractions are MEANS over actual K, including diagonal entries. Each
    result is [K,K,d]; no tensor product over separate molecular paths occurs.
    """
    k = symmetric.shape[0]
    indices = torch.arange(k, device=symmetric.device)
    diagonal = symmetric[indices, indices]
    row = symmetric.mean(dim=1)
    trace = diagonal.mean(dim=0)
    global_mean = symmetric.mean(dim=(0, 1))
    identity = torch.eye(k, dtype=symmetric.dtype, device=symmetric.device).unsqueeze(-1)
    return (
        symmetric,
        identity * diagonal[:, None],
        diagonal[:, None] + diagonal[None, :],
        identity * row[:, None],
        row[:, None] + row[None, :],
        identity * trace,
        trace.expand_as(symmetric),
        identity * global_mean,
        global_mean.expand_as(symmetric),
    )


class OddMLP(nn.Module):
    """Odd map with bias-free content transforms and an even sigmoid gate."""

    def __init__(self, input_dim, hidden_dim, output_dim=PAIR_DIM):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim, bias=False)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, values):
        z = self.input(values)
        return self.output(z * torch.sigmoid(self.gate(z.abs())))


class PairLiftingWeights(nn.Module):
    """Heavy pair encoders for one order; norms belong to the stage state."""

    def __init__(self, order, hidden_dim=HIDDEN_DIM, pair_dim=PAIR_DIM):
        super().__init__()
        self.symmetric = nn.Sequential(
            nn.Linear(3 * hidden_dim + RELATIVE_GEOMETRY_DIMS[order][0], 2 * pair_dim), nn.SiLU(),
            nn.Linear(2 * pair_dim, pair_dim),
        )
        # Odd pair hidden width is d; symmetric pair hidden width is 2d.
        self.odd = OddMLP(hidden_dim + RELATIVE_GEOMETRY_DIMS[order][1], pair_dim, pair_dim)

    def forward(self, hidden, even_geometry, odd_geometry, state):
        left, right = hidden[..., :, None, :], hidden[..., None, :, :]
        plus = torch.cat((left + right, (left - right).abs(),
                          left * right, even_geometry), dim=-1)
        minus = torch.cat((left - right, odd_geometry), dim=-1)
        symmetric = state.symmetric_lift_norm(self.symmetric(plus))
        odd = state.odd_lift_norm(self.odd(minus))
        return enforce_parity(symmetric, odd)


class PaSTTensorLayerState(nn.Module):
    """Stage/order/layer-specific RMSNorm and per-channel LayerScale."""

    def __init__(self, hidden_dim=PAIR_DIM, layerscale_init=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.symmetric_norm = nn.RMSNorm(self.hidden_dim, eps=RMS_EPS)
        self.odd_norm = nn.RMSNorm(self.hidden_dim, eps=RMS_EPS)
        self.symmetric_scale = nn.Parameter(torch.full((self.hidden_dim,), layerscale_init))
        self.odd_scale = nn.Parameter(torch.full((self.hidden_dim,), layerscale_init))


class PaSTTensorLayerWeights(nn.Module):
    """One tensor layer's structural mixers, parity couplings and FFNs."""

    def __init__(self, hidden_dim=PAIR_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.symmetric_mixer = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False) for _ in range(9)
        ])
        self.odd_direct = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.odd_row = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.aa = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.sa = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.symmetric_ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim), nn.SiLU(), nn.Linear(2 * self.hidden_dim, self.hidden_dim),
        )
        # Both tensor FFNs use hidden width 2d; the odd pair encoder uses d.
        self.odd_ffn = OddMLP(self.hidden_dim, 2 * self.hidden_dim, self.hidden_dim)

    def forward(self, symmetric, odd, state, conformer_mask=None):
        from pastnet.model.past_batched import batched_tensor_layer
        if symmetric.ndim == 3:
            result = batched_tensor_layer(
                self, state, symmetric.unsqueeze(0), odd.unsqueeze(0),
                None if conformer_mask is None else conformer_mask.unsqueeze(0))
            return result[0][0], result[1][0]
        return batched_tensor_layer(self, state, symmetric, odd, conformer_mask)


class PaSTCollapse(nn.Module):
    """The four symmetric and one odd 2->1 contractions, all bias-free."""

    def __init__(self, hidden_dim=PAIR_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.symmetric = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False) for _ in range(4)
        ])
        self.odd = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

    def forward(self, symmetric, odd, conformer_mask=None):
        from pastnet.model.past_batched import batched_collapse
        if symmetric.ndim == 3:
            return batched_collapse(
                self, symmetric.unsqueeze(0), odd.unsqueeze(0),
                None if conformer_mask is None else conformer_mask.unsqueeze(0))[0]
        return batched_collapse(self, symmetric, odd, conformer_mask)


def _validate_dimensions(hidden_dim, pair_dim, num_tensor_layers):
    if (type(pair_dim) is not int or pair_dim not in (16, 32)
            or hidden_dim != HIDDEN_DIM or type(num_tensor_layers) is not int
            or num_tensor_layers not in (2, 3)):
        raise ValueError("PaST requires path width 32, pair width 16 or 32 and two or three tensor layers")


class PaSTWeights(nn.Module):
    """One order's heavy weights for an entry/core/exit parameter group.

    Share the same instance across middle stages for a given order. The configured
    tensor layers have distinct parameters; there are no norms/scales here.
    """

    def __init__(self, order, *, hidden_dim=HIDDEN_DIM, pair_dim=PAIR_DIM,
                 num_tensor_layers=TENSOR_LAYERS):
        super().__init__()
        if type(order) is not int or not 0 <= order <= 3:
            raise ValueError("PaST order must be an integer from 0 to 3")
        self.order = order
        self.hidden_dim = hidden_dim
        self.pair_dim = pair_dim
        _validate_dimensions(hidden_dim, pair_dim, num_tensor_layers)
        self.num_tensor_layers = num_tensor_layers
        self.lifting = PairLiftingWeights(order, hidden_dim, pair_dim)
        self.layers = nn.ModuleList([PaSTTensorLayerWeights(pair_dim) for _ in range(num_tensor_layers)])
        self.collapse = PaSTCollapse(pair_dim)
        self.gate = nn.Sequential(nn.Linear(hidden_dim + pair_dim, hidden_dim), nn.Sigmoid())
        self.up_projection = nn.Linear(pair_dim, hidden_dim, bias=False)


class PaSTStageState(nn.Module):
    """All normalization/scaling parameters for ONE stage and ONE path order."""

    def __init__(self, hidden_dim=HIDDEN_DIM, num_tensor_layers=TENSOR_LAYERS,
                 layerscale_init=0.1, pair_dim=PAIR_DIM):
        super().__init__()
        _validate_dimensions(hidden_dim, pair_dim, num_tensor_layers)
        self.hidden_dim = hidden_dim
        self.pair_dim = pair_dim
        self.content_norm = nn.RMSNorm(self.hidden_dim, eps=RMS_EPS)
        self.symmetric_lift_norm = nn.RMSNorm(pair_dim, eps=RMS_EPS)
        self.odd_lift_norm = nn.RMSNorm(pair_dim, eps=RMS_EPS)
        self.layers = nn.ModuleList([PaSTTensorLayerState(pair_dim, layerscale_init) for _ in range(num_tensor_layers)])
        self.residual_scale = nn.Parameter(torch.full((self.hidden_dim,), layerscale_init))


class PaSTBlock(nn.Module):
    """PaST update for [K,d] or vectorized independent paths [K,P,d].

    Each stage/order owns a separate block. Heavy weights are passed explicitly
    and must be registered with their owner (including in the optimizer).
    P1 uses lengths [K] and a fixed training-fitted bond_scale. P2/P3 use radians
    [K] and validity [K]. P0 ignores geometry and consumes content only.

    pair_tensors retains the intermediate pair-core interface for diagnostics.
    For a molecule with multiple paths/orders, sample_drop_multiplier must be
    called ONCE per stage/branch, then pass that SAME scalar to every forward.
    The default samples once for this standalone fixed-path call. Tensor core
    and self branches contain no stochastic operations. No post-normalization
    follows the outer residual.
    """

    def __init__(self, order, stage=1, *, hidden_dim=HIDDEN_DIM, pair_dim=PAIR_DIM, num_stages=3,
                 num_tensor_layers=TENSOR_LAYERS, stochastic_depth_max=0.0, layerscale_init=0.1):
        super().__init__()
        if type(order) is not int or not 0 <= order <= 3:
            raise ValueError("PaST order must be an integer from 0 to 3")
        self.hidden_dim = hidden_dim
        self.pair_dim = pair_dim
        _validate_dimensions(hidden_dim, pair_dim, num_tensor_layers)
        self.num_tensor_layers = num_tensor_layers
        self.order = order
        self.stage = stage
        self.drop_probability = stage_drop_probability(stage, num_stages, stochastic_depth_max)
        self.state = PaSTStageState(hidden_dim, num_tensor_layers, layerscale_init, pair_dim)

    def _validate_weights(self, weights):
        if (not isinstance(weights, PaSTWeights) or weights.order != self.order
                or weights.hidden_dim != self.hidden_dim
                or weights.pair_dim != self.pair_dim
                or len(weights.layers) != self.num_tensor_layers
                or len(self.state.layers) != self.num_tensor_layers):
            raise ValueError("PaST weights must match the block order, width and tensor depth")

    def _prepare_lift(self, hidden, *, weights, values=None, valid=None, bond_scale=None):
        if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 2
                or hidden.shape[1] != self.hidden_dim or not 1 <= hidden.shape[0] <= 5
                or not hidden.is_floating_point() or not torch.isfinite(hidden).all()):
            raise ValueError("PaST content must be finite floating [K,32], 1<=K<=5")
        self._validate_weights(weights)
        even, odd = relative_geometry(
            self.order, hidden, values=values, valid=valid, bond_scale=bond_scale)
        normalized = self.state.content_norm(hidden)
        return normalized, weights.lifting(normalized, even, odd, self.state)

    def lift(self, hidden, *, weights, values=None, valid=None, bond_scale=None):
        return self._prepare_lift(
            hidden, weights=weights, values=values, valid=valid, bond_scale=bond_scale)[1]

    def tensor_core(self, symmetric, odd, *, weights, conformer_mask=None):
        """Run the same tensor layers/stage states for either full or self input."""
        for layer, state in zip(weights.layers, self.state.layers):
            symmetric, odd = layer(symmetric, odd, state, conformer_mask)
        return symmetric, odd

    def pair_tensors(self, hidden, *, weights, values=None, valid=None, bond_scale=None,
                     return_lifted=False):
        """Return intermediate pair tensors without collapse or residual update."""
        lifted = self.lift(hidden, weights=weights, values=values,
                           valid=valid, bond_scale=bond_scale)
        result = self.tensor_core(*lifted, weights=weights)
        return (result, lifted) if return_lifted else result

    def sample_drop_multiplier(self, reference):
        """One scalar for a whole molecule/PaST branch; reuse across paths/orders."""
        multiplier = reference.new_ones(())
        if self.training and self.drop_probability:
            keep = 1 - self.drop_probability
            multiplier = multiplier.bernoulli_(keep) / keep
        return multiplier

    def forward(self, hidden, *, weights, values=None, valid=None, bond_scale=None,
                conformer_mask=None, relative=None, return_debug=False, drop_multiplier=None):
        from pastnet.model.past_batched import forward_paths
        if isinstance(hidden, torch.Tensor) and hidden.ndim == 2:
            # Shape adaptation only: single paths use the same packed operator.
            convert = lambda value: value[:, None] if isinstance(value, torch.Tensor) else value
            if relative is not None:
                relative = tuple(value.unsqueeze(0) if value.ndim == 3 else value
                                 for value in relative)
            result = forward_paths(
                self, hidden[:, None], weights=weights, values=convert(values), valid=convert(valid),
                bond_scale=bond_scale, conformer_mask=convert(conformer_mask), relative=relative,
                return_debug=return_debug, drop_multiplier=drop_multiplier)
            if not return_debug:
                return result[:, 0]
            output, debug = result
            for key in ("normalized_hidden", "m_full", "m_self", "m_cross", "gate", "message", "delta"):
                debug[key] = debug[key][:, 0]
            for key in ("lifted", "pair"):
                debug[key] = tuple(value[0] for value in debug[key])
            return output[:, 0], debug
        return forward_paths(
            self, hidden, weights=weights, values=values, valid=valid,
            bond_scale=bond_scale, conformer_mask=conformer_mask, relative=relative,
            return_debug=return_debug, drop_multiplier=drop_multiplier)
