"""Table 2 same-path attention control.

Construct P0 first, replace only its per-stage/per-order PaST modules inside
fork_rng, and retain both all other initial weights and the reference RNG state.
The attention control uses ordinary self-attention (including self) plus FFN,
with the identical cached relative geometry as a learned per-head logit bias.
"""

import torch
from torch import nn

from pastnet.model.network import PaSTNet
from pastnet.model.past import RMS_EPS
from pastnet.model.past_batched import _relative_inputs
from pastnet.geometry.relative import RELATIVE_GEOMETRY_DIMS


ATTENTION_HEADS = 4
ATTENTION_LAYERS = 2
# Closest integer FFN widths to each corresponding P0 PaST order's capacity.
ATTENTION_FFN_DIMS = (96, 98, 96, 97)


class SamePathAttentionWeights(nn.Module):
    def __init__(self, order, ffn_dim):
        super().__init__()
        self.order = order
        self.hidden_dim = 32
        self.ffn_dim = ffn_dim
        geometry_dim = sum(RELATIVE_GEOMETRY_DIMS[order])
        self.layers = nn.ModuleList()
        for _ in range(ATTENTION_LAYERS):
            layer = nn.ModuleDict({
                "attention": nn.MultiheadAttention(32, ATTENTION_HEADS, dropout=0.0,
                                                   batch_first=True),
                "ffn": nn.Sequential(nn.Linear(32, ffn_dim), nn.SiLU(), nn.Linear(ffn_dim, 32)),
            })
            if geometry_dim:
                layer["geometry_bias"] = nn.Linear(geometry_dim, ATTENTION_HEADS, bias=False)
            self.layers.append(layer)


class SamePathAttentionBlock(nn.Module):
    """Attention across K only: each of the P paths is an independent batch item."""

    def __init__(self, order, stage=1, layerscale_init=0.1):
        super().__init__()
        self.order, self.stage, self.hidden_dim = order, stage, 32
        self.layers = nn.ModuleList([
            nn.ModuleDict({"attention_norm": nn.RMSNorm(32, eps=RMS_EPS),
                           "ffn_norm": nn.RMSNorm(32, eps=RMS_EPS),
                           "scales": nn.ParameterDict({
                               "attention": nn.Parameter(torch.full((32,), layerscale_init)),
                               "ffn": nn.Parameter(torch.full((32,), layerscale_init)),
                           })}) for _ in range(ATTENTION_LAYERS)
        ])

    def forward(self, hidden, *, weights, conformer_mask=None, relative=None,
                values=None, valid=None, bond_scale=None, return_debug=False):
        if (hidden.ndim != 3 or hidden.shape[-1] != 32
                or not 1 <= hidden.shape[0] <= 5 or not hidden.is_floating_point()):
            raise ValueError("Same-path attention expects floating [K,P,32], 1<=K<=5")
        if not isinstance(weights, SamePathAttentionWeights) or weights.order != self.order:
            raise ValueError("Attention weights must match path order")
        k, paths = hidden.shape[:2]
        if conformer_mask is None:
            conformer_mask = torch.ones(k, paths, dtype=torch.bool, device=hidden.device)
        if conformer_mask.dtype != torch.bool or conformer_mask.shape != (k, paths):
            raise ValueError("Conformer mask must be boolean [K,P]")
        if (conformer_mask.sum(0) < 1).any():
            raise ValueError("Every path must have a valid conformer")
        clean = torch.where(conformer_mask[..., None], hidden, torch.zeros_like(hidden))
        if not torch.isfinite(clean).all():
            raise ValueError("Valid attention inputs must be finite")
        if paths == 0:
            result = clean.to(torch.promote_types(hidden.dtype, self.layers[0]["scales"]["ffn"].dtype))
            return (result, {"attention_layers": [], "delta": result}) if return_debug else result
        geometry = torch.cat(_relative_inputs(
            self, clean, relative, values=values, valid=valid,
            bond_scale=bond_scale, conformer_mask=conformer_mask), dim=-1)
        mask = conformer_mask.T
        x = clean.transpose(0, 1)
        trace = []
        for layer, state in zip(weights.layers, self.layers):
            normalized = state["attention_norm"](x)
            # [P,heads,query K,key K], never a contraction across paths.
            if "geometry_bias" in layer:
                bias = layer["geometry_bias"](geometry).permute(0, 3, 1, 2)
            else:
                bias = x.new_zeros(paths, ATTENTION_HEADS, k, k)
            bias = bias.masked_fill(~mask[:, None, None, :], float("-inf"))
            message, probabilities = layer["attention"](
                normalized, normalized, normalized,
                attn_mask=bias.reshape(paths * ATTENTION_HEADS, k, k),
                need_weights=return_debug, average_attn_weights=False)
            x = x + state["scales"]["attention"] * message
            x = x + state["scales"]["ffn"] * layer["ffn"](state["ffn_norm"](x))
            x = torch.where(mask[..., None], x, torch.zeros_like(x))
            if return_debug:
                trace.append(probabilities)
        result = x.transpose(0, 1)
        if return_debug:
            return result, dict(attention_layers=trace, delta=result - clean)
        return result


class PaSTNetAblation(PaSTNet):
    def __init__(self, *, variant, **options):
        if variant != "same_path_attention":
            raise ValueError(f"Unknown PaSTNet ablation: {variant}")
        depth = 2
        requested_depth = options.pop("num_tensor_layers", depth)
        if type(requested_depth) is not int or requested_depth != depth:
            raise ValueError(f"{variant} requires num_tensor_layers={depth}")
        super().__init__(num_tensor_layers=2, **options)
        self.variant = variant
        self.reference_parameter_count = sum(p.numel() for p in self.parameters())
        self.backbone.num_tensor_layers = 0
        scale = options.get("layerscale_init", 0.1)
        # All new modules are constructed on CPU; leave the loader/global RNG
        # at exactly the state produced by the original model constructor.
        with torch.random.fork_rng(devices=[]):
            for group in self.backbone.heavy_groups.values():
                group.past = nn.ModuleList([
                    SamePathAttentionWeights(order, ATTENTION_FFN_DIMS[order])
                    for order in range(4)])
            for stage in self.backbone.stages:
                stage.past = nn.ModuleList([
                    SamePathAttentionBlock(order, stage.stage, layerscale_init=scale)
                    for order in range(4)])

    def parameter_counts(self):
        counts = super().parameter_counts()
        counts["cross_conformer"] = sum(
            p.numel() for name, p in self.named_parameters() if ".past." in name)
        counts["reference_total"] = self.reference_parameter_count
        counts["delta_from_reference"] = counts["total"] - self.reference_parameter_count
        return counts
