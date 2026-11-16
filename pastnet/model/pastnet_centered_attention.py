"""E1/A2: the Table 2 attention block with a shared singleton counterfactual.

For the original two-layer block B(h)=h+F(h), return h+B(h)-B([h_c]).
Thus the outer map is g(h, delta)=delta, with no additional parameters.
The complete two-layer attention/FFN stack, RMSNorms, LayerScales and geometry
biases are evaluated in both branches. Singleton calls are batched over valid
(conformer, path) tokens; there is no diagonal-logit masking approximation.
"""

import torch
from torch import nn

from pastnet.model.past_batched import _relative_inputs
from pastnet.model.pastnet_ablation import PaSTNetAblation, SamePathAttentionBlock


class CenteredSamePathAttentionBlock(SamePathAttentionBlock):
    def forward(self, hidden, *, weights, conformer_mask=None, relative=None,
                values=None, valid=None, bond_scale=None, return_debug=False):
        options = dict(weights=weights, conformer_mask=conformer_mask, relative=relative,
                       values=values, valid=valid, bond_scale=bond_scale)
        # Use the original implementation, including its input validation.
        full = super().forward(hidden, return_debug=return_debug, **options)
        full_output, full_debug = full if return_debug else (full, None)
        k, paths, width = hidden.shape
        if conformer_mask is None:
            conformer_mask = torch.ones(k, paths, dtype=torch.bool, device=hidden.device)
        clean = torch.where(conformer_mask[..., None], hidden, torch.zeros_like(hidden))
        if paths == 0:
            return (full_output, dict(full_output=full_output, singleton_output=full_output,
                                     delta=full_output)) if return_debug else full_output

        geometry = _relative_inputs(self, clean, relative, values=values, valid=valid,
                                    bond_scale=bond_scale, conformer_mask=conformer_mask)
        # Noncontiguous masks are supported. No all-padding singleton is evaluated.
        conformers, path_ids = conformer_mask.nonzero(as_tuple=True)
        singleton_hidden = clean[conformers, path_ids].unsqueeze(0)
        singleton_relative = tuple(g[path_ids, conformers, conformers][:, None, None, :]
                                   for g in geometry)
        singletons = super().forward(singleton_hidden, weights=weights,
                                     relative=singleton_relative, return_debug=return_debug)
        singleton_values, singleton_debug = singletons if return_debug else (singletons, None)
        singleton_output = torch.zeros_like(full_output).index_put(
            (conformers, path_ids), singleton_values.squeeze(0))
        delta = full_output - singleton_output
        # Mathematically zero for an effective K=1. Enforce bitwise identity even
        # when packed full/singleton CUDA kernels have different rounding.
        active = conformer_mask & (conformer_mask.sum(0, keepdim=True) > 1)
        delta = torch.where(active[..., None], delta, torch.zeros_like(delta))
        output = clean + delta
        if return_debug:
            return output, dict(full_output=full_output, singleton_output=singleton_output,
                                full=full_debug, singleton=singleton_debug, delta=delta)
        return output


class PaSTNetCenteredAttention(PaSTNetAblation):
    """Exactly A1's initialization, state-dict layout and parameter count."""

    def __init__(self, **options):
        super().__init__(variant="same_path_attention", **options)
        for stage in self.backbone.stages:
            blocks = []
            for original in stage.past:
                block = CenteredSamePathAttentionBlock(original.order, original.stage)
                block.load_state_dict(original.state_dict(), strict=True)
                blocks.append(block)
            stage.past = nn.ModuleList(blocks)
        self.variant = "centered_same_path_attention"
