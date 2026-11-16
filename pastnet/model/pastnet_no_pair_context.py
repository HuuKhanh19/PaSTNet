"""Table 2 intervention: remove communication BETWEEN pair states in 2->2.

Keep B1(S)=S and B2(S)=delta_ij*S_ij, and V1(A)=A. B2 is local
to the diagonal pair itself. Remove B3..B9 and odd row-difference V2.
Pair lifting, both nonlinear tensor layers, parity coupling, 2->1 collapse,
and the shared full-minus-singleton residual remain exactly as in PaSTNet.
"""

import torch
from torch import nn

from pastnet.model.network import PaSTNet
from pastnet.model.past_batched import _masked_pairs, batched_enforce_parity


class PairLocalTensorLayer(nn.Module):
    """Reuse surviving P0 parameters without consuming any random numbers."""

    def __init__(self, reference):
        super().__init__()
        self.hidden_dim = reference.hidden_dim
        self.symmetric_mixer = nn.ModuleList(list(reference.symmetric_mixer[:2]))
        for name in ("odd_direct", "aa", "sa", "symmetric_ffn", "odd_ffn"):
            setattr(self, name, getattr(reference, name))

    def forward(self, symmetric, odd, state, conformer_mask=None):
        single = symmetric.ndim == 3
        if single:
            symmetric, odd = symmetric.unsqueeze(0), odd.unsqueeze(0)
            if conformer_mask is not None:
                conformer_mask = conformer_mask.unsqueeze(0)
        symmetric = _masked_pairs(symmetric, conformer_mask)
        odd = _masked_pairs(odd, conformer_mask)
        identity = torch.eye(symmetric.shape[1], device=symmetric.device,
                             dtype=symmetric.dtype)[None, :, :, None]
        plus = (self.symmetric_mixer[0](symmetric)
                + self.symmetric_mixer[1](identity * symmetric)
                + self.aa(odd * odd))
        minus = self.odd_direct(odd) + self.sa(symmetric * odd)
        symmetric = symmetric + state.symmetric_scale * self.symmetric_ffn(
            state.symmetric_norm(plus))
        odd = odd + state.odd_scale * self.odd_ffn(state.odd_norm(minus))
        symmetric, odd = batched_enforce_parity(symmetric, odd)
        result = (_masked_pairs(symmetric, conformer_mask),
                  _masked_pairs(odd, conformer_mask))
        return (result[0][0], result[1][0]) if single else result


class PaSTNetNoPairContext(PaSTNet):
    def __init__(self, **options):
        super().__init__(**options)
        self.reference_parameter_count = sum(p.numel() for p in self.parameters())
        for group in self.backbone.heavy_groups.values():
            for weights in group.past:
                weights.layers = nn.ModuleList([
                    PairLocalTensorLayer(layer) for layer in weights.layers])

    def parameter_counts(self):
        counts = super().parameter_counts()
        counts.update(reference_total=self.reference_parameter_count,
                      removed_context_parameters=self.reference_parameter_count - counts["total"])
        return counts
