"""Table 2: exactly capacity-matched pair-local PaST, unchanged depth.

Widen only the two existing tensor FFNs. Per layer, symmetric width 32->72
adds 33*40=1320 weights/biases; OddMLP width 32->39 adds 728. Across
3 stages * 4 orders * 2 layers this restores exactly 49152 parameters.
The original pair-local forward retains B1/B2/V1 and parity products only.
The final 2->1 contraction still aggregates across conformers.
"""
import torch
from torch import nn
from pastnet.model.past import OddMLP
from pastnet.model.pastnet_no_pair_context import PairLocalTensorLayer, PaSTNetNoPairContext

SYMMETRIC_FFN_WIDTH = 72
ODD_FFN_WIDTH = 39


def widen_pair_local_ffns(layer):
    if not isinstance(layer, PairLocalTensorLayer) or layer.hidden_dim != 16:
        raise ValueError('Requires the original pair-local core with pair width16')
    layer.symmetric_ffn = nn.Sequential(nn.Linear(16, SYMMETRIC_FFN_WIDTH), nn.SiLU(),
                                        nn.Linear(SYMMETRIC_FFN_WIDTH, 16))
    layer.odd_ffn = OddMLP(16, ODD_FFN_WIDTH, 16)
    return layer


class PaSTNetPairLocalMatched(PaSTNetNoPairContext):
    def __init__(self, **options):
        super().__init__(**options)
        if self.reference_parameter_count != 667548:
            raise ValueError('E2 requires the frozen Full PaST architecture')
        self.original_pair_local_count = sum(p.numel() for p in self.parameters())
        # Leave all other weights AND the global loader RNG at the P0 state.
        with torch.random.fork_rng(devices=[]):
            for group in self.backbone.heavy_groups.values():
                for weights in group.past:
                    for layer in weights.layers:
                        widen_pair_local_ffns(layer)
        if sum(p.numel() for p in self.parameters()) != self.reference_parameter_count:
            raise RuntimeError('Exact parameter matching failed')

    def parameter_counts(self):
        counts = super().parameter_counts()
        counts.update(original_pair_local_total=self.original_pair_local_count,
            removed_context_parameters=49152, added_pair_local_ffn_parameters=49152,
            delta_from_reference=counts['total']-self.reference_parameter_count)
        return counts
