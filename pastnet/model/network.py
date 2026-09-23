"""PaSTNet for molecular property prediction from conformer ensembles."""

from torch import nn
from pastnet.model.backbone import PaSTNetBackbone
from pastnet.model.readout import GeoPaSTReadout
from pastnet.model.batch import forward_molecules


class PaSTNet(nn.Module):
    def __init__(self, *, bond_scale, reversal_invariant=True, hidden_dim=32,
                 num_stages=3, num_tensor_layers=2, stochastic_depth_max=0.0,
                 layerscale_init=0.1, geometry_strength_init=0.1):
        super().__init__()
        self.backbone = PaSTNetBackbone(bond_scale=bond_scale, reversal_invariant=reversal_invariant,
            hidden_dim=hidden_dim, num_stages=num_stages, num_tensor_layers=num_tensor_layers,
            stochastic_depth_max=stochastic_depth_max, layerscale_init=layerscale_init,
            geometry_strength_init=geometry_strength_init)
        self.post_stage = GeoPaSTReadout(hidden_dim=hidden_dim)

    def forward_packed(self, batch, *, return_debug=False):
        encoded = self.backbone(batch, return_debug=return_debug)
        if return_debug:
            hidden, bypass, backbone_debug = encoded
        else:
            hidden, bypass = encoded
        output = self.post_stage(hidden, batch.topology, bypass, valid_masks=batch.valid_masks,
            path_batch=batch.path_batch, num_molecules=batch.num_molecules,
            relation_indices=batch.relation_indices, return_debug=return_debug)
        if return_debug:
            prediction, post_debug = output
            return prediction, dict(backbone=backbone_debug, stage3=hidden,
                                    post_stage=post_debug, batch=batch)
        return output

    def forward(self, hrge, *, return_debug=False, execution="vectorized"):
        if execution != "vectorized":
            raise ValueError("PaSTNet uses vectorized execution exclusively")
        multiple = isinstance(hrge, (tuple, list))
        result = forward_molecules(self, hrge if multiple else (hrge,), return_debug=return_debug)
        if return_debug:
            predictions, debug = result
            return (predictions if multiple else predictions[0]), debug
        return result if multiple else result[0]

    def parameter_counts(self):
        count = lambda module: sum(p.numel() for p in module.parameters())
        return dict(backbone=count(self.backbone), spatial=count(self.backbone.spatial),
                    adapters=count(self.backbone.projection), pooling=count(self.post_stage.pooling),
                    refinement=count(self.post_stage.refinement), readout=count(self.post_stage.readout),
                    head=count(self.post_stage.head), total=count(self))
