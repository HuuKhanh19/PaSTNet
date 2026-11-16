"""A1: the current PaSTNet with identity cross-conformer residuals.

Construct the full model first, then remove only PaST modules. This preserves
the exact initialization AND post-construction RNG state of retained weights
relative to the reference model, without counting unused PaST parameters.
"""

from torch import nn

from pastnet.model.network import PaSTNet
from pastnet.model.geo_ih import absolute_geometry_inputs


class NoPaSTStage(nn.Module):
    def __init__(self, reference_stage):
        super().__init__()
        self.stage = reference_stage.stage
        self.group = reference_stage.group
        self.geo_ih = reference_stage.geo_ih

    def forward(self, hidden, batch, *, weights, geometry_encoders,
                geometry_embeddings, return_debug=False):
        after_geo = self.geo_ih(
            hidden, batch.topology, absolute_geometry_inputs(batch),
            weights=weights.geo_ih, geometry_encoders=geometry_encoders,
            geometry_embeddings=geometry_embeddings, valid_masks=batch.valid_masks,
            relation_indices=batch.relation_indices,
        )
        if return_debug:
            return after_geo, dict(stage=self.stage, group=self.group,
                after_geo_ih=after_geo, after_past=after_geo, past=[],
                cross_conformer_residual="identity_A1")
        return after_geo


class PaSTNetNoPaST(PaSTNet):
    """Keep all intra-conformer and final ensemble aggregation operations."""

    def __init__(self, **options):
        super().__init__(**options)
        self.reference_parameter_count = sum(p.numel() for p in self.parameters())
        self.backbone.stages = nn.ModuleList(
            NoPaSTStage(stage) for stage in self.backbone.stages)
        for weights in self.backbone.heavy_groups.values():
            del weights.past
        self.removed_past_parameter_count = (
            self.reference_parameter_count - sum(p.numel() for p in self.parameters()))
