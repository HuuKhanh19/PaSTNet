"""PaSTNet's three Geo-IH -> PaST stages on a masked shared path batch."""

import torch
from torch import nn

from pastnet.model.geo_ih import GeometryEncoderBank, GeoIHBlock, GeoIHWeights, absolute_geometry_inputs
from pastnet.model.input import SpatialToPathAdapter
from pastnet.model.config import validate_architecture
from pastnet.model.past import PaSTBlock, PaSTWeights
from pastnet.model.schnet import SchNetAtomEncoder
from pastnet.model.batch import PaSTNetBatch, pack_molecules, spatial_to_paths, spatial_bypass

STAGE_GROUPS = ("entry", "core", "exit")


class GeoPaSTWeights(nn.Module):
    def __init__(self, hidden_dim=32, num_tensor_layers=2):
        super().__init__()
        self.geo_ih = GeoIHWeights(hidden_dim=hidden_dim)
        self.past = nn.ModuleList([PaSTWeights(order, hidden_dim=hidden_dim,
            num_tensor_layers=num_tensor_layers) for order in range(4)])


class GeoPaSTStage(nn.Module):
    def __init__(self, stage, *, hidden_dim=32, num_stages=3, num_tensor_layers=2,
                 stochastic_depth_max=0.0, layerscale_init=0.1):
        super().__init__()
        self.stage = stage
        self.group = STAGE_GROUPS[stage-1]
        options = dict(hidden_dim=hidden_dim, num_stages=num_stages,
                       stochastic_depth_max=stochastic_depth_max, layerscale_init=layerscale_init)
        self.geo_ih = GeoIHBlock(stage=stage, **options)
        self.past = nn.ModuleList([PaSTBlock(order, stage=stage,
            num_tensor_layers=num_tensor_layers, **options) for order in range(4)])

    def forward(self, hidden, batch, *, weights, geometry_encoders,
                geometry_embeddings, return_debug=False):
        after_geo = self.geo_ih(hidden, batch.topology, absolute_geometry_inputs(batch),
            weights=weights.geo_ih, geometry_encoders=geometry_encoders,
            geometry_embeddings=geometry_embeddings, valid_masks=batch.valid_masks,
            relation_indices=batch.relation_indices)
        result, diagnostics = [], []
        for order in range(4):
            value = self.past[order](after_geo[order], weights=weights.past[order],
                conformer_mask=batch.valid_masks[order], relative=batch.relative_geometry[order],
                return_debug=return_debug)
            if return_debug:
                value, debug = value
                diagnostics.append(debug)
            result.append(value)
        result = tuple(result)
        if return_debug:
            return result, dict(stage=self.stage, group=self.group,
                after_geo_ih=after_geo, after_past=result, past=diagnostics)
        return result


class PaSTNetBackbone(nn.Module):
    def __init__(self, *, bond_scale, reversal_invariant=True, hidden_dim=32,
                 num_stages=3, num_tensor_layers=2, stochastic_depth_max=0.0,
                 layerscale_init=0.1, geometry_strength_init=0.1):
        super().__init__()
        validate_architecture(hidden_dim, num_stages, num_tensor_layers,
                              stochastic_depth_max, layerscale_init, geometry_strength_init)
        if reversal_invariant is not True:
            raise ValueError("PaSTNet requires reversal-consistent path encoding")
        self.hidden_dim, self.num_stages, self.num_tensor_layers = 32, 3, 2
        self.stage_groups = STAGE_GROUPS
        scale = torch.as_tensor(bond_scale, dtype=torch.float64)
        if scale.ndim != 0 or scale.requires_grad or not torch.isfinite(scale) or scale < 0:
            raise ValueError("PaSTNet needs a fixed finite nonnegative training bond scale")
        self.register_buffer("bond_scale", scale.detach().clone())
        self.spatial = SchNetAtomEncoder()
        self.projection = SpatialToPathAdapter()
        self.geometry_encoders = GeometryEncoderBank(hidden_dim=32,
            geometry_strength_init=geometry_strength_init)
        self.heavy_groups = nn.ModuleDict({name: GeoPaSTWeights() for name in STAGE_GROUPS})
        self.stages = nn.ModuleList([GeoPaSTStage(stage, stochastic_depth_max=stochastic_depth_max,
            layerscale_init=layerscale_init) for stage in (1, 2, 3)])

    @property
    def bond_scale_value(self):
        # Kept as a buffer in checkpoints; memoize the CPU scalar to avoid a
        # device synchronization for every molecule/stage.
        if not hasattr(self, "_bond_scale_value"):
            self._bond_scale_value = self.bond_scale.item()
        return self._bond_scale_value

    def _load_from_state_dict(self, *args, **kwargs):
        self.__dict__.pop("_bond_scale_value", None)
        return super()._load_from_state_dict(*args, **kwargs)

    def weights_for_stage(self, stage):
        if type(stage) is not int or not 1 <= stage <= 3:
            raise ValueError("PaSTNet has exactly three stages")
        return self.heavy_groups[STAGE_GROUPS[stage-1]]

    def forward(self, batch, *, return_debug=False, execution="vectorized"):
        if execution != "vectorized":
            raise ValueError("PaSTNet uses vectorized execution exclusively")
        if not isinstance(batch, PaSTNetBatch):
            batch = pack_molecules((batch,), bond_scale=self.bond_scale_value)
        atoms = self.spatial(batch.atomic_numbers, batch.spatial_edges,
                             batch.spatial_rbf, batch.spatial_envelope)
        hidden = self.projection(batch.features, spatial_to_paths(atoms, batch),
                                 batch.topology, valid_masks=batch.valid_masks)
        initial = hidden
        geometry_inputs = tuple(torch.where(mask[..., None], raw, torch.zeros_like(raw))
                                for raw, mask in zip(absolute_geometry_inputs(batch), batch.valid_masks[1:]))
        embeddings = self.geometry_encoders(geometry_inputs)
        trace = []
        for stage in self.stages:
            value = stage(hidden, batch, weights=self.weights_for_stage(stage.stage),
                          geometry_encoders=self.geometry_encoders, geometry_embeddings=embeddings,
                          return_debug=return_debug)
            if return_debug:
                hidden, details = value
                trace.append(details)
            else:
                hidden = value
        bypass = spatial_bypass(atoms, batch)
        if return_debug:
            return hidden, bypass, dict(initial=initial, stages=trace, spatial_atoms=atoms,
                                        valid_masks=batch.valid_masks)
        return hidden, bypass

    def parameter_counts(self):
        count = lambda module: sum(p.numel() for p in module.parameters())
        return dict(spatial=count(self.spatial), projection=count(self.projection),
                    geometry_encoders=count(self.geometry_encoders),
                    heavy_groups=count(self.heavy_groups), stages=count(self.stages), total=count(self))
