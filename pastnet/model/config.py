"""Frozen PaSTNet v1 architecture; incompatible historical capacities fail."""

import math


def validate_architecture(hidden_dim=32, num_stages=3, num_tensor_layers=2,
                          stochastic_depth_max=0.0, layerscale_init=0.1,
                          geometry_strength_init=0.1):
    for name, value, expected in (("hidden_dim", hidden_dim, 32),
                                  ("num_stages", num_stages, 3),
                                  ("num_tensor_layers", num_tensor_layers, 2)):
        if type(value) is not int or value != expected:
            raise ValueError(f"PaSTNet v1 fixes {name}={expected}")
    for name, value in (("stochastic_depth_max", stochastic_depth_max),
                        ("layerscale_init", layerscale_init),
                        ("geometry_strength_init", geometry_strength_init)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if stochastic_depth_max != 0:
        raise ValueError("PaSTNet v1 fixes stochastic_depth_max=0.0")


def stage_drop_probability(stage, num_stages=3, stochastic_depth_max=0.0):
    validate_architecture(num_stages=num_stages, stochastic_depth_max=stochastic_depth_max)
    if type(stage) is not int or not 1 <= stage <= num_stages:
        raise ValueError(f"stage must be an integer from 1 to {num_stages}")
    return stochastic_depth_max * (stage - 1) / (num_stages - 1)
