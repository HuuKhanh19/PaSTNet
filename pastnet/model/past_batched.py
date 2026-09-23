"""Masked, vectorized PaST over independent paths.

Public path states are [K,P,32]; pair states are [P,K,K,16]. Only the
conformer axes are contracted. The singleton counterfactual uses the same
learned core on [P*K,1,1,16], including the same stage normalization/scales.
"""

import torch

from pastnet.geometry.relative import RELATIVE_BOND_EPS, RELATIVE_GEOMETRY_DIMS


def batched_enforce_parity(symmetric, odd):
    """Project only the two conformer axes of [P,K,K,d]."""
    return ((symmetric + symmetric.transpose(1, 2)) * 0.5,
            (odd - odd.transpose(1, 2)) * 0.5)


def _pair_mask(conformer_mask):
    return (conformer_mask[:, :, None] & conformer_mask[:, None, :]).unsqueeze(-1)


def _masked_pairs(values, conformer_mask):
    if conformer_mask is None:
        return values
    return torch.where(_pair_mask(conformer_mask), values, torch.zeros_like(values))


def _pair_statistics(values, conformer_mask=None):
    """Diagonal, row, trace and global means over actual valid conformers.

    Sensitive sums/divisions run in FP32 (or FP64 for reference execution).
    Mask before arithmetic so padded NaNs cannot contaminate values/gradients.
    Entirely masked singleton slots are internal padding and contract to zero.
    """
    clean = _masked_pairs(values, conformer_mask)
    calculation = clean if clean.dtype == torch.float64 else clean.float()
    k = clean.shape[1]
    indices = torch.arange(k, device=clean.device)
    diagonal = calculation[:, indices, indices]
    if conformer_mask is None:
        count = calculation.new_full((clean.shape[0], 1), k)
    else:
        count = conformer_mask.sum(dim=1, keepdim=True).to(calculation.dtype).clamp_min(1)
    row = calculation.sum(dim=2) / count[:, :, None]
    trace = diagonal.sum(dim=1) / count
    global_mean = calculation.sum(dim=(1, 2)) / count.square()
    return tuple(value.to(values.dtype) for value in (diagonal, row, trace, global_mean))


def batched_relative_geometry(order, hidden, *, values=None, valid=None,
                              bond_scale=None, conformer_mask=None):
    """Raw relative geometry [P,K,K,g] from [K,P,d], with padding excluded.

    Geometry definitions are unchanged: eight bond RBF channels, or angular
    cosine/sine harmonics with a pair-valid indicator in the even branch.
    Invalid geometry is replaced before arithmetic, including NaN placeholders.
    """
    if type(order) is not int or not 0 <= order <= 3:
        raise ValueError("PaST path order must be 0, 1, 2 or 3")
    if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 3
            or hidden.shape[-1] < 1 or not 1 <= hidden.shape[0] <= 5
            or not hidden.is_floating_point()):
        raise ValueError("Batched PaST content must be floating [K,P,d], 1<=K<=5")
    k, paths = hidden.shape[:2]
    if order == 0:
        return hidden.new_empty(paths, k, k, 0), hidden.new_empty(paths, k, k, 0)
    if conformer_mask is None:
        conformer_mask = torch.ones(k, paths, dtype=torch.bool, device=hidden.device)
    elif conformer_mask.dtype != torch.bool or conformer_mask.shape != (k, paths):
        raise ValueError("Conformer mask must be boolean [K,P]")

    dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
    if values is None:
        raise ValueError(f"P{order} requires geometry values [K,P]")
    values = torch.as_tensor(values, dtype=dtype, device=hidden.device)
    if values.shape != (k, paths):
        raise ValueError("Batched relative geometry values must have shape [K,P]")
    if order == 1:
        if not torch.isfinite(values[conformer_mask]).all() or (values[conformer_mask] < 0).any():
            raise ValueError("Valid bond lengths must be finite and nonnegative")
        if bond_scale is None:
            raise ValueError("P1 requires a fitted training-only bond scale")
        scale = torch.as_tensor(bond_scale, dtype=dtype, device=hidden.device)
        if (scale.ndim != 0 or scale.requires_grad
                or not torch.isfinite(scale) or scale < 0):
            raise ValueError("Bond scale must be a fixed finite nonnegative scalar")
        values = torch.where(conformer_mask, values, torch.zeros_like(values)).transpose(0, 1)
        signed_u = (values[:, :, None] - values[:, None, :]) / (scale + RELATIVE_BOND_EPS)
        centers = torch.arange(8, dtype=dtype, device=hidden.device) * (2.0 / 7.0)
        even = torch.exp(-((7.0 / 2.0) ** 2)
                         * (signed_u.abs().unsqueeze(-1) - centers).square())
        odd = signed_u.unsqueeze(-1) * even
        mask = _pair_mask(conformer_mask.transpose(0, 1))
        return (torch.where(mask, even, torch.zeros_like(even)).to(hidden.dtype),
                torch.where(mask, odd, torch.zeros_like(odd)).to(hidden.dtype))

    if valid is None:
        raise ValueError("Angular geometry requires a boolean validity mask [K,P]")
    valid = torch.as_tensor(valid, device=hidden.device)
    if valid.dtype != torch.bool or valid.shape != (k, paths):
        raise ValueError("Batched angular validity must be boolean [K,P]")
    valid = valid & conformer_mask
    if not torch.isfinite(values[valid]).all():
        raise ValueError("Valid angular geometry must be finite")
    safe_values = torch.where(valid, values, torch.zeros_like(values)).transpose(0, 1)
    delta = safe_values[:, :, None] - safe_values[:, None, :]
    mask = _pair_mask(valid.transpose(0, 1))
    harmonics = torch.arange(1, order + 1, dtype=dtype, device=hidden.device)
    phase = delta.unsqueeze(-1) * harmonics
    even = torch.cat((torch.where(mask, phase.cos(), torch.zeros_like(phase)),
                      mask.to(dtype)), dim=-1)
    odd = torch.where(mask, phase.sin(), torch.zeros_like(phase))
    return even.to(hidden.dtype), odd.to(hidden.dtype)


def batched_symmetric_basis(symmetric, conformer_mask=None):
    """All nine symmetric 2->2 maps, with actual-K masked contractions."""
    symmetric = _masked_pairs(symmetric, conformer_mask)
    k = symmetric.shape[1]
    diagonal, row, trace, global_mean = _pair_statistics(symmetric, conformer_mask)
    identity = torch.eye(k, dtype=symmetric.dtype,
                         device=symmetric.device)[None, :, :, None]
    maps = (
        symmetric,
        identity * diagonal[:, :, None],
        diagonal[:, :, None] + diagonal[:, None, :],
        identity * row[:, :, None],
        row[:, :, None] + row[:, None, :],
        identity * trace[:, None, None],
        trace[:, None, None].expand_as(symmetric),
        identity * global_mean[:, None, None],
        global_mean[:, None, None].expand_as(symmetric),
    )
    return tuple(_masked_pairs(value, conformer_mask) for value in maps)


def batched_tensor_layer(layer, state, symmetric, odd, conformer_mask=None):
    """One shared structural/parity layer; no loop over paths or conformers."""
    symmetric = _masked_pairs(symmetric, conformer_mask)
    odd = _masked_pairs(odd, conformer_mask)
    plus = sum(linear(values) for linear, values in
               zip(layer.symmetric_mixer, batched_symmetric_basis(symmetric, conformer_mask)))
    row = _pair_statistics(odd, conformer_mask)[1]
    minus = layer.odd_direct(odd) + layer.odd_row(row[:, :, None] - row[:, None, :])
    plus = plus + layer.aa(odd * odd)
    minus = minus + layer.sa(symmetric * odd)
    updated_symmetric = symmetric + state.symmetric_scale * layer.symmetric_ffn(
        state.symmetric_norm(plus))
    updated_odd = odd + state.odd_scale * layer.odd_ffn(state.odd_norm(minus))
    symmetric, odd = batched_enforce_parity(updated_symmetric, updated_odd)
    return _masked_pairs(symmetric, conformer_mask), _masked_pairs(odd, conformer_mask)


def batched_tensor_core(block, symmetric, odd, *, weights, conformer_mask=None):
    """Full and singleton evaluation share modules, stage states and autograd."""
    return block.tensor_core(symmetric, odd, weights=weights, conformer_mask=conformer_mask)


def batched_collapse(collapse, symmetric, odd, conformer_mask=None):
    """Four symmetric and one odd 2->1 contractions, returning [P,K,16]."""
    diagonal, row, trace, global_mean = _pair_statistics(symmetric, conformer_mask)
    plus = sum(linear(values) for linear, values in zip(
        collapse.symmetric, (diagonal, row, trace[:, None], global_mean[:, None])))
    result = plus + collapse.odd(_pair_statistics(odd, conformer_mask)[1])
    if conformer_mask is not None:
        result = torch.where(conformer_mask[:, :, None], result, torch.zeros_like(result))
    return result


def _relative_inputs(block, hidden, relative, *, values, valid, bond_scale, conformer_mask):
    if relative is None:
        return batched_relative_geometry(
            block.order, hidden, values=values, valid=valid, bond_scale=bond_scale,
            conformer_mask=conformer_mask)
    if not isinstance(relative, (tuple, list)) or len(relative) != 2:
        raise ValueError("Cached relative geometry must be an (even, odd) pair")
    k, paths = hidden.shape[:2]
    pair_mask = _pair_mask(conformer_mask.transpose(0, 1))
    output = []
    for value, width in zip(relative, RELATIVE_GEOMETRY_DIMS[block.order]):
        if (not isinstance(value, torch.Tensor) or value.shape != (paths, k, k, width)
                or not value.is_floating_point()):
            raise ValueError("Cached relative geometry has incompatible [P,K,K,g] shape")
        value = value.to(device=hidden.device, dtype=hidden.dtype)
        safe = torch.where(pair_mask, value, torch.zeros_like(value))
        if not torch.isfinite(safe).all():
            raise ValueError("Valid cached relative geometry must be finite")
        output.append(safe)
    return tuple(output)


def forward_paths(block, hidden, *, weights, values=None, valid=None,
                  bond_scale=None, conformer_mask=None, relative=None,
                  drop_multiplier=None, return_debug=False):
    """Update packed [K,P,32] paths, using a boolean [K,P] padding mask."""
    if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 3
            or hidden.shape[-1] != block.hidden_dim or not 1 <= hidden.shape[0] <= 5
            or not hidden.is_floating_point()):
        raise ValueError("Batched PaST content must be floating [K,P,32], 1<=K<=5")
    block._validate_weights(weights)
    k, paths = hidden.shape[:2]
    if conformer_mask is None:
        conformer_mask = torch.ones(k, paths, dtype=torch.bool, device=hidden.device)
    elif (not isinstance(conformer_mask, torch.Tensor) or conformer_mask.dtype != torch.bool
          or conformer_mask.shape != (k, paths)):
        raise ValueError("Conformer mask must be boolean [K,P]")
    conformer_mask = conformer_mask.to(device=hidden.device)
    if (conformer_mask.sum(dim=0) < 1).any():
        raise ValueError("Every path must have at least one valid conformer")
    safe_hidden = torch.where(conformer_mask[:, :, None], hidden, torch.zeros_like(hidden))
    if not torch.isfinite(safe_hidden).all():
        raise ValueError("Valid PaST content must be finite")
    if drop_multiplier is None:
        multiplier = block.sample_drop_multiplier(hidden)
    else:
        multiplier = torch.as_tensor(drop_multiplier, dtype=hidden.dtype, device=hidden.device)
        if (multiplier.ndim not in (0, 1)
                or (multiplier.ndim == 1 and multiplier.shape != (paths,))
                or multiplier.requires_grad or not torch.isfinite(multiplier).all()
                or (multiplier < 0).any()):
            raise ValueError("PaST drop multiplier must be fixed finite nonnegative scalar or [P]")

    if paths == 0:
        # Match the nonempty residual, whose learned LayerScale can promote
        # BF16 pair/channel work back to the FP32 path-state dtype.
        safe_hidden = safe_hidden.to(torch.promote_types(hidden.dtype, block.state.residual_scale.dtype))
        if not return_debug:
            return safe_hidden
        pairs = (safe_hidden.new_empty(0, k, k, block.pair_dim),
                 safe_hidden.new_empty(0, k, k, block.pair_dim))
        contracted = safe_hidden.new_empty(k, 0, block.pair_dim)
        return safe_hidden, dict(
            normalized_hidden=safe_hidden, lifted=pairs, pair=pairs,
            m_full=contracted, m_self=contracted, m_cross=contracted,
            gate=safe_hidden, message=safe_hidden, delta=safe_hidden, drop_multiplier=multiplier,
        )

    mask = conformer_mask.transpose(0, 1)
    even, relative_odd = _relative_inputs(
        block, safe_hidden, relative, values=values, valid=valid,
        bond_scale=bond_scale, conformer_mask=conformer_mask)
    normalized = block.state.content_norm(safe_hidden.transpose(0, 1))
    initial_s, initial_a = weights.lifting(normalized, even, relative_odd, block.state)
    lifted = (_masked_pairs(initial_s, mask), _masked_pairs(initial_a, mask))
    pair = batched_tensor_core(block, *lifted, weights=weights, conformer_mask=mask)
    m_full = weights.collapse(*pair, conformer_mask=mask)

    # All paths/conformers run simultaneously through the identical core from
    # their INITIAL diagonal. No detach, separate weights, or conformer loops.
    indices = torch.arange(k, device=hidden.device)
    single_s = lifted[0][:, indices, indices].reshape(paths * k, 1, 1, block.pair_dim)
    single_a = torch.zeros_like(single_s)
    single_mask = mask.reshape(paths * k, 1)
    single_pair = batched_tensor_core(
        block, single_s, single_a, weights=weights, conformer_mask=single_mask)
    m_self = weights.collapse(*single_pair, conformer_mask=single_mask).reshape(paths, k, block.pair_dim)
    # For K_actual=1 the full problem IS this singleton problem. Reuse its
    # computed result to avoid backend-dependent GEMM/reduction roundoff; the
    # real shared-core full-minus-self subtraction then cancels exactly.
    singleton = (mask.sum(dim=1) == 1)[:, None, None]
    m_full = torch.where(singleton, m_self, m_full)
    m_cross = m_full - m_self
    gate = weights.gate(torch.cat((normalized, m_cross), dim=-1))
    gate = torch.where(mask[:, :, None], gate, torch.zeros_like(gate))
    message = weights.up_projection(m_cross)
    delta = gate * message
    applied_multiplier = multiplier if multiplier.ndim == 0 else multiplier[:, None, None]
    result = safe_hidden + (applied_multiplier * block.state.residual_scale * delta).transpose(0, 1)
    if return_debug:
        return result, dict(
            normalized_hidden=normalized.transpose(0, 1), lifted=lifted, pair=pair,
            m_full=m_full.transpose(0, 1), m_self=m_self.transpose(0, 1),
            m_cross=m_cross.transpose(0, 1), gate=gate.transpose(0, 1),
            message=message.transpose(0, 1), delta=delta.transpose(0, 1),
            drop_multiplier=multiplier,
        )
    return result
