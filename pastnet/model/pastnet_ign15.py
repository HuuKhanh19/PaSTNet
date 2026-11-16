"""Generic ordered-pair 15-map IGN baseline; only the relational operator changes.

The equality-partition basis spans Bell(4)=15 linear equivariant 2->2 maps.
Reductions use actual-valid-K means, as in PaST. No exchange-parity constraint
or parity products are applied after the original lifting is concatenated.
"""
import torch
from torch import nn
from pastnet.model.network import PaSTNet
from pastnet.model.past_batched import _masked_pairs, _pair_statistics, _relative_inputs

IGN_WIDTH = 20
BASIS_NAMES = ('R_ij','R_ji','diag_i','diag_j','delta_diag',
    'row_i','row_j','col_i','col_j','delta_row','delta_col',
    'trace','delta_trace','global','delta_global')


def generic_statistics(pair, mask):
    diagonal,row,trace,glob = _pair_statistics(pair,mask)
    col = _pair_statistics(pair.transpose(1,2),mask)[1]
    return diagonal,row,col,trace,glob


def basis15(pair, mask=None):
    pair = _masked_pairs(pair,mask)
    d,r,c,t,g = generic_statistics(pair,mask)
    eye = torch.eye(pair.shape[1],device=pair.device,dtype=pair.dtype)[None,:,:,None]
    maps = (pair,pair.transpose(1,2),d[:,:,None].expand_as(pair),d[:,None,:].expand_as(pair),eye*d[:,:,None],
        r[:,:,None].expand_as(pair),r[:,None,:].expand_as(pair),c[:,:,None].expand_as(pair),c[:,None,:].expand_as(pair),
        eye*r[:,:,None],eye*c[:,:,None],t[:,None,None].expand_as(pair),eye*t[:,None,None],
        g[:,None,None].expand_as(pair),eye*g[:,None,None])
    return tuple(_masked_pairs(value,mask) for value in maps)


class IGNLayer(nn.Module):
    def __init__(self,width=IGN_WIDTH):
        super().__init__()
        self.maps = nn.ModuleList([nn.Linear(width,width,bias=False) for _ in range(15)])
        # The two invariant affine biases are a broadcast vector and a diagonal vector.
        self.bias = nn.Parameter(torch.zeros(width))
        self.diagonal_bias = nn.Parameter(torch.zeros(width))

    def forward(self,pair,mask):
        mixed = sum(linear(value) for linear,value in zip(self.maps,basis15(pair,mask)))
        eye = torch.eye(pair.shape[1],device=pair.device,dtype=mixed.dtype)[None,:,:,None]
        output = torch.nn.functional.silu(mixed + self.bias + eye*self.diagonal_bias)
        return _masked_pairs(output,mask)


class IGNCollapse(nn.Module):
    def __init__(self,width=IGN_WIDTH):
        super().__init__()
        self.maps=nn.ModuleList([nn.Linear(width,16,bias=False) for _ in range(5)])

    def forward(self,pair,mask):
        d,r,c,t,g=generic_statistics(pair,mask)
        output=sum(linear(value) for linear,value in zip(self.maps,(d,r,c,t[:,None],g[:,None])))
        return torch.where(mask[...,None],output,torch.zeros_like(output))


class IGNWeights(nn.Module):
    def __init__(self,reference):
        super().__init__()
        self.lifting=reference.lifting
        self.input=nn.Linear(32,IGN_WIDTH,bias=False)
        self.layers=nn.ModuleList([IGNLayer(),IGNLayer()])
        self.collapse=IGNCollapse()
        self.gate=reference.gate
        self.up_projection=reference.up_projection

    def core(self,pair,mask):
        for layer in self.layers:pair=layer(pair,mask)
        return self.collapse(pair,mask)

    def update(self,normalized,message,mask):
        result=self.gate(torch.cat((normalized,message),dim=-1))*self.up_projection(message)
        return torch.where(mask[...,None],result,torch.zeros_like(result))


class IGNBlock(nn.Module):
    def __init__(self,reference):
        super().__init__()
        self.order=reference.order
        self.state=reference.state
        # All old tensor-layer parity norms/scales disappear, rather than remaining unused.
        self.state.layers=nn.ModuleList()

    def forward(self,hidden,*,weights,values=None,valid=None,bond_scale=None,
                conformer_mask=None,relative=None,return_debug=False,drop_multiplier=None):
        if hidden.ndim!=3 or hidden.shape[-1]!=32 or not 1<=hidden.shape[0]<=5:
            raise ValueError('IGN requires [K,P,32], K in [1,5]')
        k,paths=hidden.shape[:2]
        mask=conformer_mask
        if mask is None:mask=torch.ones(k,paths,device=hidden.device,dtype=torch.bool)
        if mask.shape!=(k,paths) or mask.dtype!=torch.bool or (mask.sum(0)<1).any():
            raise ValueError('Invalid conformer mask')
        safe=torch.where(mask[...,None],hidden,torch.zeros_like(hidden))
        if not torch.isfinite(safe).all():raise ValueError('Nonfinite valid input')
        if paths==0:
            result=safe.to(torch.promote_types(hidden.dtype,self.state.residual_scale.dtype))
            return (result,dict(empty=True)) if return_debug else result
        even,odd=_relative_inputs(self,safe,relative,values=values,valid=valid,bond_scale=bond_scale,conformer_mask=mask)
        normalized=self.state.content_norm(safe.transpose(0,1))
        s,a=weights.lifting(normalized,even,odd,self.state)
        mask=mask.transpose(0,1)
        pair=_masked_pairs(weights.input(torch.cat((s,a),dim=-1)),mask)
        full_message=weights.core(pair,mask)
        indices=torch.arange(k,device=hidden.device)
        singleton=pair[:,indices,indices].reshape(paths*k,1,1,IGN_WIDTH)
        singleton_mask=mask.reshape(paths*k,1)
        self_message=weights.core(singleton,singleton_mask).reshape(paths,k,16)
        # Evaluate the same complete F (including existing gate/up projection) twice.
        # This implements h + F(full) - F(singleton), as specified for E6.
        full_update=weights.update(normalized,full_message,mask)
        self_update=weights.update(normalized,self_message,mask)
        full_update=torch.where((mask.sum(1)==1)[:,None,None],self_update,full_update)
        delta=full_update-self_update
        multiplier=1 if drop_multiplier is None else drop_multiplier
        if isinstance(multiplier,torch.Tensor) and multiplier.ndim==1:multiplier=multiplier[:,None,None]
        result=safe+(multiplier*self.state.residual_scale*delta).transpose(0,1)
        if return_debug:return result,dict(pair=pair,m_full=full_message,m_self=self_message,
            f_full=full_update,f_self=self_update,delta=delta)
        return result


class PaSTNetIGN15(PaSTNet):
    def __init__(self,**options):
        super().__init__(**options)
        reference=sum(p.numel() for p in self.parameters())
        if reference!=667548:raise ValueError('E6 requires the original Full PaST architecture')
        with torch.random.fork_rng(devices=[]):
            for group in self.backbone.heavy_groups.values():
                group.past=nn.ModuleList([IGNWeights(w) for w in group.past])
            for stage in self.backbone.stages:
                stage.past=nn.ModuleList([IGNBlock(b) for b in stage.past])
        if sum(p.numel() for p in self.parameters())!=666972:raise ValueError('Unexpected E6 capacity')

    def parameter_counts(self):
        counts=super().parameter_counts()
        counts.update(reference_total=667548,ign_width=IGN_WIDTH,delta_from_reference=counts['total']-667548)
        return counts
