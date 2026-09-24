"""SchNet atom backbone adapted from the pinned SchNet-GP source."""


from math import pi as PI
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding, Linear, ModuleList, Sequential
from torch_geometric.nn import MessagePassing


class GaussianSmearing(nn.Module):
    """Expand distances into Gaussian basis functions."""

    def __init__(self, start: float = 0.0, stop: float = 5.0,
                 num_gaussians: int = 50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(nn.Module):
    """Softplus activation shifted so that ShiftedSoftplus(0) = 0."""

    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(x) - self.shift


class CFConv(MessagePassing):
    """Continuous-filter convolution layer."""

    def __init__(self, in_channels: int, out_channels: int,
                 num_filters: int, nn: Sequential, cutoff: float):
        super().__init__(aggr='add')
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor,
                edge_attr: Tensor, cutoff_envelope: Optional[Tensor] = None) -> Tensor:
        C = (0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
             if cutoff_envelope is None else cutoff_envelope)
        W = self.nn(edge_attr) * C.view(-1, 1)
        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def aggregate(self, inputs: Tensor, index: Tensor, ptr=None, dim_size=None) -> Tensor:
        # Autocast uses BF16 for channel maps; accumulate neighbor sums in FP32.
        values = inputs.float() if inputs.dtype in (torch.float16, torch.bfloat16) else inputs
        return super().aggregate(values, index, ptr=ptr, dim_size=dim_size)

    def message(self, x_j: Tensor, W: Tensor) -> Tensor:
        return x_j * W


class InteractionBlock(nn.Module):
    """SchNet interaction block: filter-generating network + CFConv."""

    def __init__(self, hidden_channels: int, num_gaussians: int,
                 num_filters: int, cutoff: float):
        super().__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels,
                           num_filters, self.mlp, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[2].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor,
                edge_attr: Tensor, cutoff_envelope: Optional[Tensor] = None) -> Tensor:
        x = self.conv(x, edge_index, edge_weight, edge_attr, cutoff_envelope)
        x = self.act(x)
        x = self.lin(x)
        return x


SCHNET_SOURCE_SHA256 = "04ec32ba471247c7885d027b1cd5733130400ad88c97ee6d9c347ad4bcd3cd58"


SCHNET_SETTINGS = dict(
    source_sha256=SCHNET_SOURCE_SHA256,
    hidden_dim=128, num_interactions=3, cutoff=10.0, num_rbf=50,
    rbf_centers="linspace(0,10,50)",
    rbf_gamma=-GaussianSmearing(0.0,10.0,50).coeff,
    cutoff_envelope="cosine", activation="shifted_softplus",
    filter_hidden_dim=128, neighbor_cap=None, self_edges=False,
    atomic_number_embeddings=100, pretrained=False,
    output="return_atom_emb_only=True",
)


def spatial_graph(coords):
    """Cache complete directed cutoff graphs for actual conformers only.

    The upstream radius builder's default cap32 is explicitly disabled here:
    all nonself pairs at distance <=10 A are present, per PaSTNet's definition.
    Construction is vectorized over all K; no conformer forward loop occurs.
    """
    _, n, _ = coords.shape
    distances = torch.linalg.vector_norm(coords[:, :, None] - coords[:, None, :], dim=-1)
    keep = (distances <= 10.0) & ~torch.eye(n, device=coords.device, dtype=torch.bool)[None]
    conformer, source, target = keep.nonzero(as_tuple=True)
    edges = torch.stack((conformer*n + source, conformer*n + target))
    lengths = distances[conformer,source,target].float()
    rbf = GaussianSmearing(0.0,10.0,50).to(coords.device)(lengths)
    envelope = 0.5 * (torch.cos(lengths * PI / 10.0) + 1.0)
    return edges, rbf, envelope


class SchNetAtomEncoder(nn.Module):
    """The atom-only SchNet backbone, with the original initialization order."""

    def __init__(self):
        super().__init__()
        self.embedding = Embedding(100, 128, padding_idx=0)
        self.distance_expansion = GaussianSmearing(0.0, 10.0, 50)
        self.interactions = ModuleList([
            InteractionBlock(128, 50, 128, 10.0) for _ in range(3)
        ])
        # The reference constructs then removes the upstream prediction head.
        # Consume its initialization draws to preserve all subsequent weights.
        unused_lin1 = Linear(128, 64)
        unused_lin2 = Linear(64, 1)
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        torch.nn.init.xavier_uniform_(unused_lin1.weight)
        del unused_lin1, unused_lin2

    def forward(self, atomic_numbers, edges, rbf, envelope):
        hidden = self.embedding(atomic_numbers)
        rbf, envelope = rbf.to(hidden.dtype), envelope.to(hidden.dtype)
        for interaction in self.interactions:
            hidden = hidden + interaction(hidden, edges, None, rbf, envelope)
        return hidden
