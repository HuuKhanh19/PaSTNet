"""PCNN-compatible atomic and bond encodings; see THIRD_PARTY_NOTICES.md."""

import torch
from torch import nn
from rdkit import Chem
from jarvis.core.specie import get_node_attributes

def bond_length_approximation(bond_type):
    bond_length_dict = {"SINGLE": 1.0, "DOUBLE": 1.4, "TRIPLE": 1.8, "AROMATIC": 1.5}
    return bond_length_dict.get(bond_type, 1.0)

def encode_bond_14(bond):
    #7+4+2+2+6 = 21
    bond_dir = [0] * 7
    bond_dir[bond.GetBondDir()] = 1

    bond_type = [0] * 4
    bond_type[int(bond.GetBondTypeAsDouble()) - 1] = 1

    bond_length = bond_length_approximation(bond.GetBondType())

    in_ring = [0, 0]
    in_ring[int(bond.IsInRing())] = 1

    non_bond_feature = [0]*6
    return bond_dir + bond_type + [bond_length,bond_length**2] + in_ring + non_bond_feature

class BondRBF(nn.Module):
    """Smooth encoding of cached bond length, with fixed Angstrom centers."""

    def __init__(self, size=16, maximum=4.0):
        super().__init__()
        self.register_buffer('centers', torch.linspace(0.0, maximum, size))
        self.register_buffer('gamma', torch.tensor(((size - 1) / maximum) ** 2))

    def forward(self, distances):
        return torch.exp(-self.gamma * (distances.unsqueeze(-1) - self.centers).square())
