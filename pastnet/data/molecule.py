"""The RDKit molecule/atom order used by the original PCNN graph builder."""

from rdkit import Chem


def molecule_from_smiles(smiles):
    """Preserve PCNN's MolFromSmiles -> AddHs behavior, including explicit H.

    Atom i in this molecule is node i in atom_to_graph. No canonicalization or
    atom renumbering occurs here. Cache consumers must use cache.graph_smiles
    (canonical isomeric SMILES) when constructing the corresponding graph.
    """
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else Chem.AddHs(mol)


def atom_order_signature(mol):
    """Ordered atom identities and connectivity; independent of coordinates."""
    return {
        "atoms": [
            [atom.GetAtomicNum(), atom.GetIsotope(), atom.GetFormalCharge(),
             int(atom.GetChiralTag())]
            for atom in mol.GetAtoms()
        ],
        "bonds": [
            [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
            for bond in mol.GetBonds()
        ],
    }
