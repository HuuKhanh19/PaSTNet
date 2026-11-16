import unittest
import numpy as np
import torch
from rdkit import Chem
from pastnet.model.pastnet_ign15 import basis15,IGNBlock,IGNWeights,PaSTNetIGN15
from pastnet.model.past import PaSTBlock,PaSTWeights
from pastnet.model.network import PaSTNet
from pastnet.geometry.features import build_hrge_features

PARTITIONS=((0,1,0,1),(0,1,1,0),(0,1,0,0),(0,1,1,1),(0,0,0,0),
    (0,1,0,2),(0,1,1,2),(0,1,2,0),(0,1,2,1),(0,0,0,1),(0,0,1,0),
    (0,1,2,2),(0,0,1,1),(0,1,2,3),(0,0,1,2))


class IGNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_all_15_maps_against_partition_oracle_and_full_rank(self):
        k=5;x=torch.randn(1,k,k,2,dtype=torch.float64)
        for valid in ([True]*k,[True,False,True,True,False]):
            mask=torch.tensor([valid]);n=sum(valid)
            values=basis15(x.masked_fill(~(mask[:,:,None]&mask[:,None,:])[...,None],float('nan')),mask)
            for partition,actual in zip(PARTITIONS,values):
                expected=torch.zeros_like(x)
                free=len(set(partition[2:])-set(partition[:2]))
                for i in range(k):
                    for j in range(k):
                        for a in range(k):
                            for b in range(k):
                                ids=(i,j,a,b)
                                if all(valid[q] for q in ids) and all(ids[u]==ids[v] for u in range(4) for v in range(4) if partition[u]==partition[v]):
                                    expected[0,i,j]+=x[0,a,b]/n**free
                torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
        # Each input impulse is a separate batch; vectorize each full linear operator.
        impulse=torch.eye(25,dtype=torch.float64).reshape(25,5,5,1)
        matrix=torch.stack([b.reshape(-1) for b in basis15(impulse)],dim=1)
        self.assertEqual(torch.linalg.matrix_rank(matrix).item(),15)

    def test_capacity_surviving_initialization_and_rng(self):
        torch.manual_seed(42);reference=PaSTNet(bond_scale=.065);rng=torch.get_rng_state()
        torch.manual_seed(42);model=PaSTNetIGN15(bond_scale=.065)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in model.parameters()),666972)
        retained=0
        for name,value in model.state_dict().items():
            if name in reference.state_dict() and not ('.past.' in name and any(s in name for s in ('.layers.','.collapse.'))):
                torch.testing.assert_close(value,reference.state_dict()[name],atol=0,rtol=0);retained+=1
        self.assertGreater(retained,100)

    def test_permutation_masks_identity_and_both_branch_gradients(self):
        for order in range(4):
            weights=IGNWeights(PaSTWeights(order)).double()
            block=IGNBlock(PaSTBlock(order)).double()
            h=torch.randn(5,3,32,dtype=torch.float64,requires_grad=True)
            mask=torch.ones(5,3,dtype=torch.bool);mask[1:,0]=False;mask[3:,1]=False
            values=torch.rand(5,3,dtype=torch.float64)+1
            padded=h.masked_fill(~mask[...,None],float('nan'))
            kwargs=dict(weights=weights,values=values,valid=mask,conformer_mask=mask,bond_scale=.065)
            result,debug=block(padded,return_debug=True,**kwargs)
            torch.testing.assert_close(result[0,0],h[0,0],atol=0,rtol=0)
            self.assertEqual(torch.count_nonzero(result[~mask]),0)
            perm=torch.tensor([4,1,3,0,2])
            permuted=block(padded[perm],weights=weights,values=values[perm],valid=mask[perm],conformer_mask=mask[perm],bond_scale=.065)
            torch.testing.assert_close(permuted,result[perm],atol=1e-10,rtol=1e-10)
            for p in range(3):
                chosen=mask[:,p]
                alone=block(h[chosen,p:p+1],weights=weights,values=values[chosen,p:p+1],valid=mask[chosen,p:p+1],bond_scale=.065)
                torch.testing.assert_close(alone[:,0],result[chosen,p],atol=1e-10,rtol=1e-10)
            debug['f_full'].retain_grad();debug['f_self'].retain_grad()
            result.square().sum().backward()
            for key in ('f_full','f_self'):self.assertGreater(debug[key].grad.abs().sum().item(),0)
            for name,p in weights.named_parameters():
                self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)
                self.assertGreater(p.grad.abs().sum().item(),0,name)
            with torch.no_grad():
                one=block(h[:1],weights=weights,values=values[:1],valid=torch.ones_like(mask[:1]),bond_scale=.065)
                torch.testing.assert_close(one,h[:1],atol=0,rtol=0)

    def test_complete_model_packing_permutation_empty_orders_and_backward(self):
        torch.manual_seed(42)
        mol=Chem.MolFromSmiles('NCCO');xyz=np.random.default_rng(42).normal(size=(5,4,3))
        members=[build_hrge_features(mol,xyz[:k]) for k in (1,3,5)]
        members.append(build_hrge_features(Chem.MolFromSmiles('C'),xyz[:1,:1]))
        model=PaSTNetIGN15(bond_scale=.065).eval();result=model(members)
        with torch.no_grad():
            independent=torch.stack([model(m) for m in members])
            torch.testing.assert_close(result,independent,atol=5e-6,rtol=5e-5)
            permuted=model(build_hrge_features(mol,xyz[[3,0,4,2,1]]))
            torch.testing.assert_close(result[2],permuted,atol=5e-6,rtol=5e-5)
        result.square().sum().backward()
        for name,p in model.named_parameters():
            self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)


if __name__=='__main__':unittest.main()
