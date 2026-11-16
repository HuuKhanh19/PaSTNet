"""Scientific checks: active capacity, pair locality, parity, K1 and autograd."""
import unittest
from dataclasses import replace
import numpy as np
import torch
from rdkit import Chem
from pastnet.model.network import PaSTNet
from pastnet.model.past import PaSTBlock, PaSTWeights, PaSTTensorLayerState, PaSTTensorLayerWeights
from pastnet.model.past_batched import batched_enforce_parity
from pastnet.model.pastnet_no_pair_context import PairLocalTensorLayer, PaSTNetNoPairContext
from pastnet.model.pastnet_pair_local_matched import PaSTNetPairLocalMatched, widen_pair_local_ffns
from pastnet.model.input import reverse_hrge_features
from pastnet.geometry.features import build_hrge_features


class PairLocalMatchedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def setUp(self):torch.manual_seed(42)

    def test_exact_capacity_retained_weights_and_rng(self):
        original = PaSTNetNoPairContext(bond_scale=.065)
        rng = torch.get_rng_state().clone()
        torch.manual_seed(42)
        model = PaSTNetPairLocalMatched(bond_scale=.065)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(original.parameter_counts()['total'],618396)
        self.assertEqual(model.parameter_counts()['total'],667548)
        self.assertEqual(model.parameter_counts()['delta_from_reference'],0)
        for name,value in model.state_dict().items():
            if '.past.' in name and '.layers.' in name and ('symmetric_ffn' in name or 'odd_ffn' in name):continue
            self.assertTrue(torch.equal(value,original.state_dict()[name]),name)
        layers=[m for m in model.modules() if isinstance(m,PairLocalTensorLayer)]
        self.assertEqual(len(layers),24)
        for layer in layers:
            self.assertEqual(len(layer.symmetric_mixer),2)
            self.assertFalse(hasattr(layer,'odd_row'))
            self.assertEqual(layer.symmetric_ffn[0].out_features,72)
            self.assertEqual(layer.odd_ffn.input.out_features,39)

    def test_two_layers_are_pair_local_and_parity_preserving(self):
        layers=[widen_pair_local_ffns(PairLocalTensorLayer(PaSTTensorLayerWeights())).double() for _ in range(2)]
        states=[PaSTTensorLayerState().double() for _ in range(2)]
        s,a=batched_enforce_parity(torch.randn(2,5,5,16,dtype=torch.float64),torch.randn(2,5,5,16,dtype=torch.float64))
        s.requires_grad_();a.requires_grad_()
        def core(s,a):
            for layer,state in zip(layers,states):s,a=layer(s,a,state)
            return s,a
        outputs=core(s,a)
        torch.testing.assert_close(outputs[0],outputs[0].transpose(1,2),atol=0,rtol=0)
        torch.testing.assert_close(outputs[1],-outputs[1].transpose(1,2),atol=0,rtol=0)
        gradients=torch.autograd.grad(sum(t[0,1,2].square().sum() for t in outputs),(s,a))
        allowed=torch.zeros(2,5,5,dtype=torch.bool);allowed[0,1,2]=allowed[0,2,1]=True
        for grad in gradients:
            self.assertEqual(torch.count_nonzero(grad[~allowed]).item(),0)
            self.assertGreater(grad[allowed].abs().sum().item(),0)
        changed_s,changed_a=s.detach().clone(),a.detach().clone()
        changed_s[0,1,3]+=20;changed_s[0,3,1]+=20
        changed_a[0,1,3]+=20;changed_a[0,3,1]-=20
        for original,changed in zip(outputs,core(changed_s,changed_a)):
            torch.testing.assert_close(original[0,1,2],changed[0,1,2],atol=0,rtol=0)
        # The odd FFN must be an odd map even after widening.
        z=torch.randn(7,16,dtype=torch.float64)
        torch.testing.assert_close(layers[0].odd_ffn(-z),-layers[0].odd_ffn(z),atol=0,rtol=0)

    def test_all_orders_singleton_masks_contraction_and_gradients(self):
        for order in range(4):
            block,weights=PaSTBlock(order).double(),PaSTWeights(order)
            weights.layers=torch.nn.ModuleList([widen_pair_local_ffns(PairLocalTensorLayer(t)) for t in weights.layers])
            weights=weights.double()
            hidden=torch.randn(5,3,32,dtype=torch.float64,requires_grad=True)
            mask=torch.ones(5,3,dtype=torch.bool);mask[:,0]=False;mask[3,0]=True
            values=torch.rand(5,3,dtype=torch.float64)+1
            out=block(hidden.masked_fill(~mask[...,None],float('nan')),weights=weights,
                conformer_mask=mask,values=values,valid=mask,bond_scale=.065)
            torch.testing.assert_close(out[3,0],hidden[3,0],atol=0,rtol=0)
            self.assertEqual(torch.count_nonzero(out[~mask]).item(),0)
            grad,=torch.autograd.grad(out[0,1].square().sum(),hidden,retain_graph=True)
            self.assertGreater(grad[1:,1].abs().sum().item(),0)  # final contraction is retained
            for path in range(3):
                selected=mask[:,path]
                oracle=block(hidden[selected,path],weights=weights,values=values[selected,path],
                    valid=mask[selected,path],bond_scale=.065)
                torch.testing.assert_close(out[selected,path],oracle,atol=1e-10,rtol=1e-10)
            out.square().sum().backward()
            for param in list(block.parameters())+list(weights.parameters()):
                self.assertIsNotNone(param.grad)
                self.assertTrue(torch.isfinite(param.grad).all())
            for layer in weights.layers:
                self.assertGreater(layer.symmetric_ffn[0].weight.grad.abs().sum().item(),0)
                self.assertGreater(layer.odd_ffn.input.weight.grad.abs().sum().item(),0)
            h=hidden[:1].detach().requires_grad_()
            singleton=block(h,weights=weights,values=values[:1],valid=mask[:1],bond_scale=.065)
            torch.testing.assert_close(singleton,h,atol=0,rtol=0)

    def test_model_packing_permutation_reversal_and_backward(self):
        coords=np.random.default_rng(33).normal(size=(5,4,3));mol=Chem.MolFromSmiles('NCCO')
        members=[build_hrge_features(mol,coords[:k]) for k in (1,3,5)]
        members.append(build_hrge_features(Chem.MolFromSmiles('C'),coords[:1,:1]))
        model=PaSTNetPairLocalMatched(bond_scale=.065).eval();out=model(members)
        with torch.no_grad():
            torch.testing.assert_close(out,torch.stack([model(m) for m in members]),atol=4e-6,rtol=4e-5)
            torch.testing.assert_close(out[2],model(build_hrge_features(mol,coords[[3,0,4,1,2]])),atol=4e-6,rtol=4e-5)
            reversed_member=replace(members[2],features=tuple(reverse_hrge_features(v,o) for o,v in enumerate(members[2].features)))
            torch.testing.assert_close(out[2],model(reversed_member),atol=4e-6,rtol=4e-5)
        out.square().mean().backward()
        for name,p in model.named_parameters():
            self.assertIsNotNone(p.grad,name)
            self.assertTrue(torch.isfinite(p.grad).all(),name)


if __name__=='__main__':unittest.main()
