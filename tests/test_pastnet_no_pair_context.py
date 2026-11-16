"""Scientific checks for the Table 2 pair-to-pair context intervention."""

import copy
import unittest
from dataclasses import replace

import numpy as np
import torch
from rdkit import Chem

from pastnet.model.network import PaSTNet
from pastnet.model.past import PaSTBlock, PaSTWeights, PaSTTensorLayerState, PaSTTensorLayerWeights
from pastnet.model.past_batched import batched_enforce_parity
from pastnet.model.pastnet_no_pair_context import PairLocalTensorLayer, PaSTNetNoPairContext
from pastnet.model.input import reverse_hrge_features
from pastnet.geometry.features import build_hrge_features


class NoPairContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)

    def test_retains_initialization_and_rng(self):
        reference = PaSTNet(bond_scale=.065)
        rng = torch.get_rng_state().clone()
        torch.manual_seed(42)
        model = PaSTNetNoPairContext(bond_scale=.065)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        original = reference.state_dict()
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, original[name]), name)
        self.assertEqual(model.parameter_counts()["total"], 618396)
        self.assertEqual(model.parameter_counts()["removed_context_parameters"], 49152)

    def test_pair_independence_and_equivalence_to_disabled_context(self):
        reference = PaSTTensorLayerWeights().double()
        layer = PairLocalTensorLayer(copy.deepcopy(reference))
        state = PaSTTensorLayerState().double()
        with torch.no_grad():
            for linear in reference.symmetric_mixer[2:]:
                linear.weight.zero_()
            reference.odd_row.weight.zero_()
        s, a = batched_enforce_parity(torch.randn(2, 5, 5, 16, dtype=torch.float64),
                                     torch.randn(2, 5, 5, 16, dtype=torch.float64))
        s.requires_grad_()
        a.requires_grad_()
        actual = layer(s, a, state)
        for value, expected in zip(actual, reference(s, a, state)):
            torch.testing.assert_close(value, expected, atol=1e-12, rtol=1e-12)
        gradients = torch.autograd.grad(sum(x[0, 1, 2].square().sum() for x in actual), (s, a))
        allowed = torch.zeros(2, 5, 5, dtype=torch.bool)
        allowed[0, 1, 2] = allowed[0, 2, 1] = True
        for gradient in gradients:
            self.assertEqual(torch.count_nonzero(gradient[~allowed]), 0)
            self.assertGreater(gradient[allowed].abs().sum().item(), 0)
        changed_s, changed_a = s.detach().clone(), a.detach().clone()
        changed_s[0, 1, 3] += 20
        changed_s[0, 3, 1] += 20
        changed_a[0, 1, 3] += 20
        changed_a[0, 3, 1] -= 20
        for value, changed in zip(actual, layer(changed_s, changed_a, state)):
            torch.testing.assert_close(value[0, 1, 2], changed[0, 1, 2], atol=0, rtol=0)

    def test_all_orders_masking_singleton_parity_and_gradients(self):
        for order in range(4):
            block, weights = PaSTBlock(order).double(), PaSTWeights(order).double()
            weights.layers = torch.nn.ModuleList([PairLocalTensorLayer(x) for x in weights.layers])
            hidden = torch.randn(5, 3, 32, dtype=torch.float64, requires_grad=True)
            mask = torch.ones(5, 3, dtype=torch.bool)
            mask[1:, 0] = False
            hidden_input = hidden.masked_fill(~mask[..., None], float("nan"))
            values = torch.rand(5, 3, dtype=torch.float64) + 1
            output, debug = block(hidden_input, weights=weights, conformer_mask=mask,
                                  values=values, valid=mask, bond_scale=.065, return_debug=True)
            torch.testing.assert_close(output[0, 0], hidden[0, 0], atol=0, rtol=0)
            self.assertEqual(torch.count_nonzero(output[~mask]), 0)
            symmetric, odd = debug["pair"]
            torch.testing.assert_close(symmetric, symmetric.transpose(1, 2), atol=0, rtol=0)
            torch.testing.assert_close(odd, -odd.transpose(1, 2), atol=0, rtol=0)
            for path in range(3):
                selected = mask[:, path]
                independent = block(hidden[selected, path], weights=weights,
                    values=values[selected, path], valid=mask[selected, path], bond_scale=.065)
                torch.testing.assert_close(output[selected, path], independent, atol=1e-10, rtol=1e-10)
            output.square().sum().backward()
            for parameter in list(block.parameters()) + list(weights.parameters()):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_model_permutation_reversal_packing_and_backward(self):
        mol = Chem.MolFromSmiles("NCCO")
        coords = np.random.default_rng(33).normal(size=(5, 4, 3))
        members = [build_hrge_features(mol, coords[:k]) for k in (1, 3, 5)]
        members.append(build_hrge_features(Chem.MolFromSmiles("C"), coords[:1, :1]))
        model = PaSTNetNoPairContext(bond_scale=.065).eval()
        output = model(members)
        with torch.no_grad():
            independent = torch.stack([model(member) for member in members])
            torch.testing.assert_close(output, independent, atol=4e-6, rtol=4e-5)
            permuted = model(build_hrge_features(mol, coords[[3, 0, 4, 1, 2]]))
            torch.testing.assert_close(output[2], permuted, atol=4e-6, rtol=4e-5)
            reversed_member = replace(members[2], features=tuple(
                reverse_hrge_features(value, order) for order, value in enumerate(members[2].features)))
            torch.testing.assert_close(output[2], model(reversed_member), atol=4e-6, rtol=4e-5)
        output.square().mean().backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)


if __name__ == "__main__":
    unittest.main()
