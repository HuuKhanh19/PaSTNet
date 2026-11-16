"""Check the exact message intervention, unchanged initialization and symmetries."""

import copy
import unittest
from dataclasses import replace

import numpy as np
import torch
from rdkit import Chem

from pastnet.model.network import PaSTNet
from pastnet.model.past import PaSTBlock, PaSTWeights
from pastnet.model.pastnet_no_singleton_subtraction import PaSTNetNoSingletonSubtraction
from pastnet.model.input import reverse_hrge_features
from pastnet.geometry.features import build_hrge_features


class NoSingletonSubtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)

    def test_identical_parameters_initialization_and_rng(self):
        reference = PaSTNet(bond_scale=.065)
        rng = torch.get_rng_state().clone()
        torch.manual_seed(42)
        model = PaSTNetNoSingletonSubtraction(bond_scale=.065)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(model.state_dict().keys(), reference.state_dict().keys())
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, reference.state_dict()[name]), name)
        self.assertEqual(model.parameter_counts(), reference.parameter_counts())
        self.assertEqual(model.parameter_counts()["total"], 667548)
        self.assertEqual(sum(isinstance(m, PaSTBlock) and not m.singleton_subtraction
                             for m in model.modules()), 12)

    def test_exact_full_message_gate_output_and_gradients_all_orders(self):
        for order in range(4):
            reference, weights = PaSTBlock(order).double(), PaSTWeights(order).double()
            ablated = copy.deepcopy(reference)
            ablated.singleton_subtraction = False
            hidden = torch.randn(5, 3, 32, dtype=torch.float64, requires_grad=True)
            mask = torch.ones(5, 3, dtype=torch.bool)
            mask[1:, 0] = False
            mask[3:, 1] = False
            values = torch.rand(5, 3, dtype=torch.float64) + 1
            options = dict(weights=weights, conformer_mask=mask, values=values,
                           valid=mask, bond_scale=.065, return_debug=True)
            inputs = hidden.masked_fill(~mask[..., None], float("nan"))
            baseline, before = reference(inputs, **options)
            output, after = ablated(inputs, **options)
            for key in ("lifted", "pair"):
                for a, b in zip(before[key], after[key]):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
            for key in ("m_full", "m_self", "normalized_hidden"):
                torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)
            torch.testing.assert_close(before["m_cross"], before["m_full"] - before["m_self"], atol=0, rtol=0)
            torch.testing.assert_close(after["m_cross"], before["m_full"], atol=0, rtol=0)
            full = before["m_full"]
            gate = weights.gate(torch.cat((before["normalized_hidden"], full), dim=-1))
            gate = gate.masked_fill(~mask[..., None], 0)
            expected = hidden.masked_fill(~mask[..., None], 0) + ablated.state.residual_scale * gate * weights.up_projection(full)
            torch.testing.assert_close(output, expected, atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(baseline[0, 0], hidden[0, 0], atol=0, rtol=0)
            self.assertGreater((output[0, 0] - hidden[0, 0]).abs().max().item(), 1e-8)
            self.assertEqual(torch.count_nonzero(output[~mask]), 0)
            actual_grad = torch.autograd.grad(output.square().sum(), hidden, retain_graph=True)[0]
            expected_grad = torch.autograd.grad(expected.square().sum(), hidden)[0]
            torch.testing.assert_close(actual_grad, expected_grad, atol=1e-10, rtol=1e-10)
            for path in range(3):
                selected = mask[:, path]
                independent = ablated(hidden[selected, path], weights=weights,
                    values=values[selected, path], valid=mask[selected, path], bond_scale=.065)
                torch.testing.assert_close(output[selected, path], independent, atol=1e-10, rtol=1e-10)

    def test_model_permutation_reversal_packing_and_backward(self):
        mol = Chem.MolFromSmiles("NCCO")
        coords = np.random.default_rng(33).normal(size=(5, 4, 3))
        members = [build_hrge_features(mol, coords[:k]) for k in (1, 3, 5)]
        members.append(build_hrge_features(Chem.MolFromSmiles("C"), coords[:1, :1]))
        model = PaSTNetNoSingletonSubtraction(bond_scale=.065).eval()
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
