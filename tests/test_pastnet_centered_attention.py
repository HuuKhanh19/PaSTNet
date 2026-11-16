"""Scientific acceptance tests for reviewer E1, independent singleton oracle."""

import unittest
import numpy as np
import torch
from rdkit import Chem

from pastnet.model.past_batched import batched_relative_geometry
from pastnet.model.pastnet_ablation import PaSTNetAblation, SamePathAttentionBlock, SamePathAttentionWeights, ATTENTION_FFN_DIMS
from pastnet.model.pastnet_centered_attention import CenteredSamePathAttentionBlock, PaSTNetCenteredAttention
from pastnet.geometry.features import build_hrge_features


class CenteredAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(42)

    def test_identical_initialization_rng_and_parameter_count(self):
        original = PaSTNetAblation(variant="same_path_attention", bond_scale=.065)
        rng = torch.get_rng_state().clone()
        torch.manual_seed(42)
        centered = PaSTNetCenteredAttention(bond_scale=.065)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(original.parameter_counts(), centered.parameter_counts())
        self.assertEqual(original.state_dict().keys(), centered.state_dict().keys())
        for key, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, centered.state_dict()[key]), key)

    def test_explicit_singleton_oracle_masking_and_both_branch_gradients(self):
        for order in range(4):
            block = CenteredSamePathAttentionBlock(order).double()
            weights = SamePathAttentionWeights(order, ATTENTION_FFN_DIMS[order]).double()
            mask = torch.tensor([[0, 1, 1], [0, 0, 1], [1, 1, 1], [0, 0, 0], [0, 1, 1]], dtype=torch.bool)
            hidden = torch.randn(5, 3, 32, dtype=torch.float64).masked_fill(~mask[..., None], float("nan")).requires_grad_()
            geometry_valid = mask.clone()
            geometry_valid[0, 1] = False
            relative = tuple(t.requires_grad_() for t in batched_relative_geometry(order, hidden,
                values=torch.rand(5, 3, dtype=torch.float64) + 1, valid=geometry_valid,
                conformer_mask=mask, bond_scale=.065))
            output, debug = block(hidden, weights=weights, relative=relative, conformer_mask=mask, return_debug=True)
            clean = torch.where(mask[..., None], hidden, torch.zeros_like(hidden))
            full = SamePathAttentionBlock.forward(block, hidden, weights=weights, relative=relative, conformer_mask=mask)
            expected = torch.zeros_like(full)
            for p in range(3):
                for c in range(5):
                    if mask[c, p]:
                        rel = tuple(t[p:p+1, c:c+1, c:c+1] for t in relative)
                        singleton = SamePathAttentionBlock.forward(block, hidden[c:c+1, p:p+1], weights=weights, relative=rel)[0, 0]
                        expected[c, p] = clean[c, p] + full[c, p] - singleton
            torch.testing.assert_close(output, expected, atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(output[2, 0], hidden[2, 0], atol=0, rtol=0)
            self.assertEqual(torch.count_nonzero(output[~mask]).item(), 0)
            params = [hidden] + list(block.parameters()) + list(weights.parameters())
            actual_grad = torch.autograd.grad(output.square().sum(), params, retain_graph=True)
            expected_grad = torch.autograd.grad(expected.square().sum(), params, retain_graph=True)
            for a, b in zip(actual_grad, expected_grad):
                self.assertTrue(torch.isfinite(a).all())
                torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
            branch_grad = torch.autograd.grad(output.square().sum(),
                (debug["full_output"], debug["singleton_output"]))
            self.assertGreater(branch_grad[0].abs().sum().item(), 0)
            torch.testing.assert_close(branch_grad[0], -branch_grad[1], atol=0, rtol=0)

    def test_singleton_identity_training_and_eval_empty_orders(self):
        for order in range(4):
            block = CenteredSamePathAttentionBlock(order)
            weights = SamePathAttentionWeights(order, ATTENTION_FFN_DIMS[order])
            for training in (True, False):
                block.train(training)
                for paths in (0, 3):
                    hidden = torch.randn(1, paths, 32, requires_grad=True)
                    output = block(hidden, weights=weights, values=torch.rand(1, paths)+1,
                                   valid=torch.ones(1, paths, dtype=torch.bool), bond_scale=.065)
                    torch.testing.assert_close(output, hidden, atol=0, rtol=0)
                    grad, = torch.autograd.grad(output.sum(), hidden)
                    torch.testing.assert_close(grad, torch.ones_like(hidden), atol=0, rtol=0)

    def test_cross_conformer_geometry_permutation_and_path_isolation(self):
        for order in range(4):
            block = CenteredSamePathAttentionBlock(order).double()
            weights = SamePathAttentionWeights(order, ATTENTION_FFN_DIMS[order]).double()
            hidden = torch.randn(5, 2, 32, dtype=torch.float64, requires_grad=True)
            relative = batched_relative_geometry(order, hidden, values=torch.rand(5, 2, dtype=torch.float64)+1,
                valid=torch.ones(5, 2, dtype=torch.bool), bond_scale=.065)
            output = block(hidden, weights=weights, relative=relative)
            grad, = torch.autograd.grad(output[0, 0].square().sum(), hidden)
            self.assertGreater(grad[1:, 0].abs().sum().item(), 0)
            self.assertEqual(grad[:, 1].abs().sum().item(), 0)
            permutation = torch.tensor([3, 0, 4, 1, 2])
            permuted = block(hidden[permutation], weights=weights,
                relative=tuple(t[:, permutation][:, :, permutation] for t in relative))
            torch.testing.assert_close(permuted, output[permutation], atol=1e-12, rtol=1e-12)
            if order:
                changed = block(hidden, weights=weights, relative=tuple(torch.zeros_like(t) for t in relative))
                self.assertGreater((changed-output).abs().max().item(), 1e-7)

    def test_packed_model_and_backward(self):
        coords = np.random.default_rng(33).normal(size=(5, 4, 3))
        mol = Chem.MolFromSmiles("NCCO")
        members = [build_hrge_features(mol, coords[:k]) for k in (1, 3, 5)]
        members.append(build_hrge_features(Chem.MolFromSmiles("C"), coords[:1, :1]))
        model = PaSTNetCenteredAttention(bond_scale=.065)
        prediction = model(members)
        with torch.no_grad():
            independent = torch.stack([model(m) for m in members])
            torch.testing.assert_close(prediction, independent, atol=4e-6, rtol=4e-5)
        prediction.square().mean().backward()
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)


if __name__ == "__main__":
    unittest.main()
