import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import unittest
import torch
from src.mult import *

class TestTempConv(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.seq_len = 10
        self.channels = 3
        self.kernel_size = 5 # Odd kernel size for symmetric padding
        self.batch_size = 4
        # Use batch version for consistency
        self.test_input_batched = torch.rand(self.batch_size, self.seq_len, self.channels, device=self.device)
        self.test_input_single = torch.rand(self.seq_len, self.channels, device=self.device)

    def test_forward_shape_batched(self):
        model = TempConv(kernel_size=self.kernel_size, channels=self.channels).to(self.device)
        model.eval() # Test eval mode
        with torch.no_grad():
            output = model(self.test_input_batched)
        self.assertEqual(output.shape, self.test_input_batched.shape)

    def test_forward_shape_single(self):
        model = TempConv(kernel_size=self.kernel_size, channels=self.channels).to(self.device)
        model.eval()
        with torch.no_grad():
             output = model(self.test_input_single)
        # Should return (seq_len, channels)
        self.assertEqual(output.shape, self.test_input_single.shape)
        self.assertEqual(output.dim(), 2)

    def test_forward_shape_different_ks(self):
        ks = 3
        model = TempConv(kernel_size=ks, channels=self.channels).to(self.device)
        model.eval()
        with torch.no_grad():
            output = model(self.test_input_batched)
        self.assertEqual(output.shape, self.test_input_batched.shape)

class TestPositionalEncoding(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.seq_len = 50
        self.feature_dim = 128
        self.batch_size = 8
        self.dropout = 0.1
        self.test_input = torch.randn(self.batch_size, self.seq_len, self.feature_dim, device=self.device)

    def test_forward_shape(self):
        model = PositionalEncoding(feature_dim=self.feature_dim, seq_len=self.seq_len).to(self.device)
        model.eval()
        with torch.no_grad():
            output = model(self.test_input)
        self.assertEqual(output.shape, self.test_input.shape)

    def test_forward_values_eval(self):
        model = PositionalEncoding(feature_dim=self.feature_dim, seq_len=self.seq_len, dropout=0.0).to(self.device) # Dropout 0
        model.eval()
        input_clone = self.test_input.clone()
        with torch.no_grad():
            output = model(self.test_input)
        # Output should be different from input (PE added)
        self.assertFalse(torch.allclose(input_clone, output))
        # Check if PE was added correctly (roughly) - compare first element
        expected_first = input_clone[0, 0, :] + model.pe[0, 0, :]
        self.assertTrue(torch.allclose(output[0, 0, :], expected_first))

    def test_forward_values_train_dropout(self):
        if self.dropout == 0.0: self.dropout = 0.5 # Ensure dropout is active
        model = PositionalEncoding(feature_dim=self.feature_dim, seq_len=self.seq_len, dropout=self.dropout).to(self.device)
        model.train() # Train mode for dropout
        input_clone = self.test_input.clone()
        output1 = model(input_clone)
        output2 = model(input_clone) # Run again

        # Outputs should be different due to dropout
        self.assertFalse(torch.equal(output1, output2))
        # Output should be different from input + PE (due to dropout)
        with torch.no_grad():
             expected_no_dropout = input_clone + model.pe[:, :self.seq_len, :]
        self.assertFalse(torch.allclose(output1, expected_no_dropout))

    def test_forward_shorter_sequence(self):
        model = PositionalEncoding(feature_dim=self.feature_dim, seq_len=self.seq_len).to(self.device)
        model.eval()
        short_seq_len = self.seq_len // 2
        short_input = torch.randn(self.batch_size, short_seq_len, self.feature_dim, device=self.device)
        with torch.no_grad():
            output = model(short_input)
        self.assertEqual(output.shape, short_input.shape)
        # Check PE was added correctly using the sliced buffer
        expected_first_short = short_input[0, 0, :] + model.pe[0, 0, :]
        self.assertTrue(torch.allclose(output[0, 0, :], expected_first_short))

    def test_forward_longer_sequence_error(self):
         # PE buffer is only self.seq_len long
        model = PositionalEncoding(feature_dim=self.feature_dim, seq_len=self.seq_len).to(self.device)
        long_seq_len = self.seq_len + 10
        long_input = torch.randn(self.batch_size, long_seq_len, self.feature_dim, device=self.device)
        with self.assertRaises(ValueError): # Expecting the check to raise error
             model(long_input)

class TestCrossModalTransformerLayer(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.batch_size = 4
        self.seq_len_p = 25
        self.seq_len_s = 35
        self.primary_feat_dim = 64
        self.secondary_feat_dim = 96
        self.dims = 128 # Must be divisible by num_heads
        self.num_heads = 8
        self.dropout = 0.1

        self.primary_input = torch.rand(self.batch_size, self.seq_len_p, self.primary_feat_dim, device=self.device)
        self.secondary_input = torch.rand(self.batch_size, self.seq_len_s, self.secondary_feat_dim, device=self.device)

    def test_forward_shape(self):
        model = CrossModalTransformerLayer(
            dims=self.dims, num_heads=self.num_heads,
            primary_feature_dim=self.primary_feat_dim, secondary_feature_dim=self.secondary_feat_dim,
            dropout=self.dropout
        ).to(self.device)
        model.eval() # Test eval mode shape
        with torch.no_grad():
            output = model(self.primary_input, self.secondary_input)
        expected_shape = (self.batch_size, self.seq_len_p, self.dims)
        self.assertEqual(output.shape, expected_shape)

    def test_forward_train_vs_eval_dropout(self):
        if self.dropout == 0.0: self.dropout = 0.5 # Ensure dropout is active
        model = CrossModalTransformerLayer(
            dims=self.dims, num_heads=self.num_heads,
            primary_feature_dim=self.primary_feat_dim, secondary_feature_dim=self.secondary_feat_dim,
            dropout=self.dropout
        ).to(self.device)

        # Eval mode
        model.eval()
        with torch.no_grad():
            output_eval = model(self.primary_input, self.secondary_input)

        # Train mode
        model.train()
        output_train1 = model(self.primary_input, self.secondary_input)
        output_train2 = model(self.primary_input, self.secondary_input) # Second run in train mode

        # Eval output should be deterministic (if inputs are same)
        # We need to run eval again to compare, as internal states might change
        model.eval()
        with torch.no_grad():
             output_eval2 = model(self.primary_input, self.secondary_input)
        self.assertTrue(torch.allclose(output_eval, output_eval2, atol=1e-6))

        # Train outputs should differ due to dropout
        self.assertFalse(torch.allclose(output_train1, output_train2, atol=1e-6))

        # Train output should differ from Eval output
        self.assertFalse(torch.allclose(output_train1, output_eval, atol=1e-6))

class TestCrossModalTransformer(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.batch_size = 4
        self.seq_len_p = 15
        self.seq_len_s = 20
        self.primary_feat_dim = 32
        self.secondary_feat_dim = 48
        self.dims = 64
        self.num_heads = 4
        self.num_layers = 3 # Test with multiple layers
        self.dropout = 0.1

        self.primary_input = torch.rand(self.batch_size, self.seq_len_p, self.primary_feat_dim, device=self.device)
        self.secondary_input = torch.rand(self.batch_size, self.seq_len_s, self.secondary_feat_dim, device=self.device)

    def test_forward_shape(self):
        model = CrossModalTransformer(
            primary_feature_dim=self.primary_feat_dim, secondary_feature_dim=self.secondary_feat_dim,
            dims=self.dims, num_heads=self.num_heads, num_layers=self.num_layers, dropout=self.dropout
        ).to(self.device)
        model.eval()
        with torch.no_grad():
            output = model(self.primary_input, self.secondary_input)
        expected_shape = (self.batch_size, self.seq_len_p, self.dims)
        self.assertEqual(output.shape, expected_shape)

    def test_forward_shape_single_layer(self):
        model = CrossModalTransformer(
            primary_feature_dim=self.primary_feat_dim, secondary_feature_dim=self.secondary_feat_dim,
            dims=self.dims, num_heads=self.num_heads, num_layers=1, dropout=self.dropout
        ).to(self.device)
        model.eval()
        with torch.no_grad():
            output = model(self.primary_input, self.secondary_input)
        expected_shape = (self.batch_size, self.seq_len_p, self.dims)
        self.assertEqual(output.shape, expected_shape)

# --- Run the tests ---
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

