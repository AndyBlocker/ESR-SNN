import torch
import pytest
import numpy as np
from typing import Tuple, List

# Import your operators
from original_operator import ST_BIFNodeATGF_MS
from cuda_operator import ST_BIFNodeATGF_MS_CUDA

class TestSNNOperator:
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @pytest.fixture
    def test_shapes(self) -> List[Tuple[int, int, int]]:
        """Returns list of (time_steps, batch_size, features) to test"""
        return [
            (10, 2, 32),    # Small batch, medium features
            (5, 16, 64),    # Medium batch, large features
            (20, 1, 16),    # Long sequence, single batch
            (15, 8, 128),   # Medium sequence, large features
            (32, 4, 75264),   # large sequence, large features
            (128, 4, 1024),   # huge sequence, medium features
        ]

    @pytest.fixture
    def threshold_values(self, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (v_th, T_max, T_min) threshold values"""
        return (
            torch.tensor(0.1, device=device),
            torch.tensor(4.0, device=device),
            torch.tensor(-4.0, device=device)
        )

    def generate_input_data(self, shape: Tuple[int, int, int], device: torch.device, dtype) -> torch.Tensor:
        """Generate input data with specific distribution"""
        time_steps, batch_size, features = shape
        # Generate values between -0.5 and 0.5 for realistic neuron inputs
        x = torch.rand(time_steps, batch_size, features, device=device, dtype=dtype) - 0.5
        x.requires_grad = True
        return x

    def assert_tensor_close(self, a: torch.Tensor, b: torch.Tensor, rtol=1e-5, atol=1e-6):
        """Custom assertion for tensor comparison with detailed error message"""
        if not torch.allclose(a, b, rtol=rtol, atol=atol):
            max_diff = torch.max(torch.abs(a - b)).item()
            mean_diff = torch.mean(torch.abs(a - b)).item()
            diff_locations = torch.where(torch.abs(a - b) > atol)
            pytest.fail(f"Tensors not close!\nMax difference: {max_diff}\n"
                       f"Mean difference: {mean_diff}\n"
                       f"Number of differing elements: {len(diff_locations[0])}\n"
                       f"First few differences:\n"
                       f"a: {a[diff_locations][:5]}\n"
                       f"b: {b[diff_locations][:5]}")

    @pytest.mark.parametrize("use_seed, dtype", 
        [(True, torch.float16), (False, torch.float16), (True, torch.float32), (False, torch.float32)]
    )
    def test_forward_pass(self, test_shapes, threshold_values, device, use_seed, dtype):
        """Test forward pass output consistency"""
        if use_seed:
            torch.manual_seed(42)
            
        v_th, T_max, T_min = threshold_values
        v_th = v_th.type(dtype)
        T_max = T_max.type(dtype)
        T_min = T_min.type(dtype)
        prefire = v_th * 0.0
        
        for shape in test_shapes:
            # Generate input
            x = self.generate_input_data(shape, device, dtype)
            x_copy = x.clone().detach().requires_grad_(True)
            
            # Run both implementations
            pytorch_op = ST_BIFNodeATGF_MS.apply
            cuda_op = ST_BIFNodeATGF_MS_CUDA.apply
            
            spike_seq_pt, v_pt, T_seq_pt = pytorch_op(x, v_th, T_max, T_min, prefire)
            spike_seq_cuda, v_cuda, T_seq_cuda = cuda_op(x_copy, v_th, T_max, T_min, prefire)
            
            # Compare outputs
            self.assert_tensor_close(spike_seq_pt, spike_seq_cuda)
            self.assert_tensor_close(v_pt, v_cuda)
            self.assert_tensor_close(T_seq_pt, T_seq_cuda)

    @pytest.mark.parametrize("use_seed, dtype", 
        [(True, torch.float16), (False, torch.float16), (True, torch.float32), (False, torch.float32)]
    )
    def test_backward_pass(self, test_shapes, threshold_values, device, use_seed, dtype):
        """Test backward pass gradient consistency"""
        if use_seed:
            torch.manual_seed(42)
            
        v_th, T_max, T_min = threshold_values
        v_th = v_th.type(dtype)
        T_max = T_max.type(dtype)
        T_min = T_min.type(dtype)
        prefire = (v_th * 0.0).type(dtype)
        
        for shape in test_shapes:
            # Generate input
            x = self.generate_input_data(shape, device, dtype)
            x_copy = x.clone().detach().requires_grad_(True)
            
            # Forward pass
            pytorch_op = ST_BIFNodeATGF_MS.apply
            cuda_op = ST_BIFNodeATGF_MS_CUDA.apply
            
            spike_seq_pt, v_pt, T_seq_pt = pytorch_op(x, v_th, T_max, T_min, prefire)
            spike_seq_cuda, v_cuda, T_seq_cuda = cuda_op(x_copy, v_th, T_max, T_min, prefire)
            
            # Generate gradient tensors
            grad_spike = torch.randn_like(spike_seq_pt)
            grad_v = torch.randn_like(v_pt)
            grad_T = torch.randn_like(T_seq_pt)
            
            # Backward pass
            spike_seq_pt.backward(grad_spike, retain_graph=True)
            v_pt.backward(grad_v, retain_graph=True)
            T_seq_pt.backward(grad_T)
            
            spike_seq_cuda.backward(grad_spike, retain_graph=True)
            v_cuda.backward(grad_v, retain_graph=True)
            T_seq_cuda.backward(grad_T)
            
            # Compare gradients
            self.assert_tensor_close(spike_seq_pt, spike_seq_cuda, rtol=1e-3, atol=1e-4)
            self.assert_tensor_close(x.grad, x_copy.grad, rtol=1e-3, atol=1e-4)

    @pytest.mark.parametrize("batch_size,feature_size,dtype", [
        (1, 1, torch.float32), (32, 32, torch.float32), (128, 64, torch.float32), (256, 128, torch.float32), 
        (1, 1, torch.float16), (32, 32, torch.float16), (128, 64, torch.float16), (256, 128, torch.float16), 
    ])
    def test_varying_sizes(self, batch_size, feature_size, threshold_values, device, dtype):
        """Test operator with different batch and feature sizes"""
        shape = (10, batch_size, feature_size)  # Fixed time steps
        x = self.generate_input_data(shape, device, dtype)
        x_copy = x.clone().detach().requires_grad_(True)
        v_th, T_max, T_min = threshold_values
        v_th = v_th.type(dtype)
        T_max = T_max.type(dtype)
        T_min = T_min.type(dtype)
        prefire = v_th * 0.0
        
        pytorch_op = ST_BIFNodeATGF_MS.apply
        cuda_op = ST_BIFNodeATGF_MS_CUDA.apply
        
        spike_seq_pt, v_pt, T_seq_pt = pytorch_op(x, v_th, T_max, T_min, prefire)
        spike_seq_cuda, v_cuda, T_seq_cuda = cuda_op(x_copy, v_th, T_max, T_min, prefire)
        
        self.assert_tensor_close(spike_seq_pt, spike_seq_cuda)
        self.assert_tensor_close(v_pt, v_cuda)
        self.assert_tensor_close(T_seq_pt, T_seq_cuda)


    def test_edge_cases(self, device, threshold_values):
        """Test edge cases and potential numerical instabilities"""
        v_th, T_max, T_min = threshold_values
        prefire = v_th * 0.0
        
        edge_cases = [
            torch.zeros((5, 2, 16), device=device),  # All zeros
            torch.ones((5, 2, 16), device=device),   # All ones
            torch.full((5, 2, 16), 0.5, device=device),  # Constant values
            torch.full((5, 2, 16), -0.5, device=device),  # Negative constant
        ]
        
        for x in edge_cases:
            x.requires_grad = True
            x_copy = x.clone().detach().requires_grad_(True)
            
            pytorch_op = ST_BIFNodeATGF_MS.apply
            cuda_op = ST_BIFNodeATGF_MS_CUDA.apply
            
            spike_seq_pt, v_pt, T_seq_pt = pytorch_op(x, v_th, T_max, T_min, prefire)
            spike_seq_cuda, v_cuda, T_seq_cuda = cuda_op(x_copy, v_th, T_max, T_min, prefire)
            
            self.assert_tensor_close(spike_seq_pt, spike_seq_cuda)
            self.assert_tensor_close(v_pt, v_cuda)
            self.assert_tensor_close(T_seq_pt, T_seq_cuda)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])