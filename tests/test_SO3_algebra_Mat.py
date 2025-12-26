"""Unit tests for so3_Mat forward and backward."""
import math
import pytest
import torch
import pypose as pp
from pypose_warp.ltype.SO3_algebra import so3_Mat, so3_Mat_fwd, so3_Mat_bwd
from conftest import get_fwd_tolerances, get_bwd_tolerances, Operator


# =============================================================================
# Forward Pass Tests
# =============================================================================


class TestSo3_Mat_Fwd_BatchDimensions:
    """Test so3_Mat_fwd with various batch dimensions."""

    def test_1d_batch(self, device, dtype):
        """Test with 1D batch dimension."""
        so3 = pp.randn_so3(5, device=device, dtype=dtype)
        
        result = so3_Mat_fwd(so3)
        expected = so3.matrix()
        
        assert result.shape == expected.shape == (5, 3, 3)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.so3_Mat))

    def test_2d_batch(self, device, dtype):
        """Test with 2D batch dimensions."""
        so3 = pp.randn_so3(3, 4, device=device, dtype=dtype)
        
        result = so3_Mat_fwd(so3)
        expected = so3.matrix()
        
        assert result.shape == expected.shape == (3, 4, 3, 3)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.so3_Mat))

    def test_3d_batch(self, device, dtype):
        """Test with 3D batch dimensions."""
        so3 = pp.randn_so3(2, 3, 4, device=device, dtype=dtype)
        
        result = so3_Mat_fwd(so3)
        expected = so3.matrix()
        
        assert result.shape == expected.shape == (2, 3, 4, 3, 3)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.so3_Mat))

    def test_4d_batch(self, device, dtype):
        """Test with 4D batch dimensions."""
        so3 = pp.randn_so3(2, 2, 3, 4, device=device, dtype=dtype)
        
        result = so3_Mat_fwd(so3)
        expected = so3.matrix()
        
        assert result.shape == expected.shape == (2, 2, 3, 4, 3, 3)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.so3_Mat))

    def test_scalar_no_batch(self, device, dtype):
        """Test with no batch dimensions (scalar input)."""
        so3 = pp.randn_so3(device=device, dtype=dtype)
        
        result = so3_Mat_fwd(so3)
        expected = so3.matrix()
        
        assert result.shape == expected.shape == (3, 3)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.so3_Mat))


class TestSo3_Mat_Fwd_Precision:
    """Test so3_Mat_fwd precision handling."""

    def test_precision(self, device, dtype):
        """Test precision for various dtypes."""
        so3 = pp.randn_so3(10, device=device, dtype=dtype)
        
        result = so3_Mat_fwd(so3)
        expected = so3.matrix()
        
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.so3_Mat))

    def test_output_dtype_preserved(self, device, dtype):
        """Test that output dtype matches input dtype."""
        so3 = pp.randn_so3(5, device=device, dtype=dtype)
        
        result = so3_Mat_fwd(so3)
        assert result.dtype == dtype


class TestSo3_Mat_Fwd_EdgeCases:
    """Test so3_Mat_fwd with edge cases."""

    def test_identity_rotation(self, device, dtype):
        """Test with identity rotation (zero axis-angle)."""
        so3 = pp.identity_so3(5, device=device, dtype=dtype)
        
        result = so3_Mat_fwd(so3)
        expected = torch.eye(3, device=device, dtype=dtype).expand(5, 3, 3)
        
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.so3_Mat))

    def test_small_angle(self, device):
        """Test with small angles (near zero, test Taylor expansion)."""
        dtype = torch.float64
        # Small axis-angle vector (near identity)
        so3_tensor = torch.tensor([[1e-8, 0., 0.]], device=device, dtype=dtype)
        so3 = pp.LieTensor(so3_tensor, ltype=pp.so3_type)
        
        result = so3_Mat_fwd(so3)
        expected = so3.matrix()
        
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_180_degree_rotation_x(self, device):
        """Test 180 degree rotation around x-axis."""
        dtype = torch.float64
        # 180 degree rotation: axis-angle = pi * [1, 0, 0]
        so3_tensor = torch.tensor([[math.pi, 0., 0.]], device=device, dtype=dtype)
        so3 = pp.LieTensor(so3_tensor, ltype=pp.so3_type)
        
        result = so3_Mat_fwd(so3)
        expected = torch.tensor([[[1., 0., 0.],
                                   [0., -1., 0.],
                                   [0., 0., -1.]]], device=device, dtype=dtype)
        
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_180_degree_rotation_y(self, device):
        """Test 180 degree rotation around y-axis."""
        dtype = torch.float64
        so3_tensor = torch.tensor([[0., math.pi, 0.]], device=device, dtype=dtype)
        so3 = pp.LieTensor(so3_tensor, ltype=pp.so3_type)
        
        result = so3_Mat_fwd(so3)
        expected = torch.tensor([[[-1., 0., 0.],
                                   [0., 1., 0.],
                                   [0., 0., -1.]]], device=device, dtype=dtype)
        
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_180_degree_rotation_z(self, device):
        """Test 180 degree rotation around z-axis."""
        dtype = torch.float64
        so3_tensor = torch.tensor([[0., 0., math.pi]], device=device, dtype=dtype)
        so3 = pp.LieTensor(so3_tensor, ltype=pp.so3_type)
        
        result = so3_Mat_fwd(so3)
        expected = torch.tensor([[[-1., 0., 0.],
                                   [0., -1., 0.],
                                   [0., 0., 1.]]], device=device, dtype=dtype)
        
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_90_degree_rotation_x(self, device):
        """Test 90 degree rotation around x-axis."""
        dtype = torch.float64
        so3_tensor = torch.tensor([[math.pi/2, 0., 0.]], device=device, dtype=dtype)
        so3 = pp.LieTensor(so3_tensor, ltype=pp.so3_type)
        
        result = so3_Mat_fwd(so3)
        expected = torch.tensor([[[1., 0., 0.],
                                   [0., 0., -1.],
                                   [0., 1., 0.]]], device=device, dtype=dtype)
        
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_rotation_matrix_is_orthogonal(self, device):
        """Test that output rotation matrices are orthogonal (R^T R = I)."""
        dtype = torch.float64
        so3 = pp.randn_so3(10, device=device, dtype=dtype)
        
        result = so3_Mat_fwd(so3)
        
        # R^T @ R should be identity
        RtR = torch.bmm(result.transpose(-2, -1), result)
        I = torch.eye(3, device=device, dtype=dtype).expand(10, 3, 3)
        
        torch.testing.assert_close(RtR, I, atol=1e-10, rtol=1e-10)

    def test_rotation_matrix_determinant_is_one(self, device):
        """Test that output rotation matrices have determinant 1."""
        dtype = torch.float64
        so3 = pp.randn_so3(10, device=device, dtype=dtype)
        
        result = so3_Mat_fwd(so3)
        det = torch.linalg.det(result)
        
        torch.testing.assert_close(det, torch.ones(10, device=device, dtype=dtype), atol=1e-10, rtol=1e-10)

    def test_single_element_batch(self, device, dtype):
        """Test with a batch of size 1."""
        so3 = pp.randn_so3(1, device=device, dtype=dtype)
        
        result = so3_Mat_fwd(so3)
        expected = so3.matrix()
        
        assert result.shape == expected.shape == (1, 3, 3)
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.so3_Mat))

    def test_large_batch(self, device, dtype):
        """Test with a large batch size."""
        so3 = pp.randn_so3(1000, device=device, dtype=dtype)
        
        result = so3_Mat_fwd(so3)
        expected = so3.matrix()
        
        assert result.shape == expected.shape == (1000, 3, 3)
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.so3_Mat))

    def test_output_device(self, device, dtype):
        """Test that output device matches input device."""
        so3 = pp.randn_so3(5, device=device, dtype=dtype)
        
        result = so3_Mat_fwd(so3)
        assert str(result.device).startswith(device)


class TestSo3_Mat_Fwd_Errors:
    """Test so3_Mat_fwd error handling."""

    def test_5d_batch_raises(self, device, dtype):
        """Test that 5D batch dimensions raise NotImplementedError."""
        so3 = pp.randn_so3(2, 2, 2, 2, 2, device=device, dtype=dtype)
        
        with pytest.raises(NotImplementedError, match="Batch dimensions > 4"):
            so3_Mat_fwd(so3)


# =============================================================================
# Backward Pass Tests (use dtype_bwd fixture - no fp16)
# =============================================================================


class TestSo3_Mat_Bwd_Gradcheck:
    """Test so3_Mat backward with torch.autograd.gradcheck.
    
    Note: We use regular tensors for gradcheck since LieTensor wrapping
    can interfere with autograd's double-backward detection.
    """

    def test_gradcheck_1d(self, device):
        """Test gradcheck with 1D batch."""
        dtype = torch.float64
        so3 = pp.randn_so3(3, device=device, dtype=dtype, requires_grad=True)
        
        def func(x):
            return so3_Mat.apply(pp.LieTensor(x, ltype=pp.so3_type))
        
        torch.autograd.gradcheck(func, (so3.tensor(),), atol=1e-6, rtol=1e-6)

    def test_gradcheck_2d(self, device):
        """Test gradcheck with 2D batch."""
        dtype = torch.float64
        so3 = pp.randn_so3(2, 3, device=device, dtype=dtype, requires_grad=True)
        
        def func(x):
            return so3_Mat.apply(pp.LieTensor(x, ltype=pp.so3_type))
        
        torch.autograd.gradcheck(func, (so3.tensor(),), atol=1e-6, rtol=1e-6)

    def test_gradcheck_scalar(self, device):
        """Test gradcheck with scalar input."""
        dtype = torch.float64
        so3 = pp.randn_so3(device=device, dtype=dtype, requires_grad=True)
        
        def func(x):
            return so3_Mat.apply(pp.LieTensor(x, ltype=pp.so3_type))
        
        torch.autograd.gradcheck(func, (so3.tensor(),), atol=1e-6, rtol=1e-6)


class TestSo3_Mat_Bwd_EdgeCases:
    """Test so3_Mat backward with edge cases."""

    def test_identity_rotation_gradient(self, device):
        """Test backward with identity rotation."""
        dtype = torch.float64
        so3 = pp.identity_so3(5, device=device, dtype=dtype)
        so3.requires_grad_(True)
        
        result = so3_Mat.apply(so3)
        result.sum().backward()
        
        # For identity (zero axis-angle), the gradients should be finite
        assert not torch.isnan(so3.grad).any()
        assert not torch.isinf(so3.grad).any()

    def test_small_angle_gradient(self, device):
        """Test backward with small angles (test numerical stability)."""
        dtype = torch.float64
        so3_tensor = torch.tensor([[1e-8, 0., 0.]], device=device, dtype=dtype, requires_grad=True)
        so3 = pp.LieTensor(so3_tensor, ltype=pp.so3_type)
        
        result = so3_Mat.apply(so3)
        result.sum().backward()
        
        # Access gradient on the leaf tensor, not the LieTensor
        assert not torch.isnan(so3_tensor.grad).any()
        assert not torch.isinf(so3_tensor.grad).any()

    def test_grad_dtype_preserved(self, device, dtype_bwd):
        """Test that gradient dtype matches input dtype."""
        so3 = pp.randn_so3(5, device=device, dtype=dtype_bwd)
        so3.requires_grad_(True)
        
        result = so3_Mat.apply(so3)
        result.sum().backward()
        
        assert so3.grad.dtype == dtype_bwd

    def test_grad_device_preserved(self, device, dtype_bwd):
        """Test that gradient device matches input device."""
        so3 = pp.randn_so3(5, device=device, dtype=dtype_bwd)
        so3.requires_grad_(True)
        
        result = so3_Mat.apply(so3)
        result.sum().backward()
        
        assert str(so3.grad.device).startswith(device)

    def test_large_batch_gradient(self, device, dtype_bwd):
        """Test backward with a large batch size."""
        so3 = pp.randn_so3(1000, device=device, dtype=dtype_bwd)
        so3.requires_grad_(True)
        
        result = so3_Mat.apply(so3)
        result.sum().backward()
        
        assert so3.grad.shape == (1000, 3)
        assert not torch.isnan(so3.grad).any()


class TestSo3_Mat_Bwd_NumericalGradient:
    """Test so3_Mat backward against numerical gradient (finite differences)."""

    def test_numerical_gradient_match(self, device):
        """Test that analytical gradient matches numerical gradient."""
        dtype = torch.float64
        # Create a leaf tensor directly to avoid non-leaf gradient issues
        so3_tensor = torch.randn(5, 3, device=device, dtype=dtype, requires_grad=True)
        
        # Analytical gradient
        result = so3_Mat.apply(pp.LieTensor(so3_tensor, ltype=pp.so3_type))
        loss = result.sum()
        loss.backward()
        analytical_grad = so3_tensor.grad.clone()
        
        # Numerical gradient via finite differences
        eps = 1e-6
        numerical_grad = torch.zeros_like(so3_tensor)
        
        for i in range(5):
            for j in range(3):
                so3_plus = so3_tensor.detach().clone()
                so3_plus[i, j] += eps
                result_plus = so3_Mat.apply(pp.LieTensor(so3_plus, ltype=pp.so3_type))
                loss_plus = result_plus.sum()
                
                so3_minus = so3_tensor.detach().clone()
                so3_minus[i, j] -= eps
                result_minus = so3_Mat.apply(pp.LieTensor(so3_minus, ltype=pp.so3_type))
                loss_minus = result_minus.sum()
                
                numerical_grad[i, j] = (loss_plus - loss_minus) / (2 * eps)
        
        torch.testing.assert_close(analytical_grad, numerical_grad, atol=1e-5, rtol=1e-5)


class TestSo3_Mat_WarpBackendIntegration:
    """Test so3_Mat integration with warp backend LieTensor."""

    def test_warp_backend_so3_matrix(self, device, dtype):
        """Test matrix() method on warp backend so3 LieTensor."""
        from pypose_warp import to_warp_backend, to_pypose_backend
        
        so3 = pp.randn_so3(5, device=device, dtype=dtype)
        wp_so3 = to_warp_backend(so3)
        
        result = wp_so3.matrix()
        expected = so3.matrix()
        
        assert result.shape == expected.shape == (5, 3, 3)
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.so3_Mat))

    def test_warp_backend_backward(self, device, dtype_bwd):
        """Test backward through warp backend matrix."""
        from pypose_warp import to_warp_backend
        
        so3_tensor = torch.randn(5, 3, device=device, dtype=dtype_bwd, requires_grad=True)
        so3 = pp.LieTensor(so3_tensor, ltype=pp.so3_type)
        wp_so3 = to_warp_backend(so3)
        
        # Use warp backend matrix
        result = wp_so3.matrix()
        loss = result.sum()
        loss.backward()
        
        assert so3_tensor.grad is not None
        assert so3_tensor.grad.shape == (5, 3)
        assert not torch.isnan(so3_tensor.grad).any()
