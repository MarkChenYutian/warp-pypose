"""Unit tests for se3_Mat forward and backward."""
import math
import pytest
import torch
import pypose as pp
from pypose_warp.ltype.SE3_algebra import se3_Mat, se3_Mat_fwd, se3_Mat_bwd
from conftest import get_fwd_tolerances, get_bwd_tolerances, Operator


# =============================================================================
# Forward Pass Tests
# =============================================================================


class TestSe3_Mat_Fwd_BatchDimensions:
    """Test se3_Mat_fwd with various batch dimensions."""

    def test_1d_batch(self, device, dtype):
        """Test with 1D batch dimension."""
        se3 = pp.randn_se3(5, device=device, dtype=dtype)
        
        result = se3_Mat_fwd(se3)
        expected = se3.matrix()
        
        assert result.shape == expected.shape == (5, 4, 4)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.se3_Mat))

    def test_2d_batch(self, device, dtype):
        """Test with 2D batch dimensions."""
        se3 = pp.randn_se3(3, 4, device=device, dtype=dtype)
        
        result = se3_Mat_fwd(se3)
        expected = se3.matrix()
        
        assert result.shape == expected.shape == (3, 4, 4, 4)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.se3_Mat))

    def test_3d_batch(self, device, dtype):
        """Test with 3D batch dimensions."""
        se3 = pp.randn_se3(2, 3, 4, device=device, dtype=dtype)
        
        result = se3_Mat_fwd(se3)
        expected = se3.matrix()
        
        assert result.shape == expected.shape == (2, 3, 4, 4, 4)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.se3_Mat))

    def test_4d_batch(self, device, dtype):
        """Test with 4D batch dimensions."""
        se3 = pp.randn_se3(2, 2, 3, 4, device=device, dtype=dtype)
        
        result = se3_Mat_fwd(se3)
        expected = se3.matrix()
        
        assert result.shape == expected.shape == (2, 2, 3, 4, 4, 4)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.se3_Mat))

    def test_scalar_no_batch(self, device, dtype):
        """Test with no batch dimensions (scalar input)."""
        se3 = pp.randn_se3(device=device, dtype=dtype)
        
        result = se3_Mat_fwd(se3)
        expected = se3.matrix()
        
        assert result.shape == expected.shape == (4, 4)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.se3_Mat))


class TestSe3_Mat_Fwd_Precision:
    """Test se3_Mat_fwd precision handling."""

    def test_precision(self, device, dtype):
        """Test precision for various dtypes."""
        # Use smaller random values for fp16 to avoid NaN from numerical overflow
        if dtype == torch.float16:
            se3_tensor = torch.randn(10, 6, device=device, dtype=dtype) * 0.5
            se3 = pp.LieTensor(se3_tensor, ltype=pp.se3_type)
        else:
            se3 = pp.randn_se3(10, device=device, dtype=dtype)
        
        result = se3_Mat_fwd(se3)
        expected = se3.matrix()
        
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.se3_Mat))

    def test_output_dtype_preserved(self, device, dtype):
        """Test that output dtype matches input dtype."""
        se3 = pp.randn_se3(5, device=device, dtype=dtype)
        
        result = se3_Mat_fwd(se3)
        assert result.dtype == dtype


class TestSe3_Mat_Fwd_EdgeCases:
    """Test se3_Mat_fwd with edge cases."""

    def test_identity_twist(self, device, dtype):
        """Test with identity twist (zero se3)."""
        se3 = pp.identity_se3(5, device=device, dtype=dtype)
        
        result = se3_Mat_fwd(se3)
        expected = torch.eye(4, device=device, dtype=dtype).expand(5, 4, 4)
        
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.se3_Mat))

    def test_small_twist(self, device):
        """Test with small twists (near zero, test Taylor expansion)."""
        dtype = torch.float64
        # Small twist vector
        se3_tensor = torch.tensor([[1e-8, 0., 0., 1e-8, 0., 0.]], device=device, dtype=dtype)
        se3 = pp.LieTensor(se3_tensor, ltype=pp.se3_type)
        
        result = se3_Mat_fwd(se3)
        expected = se3.matrix()
        
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_pure_translation(self, device):
        """Test with pure translation (zero rotation)."""
        dtype = torch.float64
        # Pure translation: tau = [1, 2, 3], phi = [0, 0, 0]
        se3_tensor = torch.tensor([[1., 2., 3., 0., 0., 0.]], device=device, dtype=dtype)
        se3 = pp.LieTensor(se3_tensor, ltype=pp.se3_type)
        
        result = se3_Mat_fwd(se3)
        expected = torch.tensor([[[1., 0., 0., 1.],
                                   [0., 1., 0., 2.],
                                   [0., 0., 1., 3.],
                                   [0., 0., 0., 1.]]], device=device, dtype=dtype)
        
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_pure_rotation_180_x(self, device):
        """Test 180 degree rotation around x-axis."""
        dtype = torch.float64
        # 180 degree rotation with no translation
        se3_tensor = torch.tensor([[0., 0., 0., math.pi, 0., 0.]], device=device, dtype=dtype)
        se3 = pp.LieTensor(se3_tensor, ltype=pp.se3_type)
        
        result = se3_Mat_fwd(se3)
        expected = se3.matrix()
        
        # Verify rotation submatrix
        torch.testing.assert_close(result[:, :3, :3], expected[:, :3, :3], atol=1e-10, rtol=1e-10)
        # Translation should be zero
        torch.testing.assert_close(result[:, :3, 3], expected[:, :3, 3], atol=1e-10, rtol=1e-10)

    def test_rotation_matrix_is_orthogonal(self, device):
        """Test that output rotation matrices are orthogonal (R^T R = I)."""
        dtype = torch.float64
        se3 = pp.randn_se3(10, device=device, dtype=dtype)
        
        result = se3_Mat_fwd(se3)
        R = result[:, :3, :3]  # Extract rotation block
        
        # R^T @ R should be identity
        RtR = torch.bmm(R.transpose(-2, -1), R)
        I = torch.eye(3, device=device, dtype=dtype).expand(10, 3, 3)
        
        torch.testing.assert_close(RtR, I, atol=1e-10, rtol=1e-10)

    def test_bottom_row_is_correct(self, device, dtype):
        """Test that bottom row is [0, 0, 0, 1]."""
        se3 = pp.randn_se3(10, device=device, dtype=dtype)
        
        result = se3_Mat_fwd(se3)
        bottom_row = result[:, 3, :]
        expected = torch.tensor([0., 0., 0., 1.], device=device, dtype=dtype).expand(10, 4)
        
        torch.testing.assert_close(bottom_row, expected, atol=1e-10, rtol=1e-10)

    def test_single_element_batch(self, device, dtype):
        """Test with a batch of size 1."""
        se3 = pp.randn_se3(1, device=device, dtype=dtype)
        
        result = se3_Mat_fwd(se3)
        expected = se3.matrix()
        
        assert result.shape == expected.shape == (1, 4, 4)
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.se3_Mat))

    def test_large_batch(self, device, dtype):
        """Test with a large batch size."""
        # Use smaller random values for fp16 to avoid NaN from numerical overflow
        if dtype == torch.float16:
            se3_tensor = torch.randn(1000, 6, device=device, dtype=dtype) * 0.5
            se3 = pp.LieTensor(se3_tensor, ltype=pp.se3_type)
        else:
            se3 = pp.randn_se3(1000, device=device, dtype=dtype)
        
        result = se3_Mat_fwd(se3)
        expected = se3.matrix()
        
        assert result.shape == expected.shape == (1000, 4, 4)
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.se3_Mat))

    def test_output_device(self, device, dtype):
        """Test that output device matches input device."""
        se3 = pp.randn_se3(5, device=device, dtype=dtype)
        
        result = se3_Mat_fwd(se3)
        assert str(result.device).startswith(device)


class TestSe3_Mat_Fwd_Errors:
    """Test se3_Mat_fwd error handling."""

    def test_5d_batch_raises(self, device, dtype):
        """Test that 5D batch dimensions raise NotImplementedError."""
        se3 = pp.randn_se3(2, 2, 2, 2, 2, device=device, dtype=dtype)
        
        with pytest.raises(NotImplementedError, match="Batch dimensions > 4"):
            se3_Mat_fwd(se3)


# =============================================================================
# Backward Pass Tests (use dtype_bwd fixture - no fp16)
# =============================================================================


class TestSe3_Mat_Bwd_Gradcheck:
    """Test se3_Mat backward with torch.autograd.gradcheck."""

    def test_gradcheck_1d(self, device):
        """Test gradcheck with 1D batch."""
        dtype = torch.float64
        se3 = pp.randn_se3(3, device=device, dtype=dtype, requires_grad=True)
        
        def func(x):
            return se3_Mat.apply(pp.LieTensor(x, ltype=pp.se3_type))
        
        torch.autograd.gradcheck(func, (se3.tensor(),), atol=1e-6, rtol=1e-6)

    def test_gradcheck_2d(self, device):
        """Test gradcheck with 2D batch."""
        dtype = torch.float64
        se3 = pp.randn_se3(2, 3, device=device, dtype=dtype, requires_grad=True)
        
        def func(x):
            return se3_Mat.apply(pp.LieTensor(x, ltype=pp.se3_type))
        
        torch.autograd.gradcheck(func, (se3.tensor(),), atol=1e-6, rtol=1e-6)

    def test_gradcheck_scalar(self, device):
        """Test gradcheck with scalar input."""
        dtype = torch.float64
        se3 = pp.randn_se3(device=device, dtype=dtype, requires_grad=True)
        
        def func(x):
            return se3_Mat.apply(pp.LieTensor(x, ltype=pp.se3_type))
        
        torch.autograd.gradcheck(func, (se3.tensor(),), atol=1e-6, rtol=1e-6)


class TestSe3_Mat_Bwd_EdgeCases:
    """Test se3_Mat backward with edge cases."""

    def test_identity_twist_gradient(self, device):
        """Test backward with identity twist."""
        dtype = torch.float64
        se3 = pp.identity_se3(5, device=device, dtype=dtype)
        se3.requires_grad_(True)
        
        result = se3_Mat.apply(se3)
        result.sum().backward()
        
        # For identity, the gradients should be finite
        assert not torch.isnan(se3.grad).any()
        assert not torch.isinf(se3.grad).any()

    def test_small_twist_gradient(self, device):
        """Test backward with small twists (test numerical stability)."""
        dtype = torch.float64
        se3_tensor = torch.tensor([[1e-8, 0., 0., 1e-8, 0., 0.]], device=device, dtype=dtype, requires_grad=True)
        se3 = pp.LieTensor(se3_tensor, ltype=pp.se3_type)
        
        result = se3_Mat.apply(se3)
        result.sum().backward()
        
        assert not torch.isnan(se3_tensor.grad).any()
        assert not torch.isinf(se3_tensor.grad).any()

    def test_grad_dtype_preserved(self, device, dtype_bwd):
        """Test that gradient dtype matches input dtype."""
        se3 = pp.randn_se3(5, device=device, dtype=dtype_bwd)
        se3.requires_grad_(True)
        
        result = se3_Mat.apply(se3)
        result.sum().backward()
        
        assert se3.grad.dtype == dtype_bwd

    def test_grad_device_preserved(self, device, dtype_bwd):
        """Test that gradient device matches input device."""
        se3 = pp.randn_se3(5, device=device, dtype=dtype_bwd)
        se3.requires_grad_(True)
        
        result = se3_Mat.apply(se3)
        result.sum().backward()
        
        assert str(se3.grad.device).startswith(device)

    def test_large_batch_gradient(self, device, dtype_bwd):
        """Test backward with a large batch size."""
        se3 = pp.randn_se3(1000, device=device, dtype=dtype_bwd)
        se3.requires_grad_(True)
        
        result = se3_Mat.apply(se3)
        result.sum().backward()
        
        assert se3.grad.shape == (1000, 6)
        assert not torch.isnan(se3.grad).any()


class TestSe3_Mat_Bwd_MatchesPyPose:
    """Test that se3_Mat backward matches PyPose's matrix() backward."""

    def test_gradient_matches_pypose(self, device, dtype_bwd):
        """Test that warp backward matches PyPose backward."""
        # Create leaf tensors for gradient tracking
        se3_tensor_native = torch.randn(5, 6, device=device, dtype=dtype_bwd, requires_grad=True)
        se3_tensor_warp = se3_tensor_native.clone().detach().requires_grad_(True)
        
        # PyPose native backward
        se3_native = pp.LieTensor(se3_tensor_native, ltype=pp.se3_type)
        result_native = se3_native.matrix()
        loss_native = result_native.sum()
        loss_native.backward()
        native_grad = se3_tensor_native.grad.clone()
        
        # Warp backward
        se3_warp = pp.LieTensor(se3_tensor_warp, ltype=pp.se3_type)
        result_warp = se3_Mat.apply(se3_warp)
        loss_warp = result_warp.sum()
        loss_warp.backward()
        warp_grad = se3_tensor_warp.grad.clone()
        
        torch.testing.assert_close(warp_grad, native_grad, **get_bwd_tolerances(dtype_bwd, Operator.se3_Mat))


class TestSe3_Mat_Bwd_NumericalGradient:
    """Test se3_Mat backward against numerical gradient (finite differences)."""

    def test_numerical_gradient_match(self, device):
        """Test that analytical gradient matches numerical gradient."""
        dtype = torch.float64
        se3_tensor = torch.randn(5, 6, device=device, dtype=dtype, requires_grad=True)
        
        # Analytical gradient
        result = se3_Mat.apply(pp.LieTensor(se3_tensor, ltype=pp.se3_type))
        loss = result.sum()
        loss.backward()
        analytical_grad = se3_tensor.grad.clone()
        
        # Numerical gradient via finite differences
        eps = 1e-6
        numerical_grad = torch.zeros_like(se3_tensor)
        
        for i in range(5):
            for j in range(6):
                se3_plus = se3_tensor.detach().clone()
                se3_plus[i, j] += eps
                result_plus = se3_Mat.apply(pp.LieTensor(se3_plus, ltype=pp.se3_type))
                loss_plus = result_plus.sum()
                
                se3_minus = se3_tensor.detach().clone()
                se3_minus[i, j] -= eps
                result_minus = se3_Mat.apply(pp.LieTensor(se3_minus, ltype=pp.se3_type))
                loss_minus = result_minus.sum()
                
                numerical_grad[i, j] = (loss_plus - loss_minus) / (2 * eps)
        
        torch.testing.assert_close(analytical_grad, numerical_grad, atol=1e-5, rtol=1e-5)


class TestSe3_Mat_WarpBackendIntegration:
    """Test se3_Mat integration with warp backend LieTensor."""

    def test_warp_backend_se3_matrix(self, device, dtype):
        """Test matrix() method on warp backend se3 LieTensor."""
        from pypose_warp import to_warp_backend, to_pypose_backend
        
        se3 = pp.randn_se3(5, device=device, dtype=dtype)
        wp_se3 = to_warp_backend(se3)
        
        result = wp_se3.matrix()
        expected = se3.matrix()
        
        assert result.shape == expected.shape == (5, 4, 4)
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.se3_Mat))

    def test_warp_backend_backward(self, device, dtype_bwd):
        """Test backward through warp backend matrix."""
        from pypose_warp import to_warp_backend
        
        se3_tensor = torch.randn(5, 6, device=device, dtype=dtype_bwd, requires_grad=True)
        se3 = pp.LieTensor(se3_tensor, ltype=pp.se3_type)
        wp_se3 = to_warp_backend(se3)
        
        # Use warp backend matrix
        result = wp_se3.matrix()
        loss = result.sum()
        loss.backward()
        
        assert se3_tensor.grad is not None
        assert se3_tensor.grad.shape == (5, 6)
        assert not torch.isnan(se3_tensor.grad).any()

