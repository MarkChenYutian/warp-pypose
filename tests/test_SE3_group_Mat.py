"""Unit tests for SE3_Mat forward and backward."""
import pytest
import torch
import pypose as pp
from pypose_warp.ltype.SE3_group import SE3_Mat, SE3_Mat_fwd
from pypose_warp import to_warp_backend
from conftest import get_fwd_tolerances, get_bwd_tolerances, Operator


# =============================================================================
# Forward Pass Tests
# =============================================================================


class TestSE3MatBatchDimensions:
    """Test SE3_Mat_fwd with various batch dimensions."""

    def test_1d_batch(self, device, dtype):
        """Test with 1D batch dimension."""
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        X_warp = to_warp_backend(X)
        
        result = SE3_Mat_fwd(X_warp)
        expected = X.matrix()
        
        assert result.shape == expected.shape == (5, 4, 4)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype))

    def test_2d_batch(self, device, dtype):
        """Test with 2D batch dimensions."""
        X = pp.randn_SE3(3, 4, device=device, dtype=dtype)
        X_warp = to_warp_backend(X)
        
        result = SE3_Mat_fwd(X_warp)
        expected = X.matrix()
        
        assert result.shape == expected.shape == (3, 4, 4, 4)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype))

    def test_3d_batch(self, device, dtype):
        """Test with 3D batch dimensions."""
        X = pp.randn_SE3(2, 3, 4, device=device, dtype=dtype)
        X_warp = to_warp_backend(X)
        
        result = SE3_Mat_fwd(X_warp)
        expected = X.matrix()
        
        assert result.shape == expected.shape == (2, 3, 4, 4, 4)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype))

    def test_4d_batch(self, device, dtype):
        """Test with 4D batch dimensions."""
        X = pp.randn_SE3(2, 2, 3, 4, device=device, dtype=dtype)
        X_warp = to_warp_backend(X)
        
        result = SE3_Mat_fwd(X_warp)
        expected = X.matrix()
        
        assert result.shape == expected.shape == (2, 2, 3, 4, 4, 4)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype))

    def test_scalar_no_batch(self, device, dtype):
        """Test with no batch dimensions (scalar input)."""
        X = pp.randn_SE3(device=device, dtype=dtype)
        X_warp = to_warp_backend(X)
        
        result = SE3_Mat_fwd(X_warp)
        expected = X.matrix()
        
        assert result.shape == expected.shape == (4, 4)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype))


class TestSE3MatPrecision:
    """Test SE3_Mat_fwd precision handling."""

    def test_precision(self, device, dtype):
        """Test precision for various dtypes."""
        X = pp.randn_SE3(10, device=device, dtype=dtype)
        X_warp = to_warp_backend(X)
        
        result = SE3_Mat_fwd(X_warp)
        expected = X.matrix()
        
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype))

    def test_output_dtype_preserved(self, device, dtype):
        """Test that output dtype matches input dtype."""
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        X_warp = to_warp_backend(X)
        
        result = SE3_Mat_fwd(X_warp)
        assert result.dtype == dtype


class TestSE3MatEdgeCases:
    """Test SE3_Mat_fwd with edge cases."""

    def test_identity_transform(self, device, dtype):
        """Test with identity transformation."""
        X = pp.identity_SE3(5, device=device, dtype=dtype)
        X_warp = to_warp_backend(X)
        
        result = SE3_Mat_fwd(X_warp)
        expected = torch.eye(4, device=device, dtype=dtype).expand(5, 4, 4)
        
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype))

    def test_pure_translation(self, device):
        """Test with pure translation (identity rotation)."""
        dtype = torch.float64
        # SE3: [tx, ty, tz, qx, qy, qz, qw] = [1, 2, 3, 0, 0, 0, 1]
        t = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]], device=device, dtype=dtype)
        X = pp.LieTensor(t, ltype=pp.SE3_type)
        X_warp = to_warp_backend(X)
        
        result = SE3_Mat_fwd(X_warp)
        expected = torch.tensor([[[1., 0., 0., 1.],
                                   [0., 1., 0., 2.],
                                   [0., 0., 1., 3.],
                                   [0., 0., 0., 1.]]], device=device, dtype=dtype)
        
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_pure_rotation_180_x(self, device):
        """Test with 180 degree rotation around x-axis, no translation."""
        dtype = torch.float64
        # SE3: [tx, ty, tz, qx, qy, qz, qw] = [0, 0, 0, 1, 0, 0, 0]
        t = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype)
        X = pp.LieTensor(t, ltype=pp.SE3_type)
        X_warp = to_warp_backend(X)
        
        result = SE3_Mat_fwd(X_warp)
        expected = torch.tensor([[[1., 0., 0., 0.],
                                   [0., -1., 0., 0.],
                                   [0., 0., -1., 0.],
                                   [0., 0., 0., 1.]]], device=device, dtype=dtype)
        
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_rotation_translation_combined(self, device):
        """Test with combined rotation and translation."""
        dtype = torch.float64
        # 90 degree rotation around z, with translation (1, 2, 3)
        import math
        c = math.cos(math.pi / 4)  # cos(45°) for quat
        s = math.sin(math.pi / 4)  # sin(45°) for quat
        # Quaternion for 90° around z: (0, 0, sin(45°), cos(45°))
        t = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, s, c]], device=device, dtype=dtype)
        X = pp.LieTensor(t, ltype=pp.SE3_type)
        X_warp = to_warp_backend(X)
        
        result = SE3_Mat_fwd(X_warp)
        expected = X.matrix()
        
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_rotation_submatrix_is_orthogonal(self, device):
        """Test that rotation part of the matrix is orthogonal (R^T R = I)."""
        dtype = torch.float64
        X = pp.randn_SE3(10, device=device, dtype=dtype)
        X_warp = to_warp_backend(X)
        
        result = SE3_Mat_fwd(X_warp)
        
        # Extract 3x3 rotation submatrix
        R = result[:, :3, :3]
        
        # R^T @ R should be identity
        RtR = torch.bmm(R.transpose(-2, -1), R)
        I = torch.eye(3, device=device, dtype=dtype).expand(10, 3, 3)
        
        torch.testing.assert_close(RtR, I, atol=1e-10, rtol=1e-10)

    def test_rotation_determinant_is_one(self, device):
        """Test that rotation submatrix has determinant 1."""
        dtype = torch.float64
        X = pp.randn_SE3(10, device=device, dtype=dtype)
        X_warp = to_warp_backend(X)
        
        result = SE3_Mat_fwd(X_warp)
        
        # Extract 3x3 rotation submatrix
        R = result[:, :3, :3]
        det = torch.linalg.det(R)
        
        torch.testing.assert_close(det, torch.ones(10, device=device, dtype=dtype), atol=1e-10, rtol=1e-10)

    def test_last_row_is_0001(self, device, dtype):
        """Test that the last row is [0, 0, 0, 1]."""
        X = pp.randn_SE3(10, device=device, dtype=dtype)
        X_warp = to_warp_backend(X)
        
        result = SE3_Mat_fwd(X_warp)
        
        last_row = result[:, 3, :]
        expected = torch.tensor([0., 0., 0., 1.], device=device, dtype=dtype).expand(10, 4)
        
        torch.testing.assert_close(last_row, expected, **get_fwd_tolerances(dtype))

    def test_translation_in_last_column(self, device, dtype):
        """Test that translation appears in the last column."""
        X = pp.randn_SE3(10, device=device, dtype=dtype)
        X_warp = to_warp_backend(X)
        
        result = SE3_Mat_fwd(X_warp)
        
        # Translation should be in [:, :3, 3]
        translation_from_matrix = result[:, :3, 3]
        translation_from_X = X.tensor()[:, :3]
        
        torch.testing.assert_close(translation_from_matrix, translation_from_X, **get_fwd_tolerances(dtype))

    def test_single_element_batch(self, device, dtype):
        """Test with a batch of size 1."""
        X = pp.randn_SE3(1, device=device, dtype=dtype)
        X_warp = to_warp_backend(X)
        
        result = SE3_Mat_fwd(X_warp)
        expected = X.matrix()
        
        assert result.shape == expected.shape == (1, 4, 4)
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype))

    def test_large_batch(self, device, dtype):
        """Test with a large batch size."""
        X = pp.randn_SE3(1000, device=device, dtype=dtype)
        X_warp = to_warp_backend(X)
        
        result = SE3_Mat_fwd(X_warp)
        expected = X.matrix()
        
        assert result.shape == expected.shape == (1000, 4, 4)
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype))

    def test_output_device(self, device, dtype):
        """Test that output device matches input device."""
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        X_warp = to_warp_backend(X)
        
        result = SE3_Mat_fwd(X_warp)
        assert str(result.device).startswith(device)


class TestSE3MatErrors:
    """Test SE3_Mat_fwd error handling."""

    def test_5d_batch_raises(self, device, dtype):
        """Test that 5D batch dimensions raise NotImplementedError."""
        X = pp.randn_SE3(2, 2, 2, 2, 2, device=device, dtype=dtype)
        X_warp = to_warp_backend(X)
        
        with pytest.raises(NotImplementedError, match="Batch dimensions > 4"):
            SE3_Mat_fwd(X_warp)


# =============================================================================
# Backward Pass Tests
# =============================================================================


class TestSE3MatBwdGradcheck:
    """Test SE3_Mat backward with torch.autograd.gradcheck."""

    def test_gradcheck_1d(self, device):
        """Test gradcheck with 1D batch."""
        dtype = torch.float64
        X = pp.randn_SE3(3, device=device, dtype=dtype, requires_grad=True)
        X_warp = to_warp_backend(X)
        
        def func(x):
            return SE3_Mat.apply(pp.LieTensor(x, ltype=X_warp.ltype))
        
        torch.autograd.gradcheck(func, (X_warp.tensor(),), atol=1e-6, rtol=1e-6)

    def test_gradcheck_2d(self, device):
        """Test gradcheck with 2D batch."""
        dtype = torch.float64
        X = pp.randn_SE3(2, 3, device=device, dtype=dtype, requires_grad=True)
        X_warp = to_warp_backend(X)
        
        def func(x):
            return SE3_Mat.apply(pp.LieTensor(x, ltype=X_warp.ltype))
        
        torch.autograd.gradcheck(func, (X_warp.tensor(),), atol=1e-6, rtol=1e-6)

    def test_gradcheck_scalar(self, device):
        """Test gradcheck with scalar input."""
        dtype = torch.float64
        X = pp.randn_SE3(device=device, dtype=dtype, requires_grad=True)
        X_warp = to_warp_backend(X)
        
        def func(x):
            return SE3_Mat.apply(pp.LieTensor(x, ltype=X_warp.ltype))
        
        torch.autograd.gradcheck(func, (X_warp.tensor(),), atol=1e-6, rtol=1e-6)


class TestSE3MatBwdEdgeCases:
    """Test SE3_Mat backward with edge cases."""

    def test_identity_transform_gradient(self, device):
        """Test backward with identity transformation."""
        dtype = torch.float64
        X = pp.identity_SE3(5, device=device, dtype=dtype)
        X_warp = to_warp_backend(X)
        X_warp.requires_grad_(True)
        
        result = SE3_Mat.apply(X_warp)
        result.sum().backward()
        
        # Gradients should be finite
        assert not torch.isnan(X_warp.grad).any()
        assert not torch.isinf(X_warp.grad).any()

    def test_grad_dtype_preserved(self, device, dtype_bwd):
        """Test that gradient dtype matches input dtype."""
        X = pp.randn_SE3(5, device=device, dtype=dtype_bwd)
        X_warp = to_warp_backend(X)
        X_warp.requires_grad_(True)
        
        result = SE3_Mat.apply(X_warp)
        result.sum().backward()
        
        assert X_warp.grad.dtype == dtype_bwd

    def test_grad_device_preserved(self, device, dtype_bwd):
        """Test that gradient device matches input device."""
        X = pp.randn_SE3(5, device=device, dtype=dtype_bwd)
        X_warp = to_warp_backend(X)
        X_warp.requires_grad_(True)
        
        result = SE3_Mat.apply(X_warp)
        result.sum().backward()
        
        assert str(X_warp.grad.device).startswith(device)

    def test_large_batch_gradient(self, device, dtype_bwd):
        """Test backward with a large batch size."""
        X = pp.randn_SE3(1000, device=device, dtype=dtype_bwd)
        X_warp = to_warp_backend(X)
        X_warp.requires_grad_(True)
        
        result = SE3_Mat.apply(X_warp)
        result.sum().backward()
        
        assert X_warp.grad.shape == (1000, 7)
        assert not torch.isnan(X_warp.grad).any()

    def test_translation_gradient_correct(self, device):
        """Test that translation gradient is correct (should be all 1s when summing matrix)."""
        dtype = torch.float64
        # Use identity rotation so we can isolate translation gradient
        t = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]], device=device, dtype=dtype, requires_grad=True)
        X = pp.LieTensor(t, ltype=pp.SE3_type)
        X_warp = to_warp_backend(X)
        
        result = SE3_Mat.apply(X_warp)
        
        # Only sum the translation column (column 3, rows 0-2)
        translation_sum = result[0, :3, 3].sum()
        translation_sum.backward()
        
        # Gradient w.r.t. tx, ty, tz should each be 1.0
        expected_grad_t = torch.tensor([1., 1., 1.], device=device, dtype=dtype)
        torch.testing.assert_close(t.grad[:, :3].squeeze(), expected_grad_t, atol=1e-10, rtol=1e-10)


# =============================================================================
# Integration Tests (via warp backend .matrix() method)
# =============================================================================


class TestSE3MatIntegration:
    """Test integration via the .matrix() method on warp backend."""

    def test_matrix_method_works(self, device, dtype):
        """Test that calling .matrix() on warp backend works."""
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        X_warp = to_warp_backend(X)
        
        result = X_warp.matrix()
        expected = X.matrix()
        
        assert result.shape == expected.shape
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype))

    def test_matrix_method_backward(self, device, dtype_bwd):
        """Test that .matrix() backward works correctly."""
        X = pp.randn_SE3(5, device=device, dtype=dtype_bwd)
        X_warp = to_warp_backend(X)
        X_warp.requires_grad_(True)
        
        result = X_warp.matrix()
        result.sum().backward()
        
        assert X_warp.grad.shape == (5, 7)
        assert not torch.isnan(X_warp.grad).any()

