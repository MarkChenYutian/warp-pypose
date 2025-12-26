"""Unit tests for SO3_Mat forward and backward."""
import pytest
import torch
import pypose as pp
from pypose_warp.ltype.SO3_group import SO3_Mat, SO3_Mat_fwd
from conftest import get_tolerances


def get_bwd_tolerances(dtype: torch.dtype) -> dict:
    """
    Get appropriate atol/rtol for backward tests.
    
    Note: The backward pass for SO3_Mat computes the true numerical gradient
    of the quaternion-to-matrix conversion. This differs from PyPose's backward
    which uses a Lie algebra tangent space gradient. Our implementation matches
    the numerical gradient of the forward pass (verified via torch.autograd.gradcheck).
    """
    if dtype == torch.float16:
        return {"atol": 1e-1, "rtol": 1e-1}
    elif dtype == torch.float32:
        return {"atol": 1e-4, "rtol": 1e-4}
    elif dtype == torch.float64:
        return {"atol": 1e-10, "rtol": 1e-10}
    else:
        raise NotImplementedError(f"Unimplemented for {dtype=}")


class TestSO3MatBatchDimensions:
    """Test SO3_Mat_fwd with various batch dimensions."""

    def test_1d_batch(self, device, dtype):
        """Test with 1D batch dimension."""
        so3 = pp.randn_SO3(5, device=device, dtype=dtype)
        
        result = SO3_Mat_fwd(so3)
        expected = so3.matrix()
        
        assert result.shape == expected.shape == (5, 3, 3)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_tolerances(dtype))

    def test_2d_batch(self, device, dtype):
        """Test with 2D batch dimensions."""
        so3 = pp.randn_SO3(3, 4, device=device, dtype=dtype)
        
        result = SO3_Mat_fwd(so3)
        expected = so3.matrix()
        
        assert result.shape == expected.shape == (3, 4, 3, 3)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_tolerances(dtype))

    def test_3d_batch(self, device, dtype):
        """Test with 3D batch dimensions."""
        so3 = pp.randn_SO3(2, 3, 4, device=device, dtype=dtype)
        
        result = SO3_Mat_fwd(so3)
        expected = so3.matrix()
        
        assert result.shape == expected.shape == (2, 3, 4, 3, 3)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_tolerances(dtype))

    def test_4d_batch(self, device, dtype):
        """Test with 4D batch dimensions."""
        so3 = pp.randn_SO3(2, 2, 3, 4, device=device, dtype=dtype)
        
        result = SO3_Mat_fwd(so3)
        expected = so3.matrix()
        
        assert result.shape == expected.shape == (2, 2, 3, 4, 3, 3)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_tolerances(dtype))

    def test_scalar_no_batch(self, device, dtype):
        """Test with no batch dimensions (scalar input)."""
        so3 = pp.randn_SO3(device=device, dtype=dtype)
        
        result = SO3_Mat_fwd(so3)
        expected = so3.matrix()
        
        assert result.shape == expected.shape == (3, 3)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_tolerances(dtype))


class TestSO3MatPrecision:
    """Test SO3_Mat_fwd precision handling."""

    def test_precision(self, device, dtype):
        """Test precision for various dtypes."""
        so3 = pp.randn_SO3(10, device=device, dtype=dtype)
        
        result = SO3_Mat_fwd(so3)
        expected = so3.matrix()
        
        torch.testing.assert_close(result, expected, **get_tolerances(dtype))

    def test_output_dtype_preserved(self, device, dtype):
        """Test that output dtype matches input dtype."""
        so3 = pp.randn_SO3(5, device=device, dtype=dtype)
        
        result = SO3_Mat_fwd(so3)
        assert result.dtype == dtype


class TestSO3MatEdgeCases:
    """Test SO3_Mat_fwd with edge cases."""

    def test_identity_rotation(self, device, dtype):
        """Test with identity rotation."""
        so3 = pp.identity_SO3(5, device=device, dtype=dtype)
        
        result = SO3_Mat_fwd(so3)
        expected = torch.eye(3, device=device, dtype=dtype).expand(5, 3, 3)
        
        torch.testing.assert_close(result, expected, **get_tolerances(dtype))

    def test_180_degree_rotation_x(self, device):
        """Test 180 degree rotation around x-axis."""
        dtype = torch.float64
        # Quaternion for 180 degree rotation around x: (1, 0, 0, 0)
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype)
        so3 = pp.LieTensor(q, ltype=pp.SO3_type)
        
        result = SO3_Mat_fwd(so3)
        expected = torch.tensor([[[1., 0., 0.],
                                   [0., -1., 0.],
                                   [0., 0., -1.]]], device=device, dtype=dtype)
        
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_180_degree_rotation_y(self, device):
        """Test 180 degree rotation around y-axis."""
        dtype = torch.float64
        # Quaternion for 180 degree rotation around y: (0, 1, 0, 0)
        q = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device, dtype=dtype)
        so3 = pp.LieTensor(q, ltype=pp.SO3_type)
        
        result = SO3_Mat_fwd(so3)
        expected = torch.tensor([[[-1., 0., 0.],
                                   [0., 1., 0.],
                                   [0., 0., -1.]]], device=device, dtype=dtype)
        
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_180_degree_rotation_z(self, device):
        """Test 180 degree rotation around z-axis."""
        dtype = torch.float64
        # Quaternion for 180 degree rotation around z: (0, 0, 1, 0)
        q = torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=device, dtype=dtype)
        so3 = pp.LieTensor(q, ltype=pp.SO3_type)
        
        result = SO3_Mat_fwd(so3)
        expected = torch.tensor([[[-1., 0., 0.],
                                   [0., -1., 0.],
                                   [0., 0., 1.]]], device=device, dtype=dtype)
        
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_rotation_matrix_is_orthogonal(self, device):
        """Test that output rotation matrices are orthogonal (R^T R = I)."""
        dtype = torch.float64
        so3 = pp.randn_SO3(10, device=device, dtype=dtype)
        
        result = SO3_Mat_fwd(so3)
        
        # R^T @ R should be identity
        RtR = torch.bmm(result.transpose(-2, -1), result)
        I = torch.eye(3, device=device, dtype=dtype).expand(10, 3, 3)
        
        torch.testing.assert_close(RtR, I, atol=1e-10, rtol=1e-10)

    def test_rotation_matrix_determinant_is_one(self, device):
        """Test that output rotation matrices have determinant 1."""
        dtype = torch.float64
        so3 = pp.randn_SO3(10, device=device, dtype=dtype)
        
        result = SO3_Mat_fwd(so3)
        det = torch.linalg.det(result)
        
        torch.testing.assert_close(det, torch.ones(10, device=device, dtype=dtype), atol=1e-10, rtol=1e-10)

    def test_single_element_batch(self, device, dtype):
        """Test with a batch of size 1."""
        so3 = pp.randn_SO3(1, device=device, dtype=dtype)
        
        result = SO3_Mat_fwd(so3)
        expected = so3.matrix()
        
        assert result.shape == expected.shape == (1, 3, 3)
        torch.testing.assert_close(result, expected, **get_tolerances(dtype))

    def test_large_batch(self, device, dtype):
        """Test with a large batch size."""
        so3 = pp.randn_SO3(1000, device=device, dtype=dtype)
        
        result = SO3_Mat_fwd(so3)
        expected = so3.matrix()
        
        assert result.shape == expected.shape == (1000, 3, 3)
        torch.testing.assert_close(result, expected, **get_tolerances(dtype))

    def test_output_device(self, device, dtype):
        """Test that output device matches input device."""
        so3 = pp.randn_SO3(5, device=device, dtype=dtype)
        
        result = SO3_Mat_fwd(so3)
        assert str(result.device).startswith(device)


class TestSO3MatErrors:
    """Test SO3_Mat_fwd error handling."""

    def test_5d_batch_raises(self, device, dtype):
        """Test that 5D batch dimensions raise NotImplementedError."""
        so3 = pp.randn_SO3(2, 2, 2, 2, 2, device=device, dtype=dtype)
        
        with pytest.raises(NotImplementedError, match="Batch dimensions > 4"):
            SO3_Mat_fwd(so3)


# =============================================================================
# Backward Pass Tests
# =============================================================================


class TestSO3MatBwdGradcheck:
    """Test SO3_Mat backward with torch.autograd.gradcheck."""

    def test_gradcheck_1d(self, device):
        """Test gradcheck with 1D batch."""
        dtype = torch.float64
        so3 = pp.randn_SO3(3, device=device, dtype=dtype, requires_grad=True)
        
        def func(x):
            return SO3_Mat.apply(pp.LieTensor(x, ltype=pp.SO3_type))
        
        torch.autograd.gradcheck(func, (so3.tensor(),), atol=1e-6, rtol=1e-6)

    def test_gradcheck_2d(self, device):
        """Test gradcheck with 2D batch."""
        dtype = torch.float64
        so3 = pp.randn_SO3(2, 3, device=device, dtype=dtype, requires_grad=True)
        
        def func(x):
            return SO3_Mat.apply(pp.LieTensor(x, ltype=pp.SO3_type))
        
        torch.autograd.gradcheck(func, (so3.tensor(),), atol=1e-6, rtol=1e-6)

    def test_gradcheck_scalar(self, device):
        """Test gradcheck with scalar input."""
        dtype = torch.float64
        so3 = pp.randn_SO3(device=device, dtype=dtype, requires_grad=True)
        
        def func(x):
            return SO3_Mat.apply(pp.LieTensor(x, ltype=pp.SO3_type))
        
        torch.autograd.gradcheck(func, (so3.tensor(),), atol=1e-6, rtol=1e-6)


class TestSO3MatBwdEdgeCases:
    """Test SO3_Mat backward with edge cases."""

    def test_identity_rotation_gradient(self, device):
        """Test backward with identity rotation."""
        dtype = torch.float64
        so3 = pp.identity_SO3(5, device=device, dtype=dtype)
        so3.requires_grad_(True)
        
        result = SO3_Mat.apply(so3)
        result.sum().backward()
        
        # For identity quaternion (0,0,0,1), the gradients should be finite
        assert not torch.isnan(so3.grad).any()
        assert not torch.isinf(so3.grad).any()

    def test_grad_dtype_preserved(self, device, dtype):
        """Test that gradient dtype matches input dtype."""
        if dtype == torch.float16:
            pytest.skip("float16 backward not reliably supported")
        so3 = pp.randn_SO3(5, device=device, dtype=dtype)
        so3.requires_grad_(True)
        
        result = SO3_Mat.apply(so3)
        result.sum().backward()
        
        assert so3.grad.dtype == dtype

    def test_grad_device_preserved(self, device, dtype):
        """Test that gradient device matches input device."""
        if dtype == torch.float16:
            pytest.skip("float16 backward not reliably supported")
        so3 = pp.randn_SO3(5, device=device, dtype=dtype)
        so3.requires_grad_(True)
        
        result = SO3_Mat.apply(so3)
        result.sum().backward()
        
        assert str(so3.grad.device).startswith(device)

    def test_large_batch_gradient(self, device, dtype):
        """Test backward with a large batch size."""
        if dtype == torch.float16:
            pytest.skip("float16 backward not reliably supported")
        so3 = pp.randn_SO3(1000, device=device, dtype=dtype)
        so3.requires_grad_(True)
        
        result = SO3_Mat.apply(so3)
        result.sum().backward()
        
        assert so3.grad.shape == (1000, 4)
        assert not torch.isnan(so3.grad).any()
