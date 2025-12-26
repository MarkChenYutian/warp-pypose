"""Unit tests for SO3_Jinvp forward and backward."""
import pytest
import torch
import pypose as pp
from pypose_warp import to_pypose_backend
from pypose.lietensor.operation import so3_Jl_inv, SO3_Log
from pypose_warp.ltype.SO3_group import SO3_Jinvp, SO3_Jinvp_fwd
from conftest import get_fwd_tolerances, get_bwd_tolerances, Operator


def pypose_jinvp(X, p):
    """Reference implementation using PyPose operations."""
    X_tensor = X.tensor() if isinstance(X, pp.LieTensor) else X
    p_tensor = p.tensor() if isinstance(p, pp.LieTensor) else p
    return (so3_Jl_inv(SO3_Log.apply(X_tensor)) @ p_tensor.unsqueeze(-1)).squeeze(-1)


# =============================================================================
# Forward Pass Tests
# =============================================================================


class TestSO3_Jinvp_Fwd_BatchDimensions:
    """Test SO3_Jinvp_fwd with various batch dimensions."""

    def test_1d_batch(self, device, dtype):
        """Test with 1D batch dimension."""
        so3 = pp.randn_SO3(5, device=device, dtype=dtype)
        p = torch.randn(5, 3, device=device, dtype=dtype)

        result = SO3_Jinvp_fwd(so3, p)
        expected = pypose_jinvp(so3, p)

        assert result.shape == expected.shape == (5, 3)
        assert result.dtype == dtype
        torch.testing.assert_close(result.tensor(), expected, **get_fwd_tolerances(dtype, Operator.SO3_Jinvp))

    def test_2d_batch(self, device, dtype):
        """Test with 2D batch dimensions."""
        so3 = pp.randn_SO3(3, 4, device=device, dtype=dtype)
        p = torch.randn(3, 4, 3, device=device, dtype=dtype)

        result = SO3_Jinvp_fwd(so3, p)
        expected = pypose_jinvp(so3, p)

        assert result.shape == expected.shape == (3, 4, 3)
        assert result.dtype == dtype
        torch.testing.assert_close(result.tensor(), expected, **get_fwd_tolerances(dtype, Operator.SO3_Jinvp))

    def test_3d_batch(self, device, dtype):
        """Test with 3D batch dimensions."""
        so3 = pp.randn_SO3(2, 3, 4, device=device, dtype=dtype)
        p = torch.randn(2, 3, 4, 3, device=device, dtype=dtype)

        result = SO3_Jinvp_fwd(so3, p)
        expected = pypose_jinvp(so3, p)

        assert result.shape == expected.shape == (2, 3, 4, 3)
        assert result.dtype == dtype
        torch.testing.assert_close(result.tensor(), expected, **get_fwd_tolerances(dtype, Operator.SO3_Jinvp))

    def test_4d_batch(self, device, dtype):
        """Test with 4D batch dimensions."""
        so3 = pp.randn_SO3(2, 3, 4, 5, device=device, dtype=dtype)
        p = torch.randn(2, 3, 4, 5, 3, device=device, dtype=dtype)

        result = SO3_Jinvp_fwd(so3, p)
        expected = pypose_jinvp(so3, p)

        assert result.shape == expected.shape == (2, 3, 4, 5, 3)
        assert result.dtype == dtype
        torch.testing.assert_close(result.tensor(), expected, **get_fwd_tolerances(dtype, Operator.SO3_Jinvp))

    def test_scalar_no_batch(self, device, dtype):
        """Test with no batch dimensions (single rotation and tangent vector)."""
        so3 = pp.randn_SO3(device=device, dtype=dtype)
        p = torch.randn(3, device=device, dtype=dtype)

        result = SO3_Jinvp_fwd(so3, p)
        expected = pypose_jinvp(so3, p)

        assert result.shape == expected.shape == (3,)
        assert result.dtype == dtype
        torch.testing.assert_close(result.tensor(), expected, **get_fwd_tolerances(dtype, Operator.SO3_Jinvp))


class TestSO3_Jinvp_Fwd_Broadcasting:
    """Test SO3_Jinvp_fwd broadcasting behavior."""

    def test_broadcast_so3_singleton(self, device):
        """Test broadcasting single SO3 to multiple tangent vectors."""
        dtype = torch.float32
        so3 = pp.randn_SO3(1, device=device, dtype=dtype)
        p = torch.randn(5, 3, device=device, dtype=dtype)

        result = SO3_Jinvp_fwd(so3, p)
        expected = pypose_jinvp(so3, p)

        assert result.shape == expected.shape == (5, 3)
        torch.testing.assert_close(result.tensor(), expected, **get_fwd_tolerances(dtype, Operator.SO3_Jinvp))

    def test_broadcast_p_singleton(self, device):
        """Test broadcasting single tangent vector to multiple SO3."""
        dtype = torch.float32
        so3 = pp.randn_SO3(5, device=device, dtype=dtype)
        p = torch.randn(1, 3, device=device, dtype=dtype)

        result = SO3_Jinvp_fwd(so3, p)
        expected = pypose_jinvp(so3, p)

        assert result.shape == expected.shape == (5, 3)
        torch.testing.assert_close(result.tensor(), expected, **get_fwd_tolerances(dtype, Operator.SO3_Jinvp))

    def test_broadcast_2d_cross(self, device):
        """Test 2D cross-broadcasting: (1, 5) SO3 with (4, 1) tangent vectors."""
        dtype = torch.float32
        so3 = pp.randn_SO3(1, 5, device=device, dtype=dtype)
        p = torch.randn(4, 1, 3, device=device, dtype=dtype)

        result = SO3_Jinvp_fwd(so3, p)
        expected = pypose_jinvp(so3, p)

        assert result.shape == expected.shape == (4, 5, 3)
        torch.testing.assert_close(result.tensor(), expected, **get_fwd_tolerances(dtype, Operator.SO3_Jinvp))

    def test_broadcast_3d_complex(self, device):
        """Test complex 3D broadcasting: (3, 1, 5) SO3 with (1, 4, 1) tangent vectors."""
        dtype = torch.float32
        so3 = pp.randn_SO3(3, 1, 5, device=device, dtype=dtype)
        p = torch.randn(1, 4, 1, 3, device=device, dtype=dtype)

        result = SO3_Jinvp_fwd(so3, p)
        expected = pypose_jinvp(so3, p)

        assert result.shape == expected.shape == (3, 4, 5, 3)
        torch.testing.assert_close(result.tensor(), expected, **get_fwd_tolerances(dtype, Operator.SO3_Jinvp))

    def test_broadcast_4d_complex(self, device):
        """Test complex 4D broadcasting."""
        dtype = torch.float32
        so3 = pp.randn_SO3(2, 1, 3, 1, device=device, dtype=dtype)
        p = torch.randn(1, 4, 1, 5, 3, device=device, dtype=dtype)

        result = SO3_Jinvp_fwd(so3, p)
        expected = pypose_jinvp(so3, p)

        assert result.shape == expected.shape == (2, 4, 3, 5, 3)
        torch.testing.assert_close(result.tensor(), expected, **get_fwd_tolerances(dtype, Operator.SO3_Jinvp))

    def test_broadcast_different_ndim(self, device):
        """Test broadcasting tensors with different number of dimensions."""
        dtype = torch.float32
        so3 = pp.randn_SO3(5, device=device, dtype=dtype)  # shape (5, 4)
        p = torch.randn(3, 1, 3, device=device, dtype=dtype)  # shape (3, 1, 3)

        result = SO3_Jinvp_fwd(so3, p)
        expected = pypose_jinvp(so3, p)

        assert result.shape == expected.shape == (3, 5, 3)
        torch.testing.assert_close(result.tensor(), expected, **get_fwd_tolerances(dtype, Operator.SO3_Jinvp))


class TestSO3_Jinvp_Fwd_Precision:
    """Test SO3_Jinvp_fwd precision handling."""

    def test_fp32_precision(self, device):
        """Test float32 precision and accuracy."""
        dtype = torch.float32
        so3 = pp.randn_SO3(10, device=device, dtype=dtype)
        p = torch.randn(10, 3, device=device, dtype=dtype)

        result = SO3_Jinvp_fwd(so3, p)
        expected = pypose_jinvp(so3, p)

        torch.testing.assert_close(result.tensor(), expected, **get_fwd_tolerances(dtype, Operator.SO3_Jinvp))

    def test_fp64_precision(self, device):
        """Test float64 precision and accuracy."""
        dtype = torch.float64
        so3 = pp.randn_SO3(10, device=device, dtype=dtype)
        p = torch.randn(10, 3, device=device, dtype=dtype)

        result = SO3_Jinvp_fwd(so3, p)
        expected = pypose_jinvp(so3, p)

        torch.testing.assert_close(result.tensor(), expected, **get_fwd_tolerances(dtype, Operator.SO3_Jinvp))

    def test_fp16_precision(self, device):
        """Test float16 precision and accuracy."""
        dtype = torch.float16
        so3 = pp.randn_SO3(10, device=device, dtype=dtype)
        p = torch.randn(10, 3, device=device, dtype=dtype)

        result = SO3_Jinvp_fwd(so3, p)
        expected = pypose_jinvp(so3, p)

        torch.testing.assert_close(result.tensor(), expected, **get_fwd_tolerances(dtype, Operator.SO3_Jinvp))

    def test_output_dtype_preserved(self, device, dtype):
        """Test that output dtype matches input dtype."""
        so3 = pp.randn_SO3(5, device=device, dtype=dtype)
        p = torch.randn(5, 3, device=device, dtype=dtype)

        result = SO3_Jinvp_fwd(so3, p)

        assert result.dtype == dtype


class TestSO3_Jinvp_Fwd_EdgeCases:
    """Test SO3_Jinvp_fwd edge cases."""

    def test_identity_rotation(self, device):
        """Test with identity rotation: Jl_inv(Log(I)) = I, so result = p."""
        dtype = torch.float32
        so3 = pp.identity_SO3(5, device=device, dtype=dtype)
        p = torch.randn(5, 3, device=device, dtype=dtype)

        result = SO3_Jinvp_fwd(so3, p)

        # For identity, Jl_inv(0) = I, so Jinvp(I, p) = p
        torch.testing.assert_close(result.tensor(), p, **get_fwd_tolerances(dtype, Operator.SO3_Jinvp))

    def test_zero_tangent_vector(self, device):
        """Test with zero tangent vector: result should be zero."""
        dtype = torch.float32
        so3 = pp.randn_SO3(5, device=device, dtype=dtype)
        p = torch.zeros(5, 3, device=device, dtype=dtype)

        result = SO3_Jinvp_fwd(so3, p)

        torch.testing.assert_close(result.tensor(), torch.zeros_like(p), **get_fwd_tolerances(dtype, Operator.SO3_Jinvp))

    def test_single_element_batch(self, device):
        """Test with batch size of 1."""
        dtype = torch.float32
        so3 = pp.randn_SO3(1, device=device, dtype=dtype)
        p = torch.randn(1, 3, device=device, dtype=dtype)

        result = SO3_Jinvp_fwd(so3, p)
        expected = pypose_jinvp(so3, p)

        assert result.shape == (1, 3)
        torch.testing.assert_close(result.tensor(), expected, **get_fwd_tolerances(dtype, Operator.SO3_Jinvp))

    def test_large_batch(self, device):
        """Test with large batch size."""
        dtype = torch.float32
        so3 = pp.randn_SO3(1000, device=device, dtype=dtype)
        p = torch.randn(1000, 3, device=device, dtype=dtype)

        result = SO3_Jinvp_fwd(so3, p)
        expected = pypose_jinvp(so3, p)

        assert result.shape == (1000, 3)
        torch.testing.assert_close(result.tensor(), expected, **get_fwd_tolerances(dtype, Operator.SO3_Jinvp))

    def test_output_device(self, device):
        """Test that output is on the same device as input."""
        dtype = torch.float32
        so3 = pp.randn_SO3(5, device=device, dtype=dtype)
        p = torch.randn(5, 3, device=device, dtype=dtype)

        result = SO3_Jinvp_fwd(so3, p)

        assert result.device == p.device == so3.device

    def test_output_is_lietensor(self, device):
        """Test that output is a LieTensor with so3_type."""
        dtype = torch.float32
        so3 = pp.randn_SO3(5, device=device, dtype=dtype)
        p = torch.randn(5, 3, device=device, dtype=dtype)

        result = SO3_Jinvp_fwd(so3, p)

        assert isinstance(result, pp.LieTensor)
        assert to_pypose_backend(result).ltype == pp.so3_type


class TestSO3_Jinvp_Fwd_Errors:
    """Test SO3_Jinvp_fwd error handling."""

    def test_incompatible_shapes_raises(self, device):
        """Test that incompatible shapes raise ValueError."""
        dtype = torch.float32
        so3 = pp.randn_SO3(3, device=device, dtype=dtype)
        p = torch.randn(5, 3, device=device, dtype=dtype)  # Incompatible

        with pytest.raises(ValueError, match="not broadcastable"):
            SO3_Jinvp_fwd(so3, p)

    def test_5d_batch_raises(self, device):
        """Test that 5D batch dimensions raise NotImplementedError."""
        dtype = torch.float32
        so3 = pp.randn_SO3(2, 2, 2, 2, 2, device=device, dtype=dtype)
        p = torch.randn(2, 2, 2, 2, 2, 3, device=device, dtype=dtype)

        with pytest.raises(NotImplementedError, match="Batch dimensions > 4"):
            SO3_Jinvp_fwd(so3, p)


# =============================================================================
# Backward Pass Tests (use dtype_bwd fixture - no fp16)
# =============================================================================


class TestSO3_Jinvp_Bwd_BatchDimensions:
    """Test SO3_Jinvp backward with various batch dimensions."""

    def test_1d_batch(self, device, dtype_bwd):
        """Test backward with 1D batch dimension."""
        so3_data = pp.randn_SO3(5, device=device, dtype=dtype_bwd)
        p_data = torch.randn(5, 3, device=device, dtype=dtype_bwd)

        # Our implementation
        so3_ours = so3_data.clone().requires_grad_(True)
        p_ours = p_data.clone().requires_grad_(True)
        result_ours = SO3_Jinvp.apply(so3_ours, p_ours)
        result_ours.tensor().sum().backward()

        # PyPose reference
        so3_ref = so3_data.clone().requires_grad_(True)
        p_ref = p_data.clone().requires_grad_(True)
        result_ref = pypose_jinvp(so3_ref, p_ref)
        result_ref.sum().backward()

        assert so3_ours.grad.shape == so3_ref.grad.shape == (5, 4)
        assert p_ours.grad.shape == p_ref.grad.shape == (5, 3)
        torch.testing.assert_close(so3_ours.grad, so3_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_Jinvp))
        torch.testing.assert_close(p_ours.grad, p_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_Jinvp))

    def test_2d_batch(self, device, dtype_bwd):
        """Test backward with 2D batch dimensions."""
        so3_data = pp.randn_SO3(3, 4, device=device, dtype=dtype_bwd)
        p_data = torch.randn(3, 4, 3, device=device, dtype=dtype_bwd)

        so3_ours = so3_data.clone().requires_grad_(True)
        p_ours = p_data.clone().requires_grad_(True)
        result_ours = SO3_Jinvp.apply(so3_ours, p_ours)
        result_ours.tensor().sum().backward()

        so3_ref = so3_data.clone().requires_grad_(True)
        p_ref = p_data.clone().requires_grad_(True)
        result_ref = pypose_jinvp(so3_ref, p_ref)
        result_ref.sum().backward()

        assert so3_ours.grad.shape == so3_ref.grad.shape == (3, 4, 4)
        assert p_ours.grad.shape == p_ref.grad.shape == (3, 4, 3)
        torch.testing.assert_close(so3_ours.grad, so3_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_Jinvp))
        torch.testing.assert_close(p_ours.grad, p_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_Jinvp))

    def test_3d_batch(self, device, dtype_bwd):
        """Test backward with 3D batch dimensions."""
        so3_data = pp.randn_SO3(2, 3, 4, device=device, dtype=dtype_bwd)
        p_data = torch.randn(2, 3, 4, 3, device=device, dtype=dtype_bwd)

        so3_ours = so3_data.clone().requires_grad_(True)
        p_ours = p_data.clone().requires_grad_(True)
        result_ours = SO3_Jinvp.apply(so3_ours, p_ours)
        result_ours.tensor().sum().backward()

        so3_ref = so3_data.clone().requires_grad_(True)
        p_ref = p_data.clone().requires_grad_(True)
        result_ref = pypose_jinvp(so3_ref, p_ref)
        result_ref.sum().backward()

        assert so3_ours.grad.shape == so3_ref.grad.shape == (2, 3, 4, 4)
        assert p_ours.grad.shape == p_ref.grad.shape == (2, 3, 4, 3)
        torch.testing.assert_close(so3_ours.grad, so3_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_Jinvp))
        torch.testing.assert_close(p_ours.grad, p_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_Jinvp))

    def test_4d_batch(self, device, dtype_bwd):
        """Test backward with 4D batch dimensions."""
        so3_data = pp.randn_SO3(2, 2, 3, 4, device=device, dtype=dtype_bwd)
        p_data = torch.randn(2, 2, 3, 4, 3, device=device, dtype=dtype_bwd)

        so3_ours = so3_data.clone().requires_grad_(True)
        p_ours = p_data.clone().requires_grad_(True)
        result_ours = SO3_Jinvp.apply(so3_ours, p_ours)
        result_ours.tensor().sum().backward()

        so3_ref = so3_data.clone().requires_grad_(True)
        p_ref = p_data.clone().requires_grad_(True)
        result_ref = pypose_jinvp(so3_ref, p_ref)
        result_ref.sum().backward()

        assert so3_ours.grad.shape == so3_ref.grad.shape == (2, 2, 3, 4, 4)
        assert p_ours.grad.shape == p_ref.grad.shape == (2, 2, 3, 4, 3)
        torch.testing.assert_close(so3_ours.grad, so3_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_Jinvp))
        torch.testing.assert_close(p_ours.grad, p_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_Jinvp))

    def test_scalar_no_batch(self, device, dtype_bwd):
        """Test backward with no batch dimensions."""
        so3_data = pp.randn_SO3(device=device, dtype=dtype_bwd)
        p_data = torch.randn(3, device=device, dtype=dtype_bwd)

        so3_ours = so3_data.clone().requires_grad_(True)
        p_ours = p_data.clone().requires_grad_(True)
        result_ours = SO3_Jinvp.apply(so3_ours, p_ours)
        result_ours.tensor().sum().backward()

        so3_ref = so3_data.clone().requires_grad_(True)
        p_ref = p_data.clone().requires_grad_(True)
        result_ref = pypose_jinvp(so3_ref, p_ref)
        result_ref.sum().backward()

        assert so3_ours.grad.shape == so3_ref.grad.shape == (4,)
        assert p_ours.grad.shape == p_ref.grad.shape == (3,)
        torch.testing.assert_close(so3_ours.grad, so3_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_Jinvp))
        torch.testing.assert_close(p_ours.grad, p_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_Jinvp))


class TestSO3_Jinvp_Bwd_Broadcasting:
    """Test SO3_Jinvp backward with broadcasting."""

    def test_broadcast_so3_singleton(self, device, dtype_bwd):
        """Test backward with single SO3 broadcast to multiple tangent vectors."""
        so3_data = pp.randn_SO3(1, device=device, dtype=dtype_bwd)
        p_data = torch.randn(5, 3, device=device, dtype=dtype_bwd)

        so3_ours = so3_data.clone().requires_grad_(True)
        p_ours = p_data.clone().requires_grad_(True)
        result_ours = SO3_Jinvp.apply(so3_ours, p_ours)
        result_ours.tensor().sum().backward()

        so3_ref = so3_data.clone().requires_grad_(True)
        p_ref = p_data.clone().requires_grad_(True)
        result_ref = pypose_jinvp(so3_ref, p_ref)
        result_ref.sum().backward()

        assert so3_ours.grad.shape == so3_ref.grad.shape == (1, 4)
        assert p_ours.grad.shape == p_ref.grad.shape == (5, 3)
        torch.testing.assert_close(so3_ours.grad, so3_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_Jinvp))
        torch.testing.assert_close(p_ours.grad, p_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_Jinvp))

    def test_broadcast_p_singleton(self, device, dtype_bwd):
        """Test backward with single tangent vector broadcast to multiple SO3."""
        so3_data = pp.randn_SO3(5, device=device, dtype=dtype_bwd)
        p_data = torch.randn(1, 3, device=device, dtype=dtype_bwd)

        so3_ours = so3_data.clone().requires_grad_(True)
        p_ours = p_data.clone().requires_grad_(True)
        result_ours = SO3_Jinvp.apply(so3_ours, p_ours)
        result_ours.tensor().sum().backward()

        so3_ref = so3_data.clone().requires_grad_(True)
        p_ref = p_data.clone().requires_grad_(True)
        result_ref = pypose_jinvp(so3_ref, p_ref)
        result_ref.sum().backward()

        assert so3_ours.grad.shape == so3_ref.grad.shape == (5, 4)
        assert p_ours.grad.shape == p_ref.grad.shape == (1, 3)
        torch.testing.assert_close(so3_ours.grad, so3_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_Jinvp))
        torch.testing.assert_close(p_ours.grad, p_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_Jinvp))

    def test_broadcast_2d_cross(self, device, dtype_bwd):
        """Test backward with 2D cross-broadcasting."""
        so3_data = pp.randn_SO3(1, 5, device=device, dtype=dtype_bwd)
        p_data = torch.randn(4, 1, 3, device=device, dtype=dtype_bwd)

        so3_ours = so3_data.clone().requires_grad_(True)
        p_ours = p_data.clone().requires_grad_(True)
        result_ours = SO3_Jinvp.apply(so3_ours, p_ours)
        result_ours.tensor().sum().backward()

        so3_ref = so3_data.clone().requires_grad_(True)
        p_ref = p_data.clone().requires_grad_(True)
        result_ref = pypose_jinvp(so3_ref, p_ref)
        result_ref.sum().backward()

        assert so3_ours.grad.shape == so3_ref.grad.shape == (1, 5, 4)
        assert p_ours.grad.shape == p_ref.grad.shape == (4, 1, 3)
        torch.testing.assert_close(so3_ours.grad, so3_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_Jinvp))
        torch.testing.assert_close(p_ours.grad, p_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_Jinvp))

    def test_broadcast_3d_complex(self, device, dtype_bwd):
        """Test backward with complex 3D broadcasting."""
        so3_data = pp.randn_SO3(3, 1, 5, device=device, dtype=dtype_bwd)
        p_data = torch.randn(1, 4, 1, 3, device=device, dtype=dtype_bwd)

        so3_ours = so3_data.clone().requires_grad_(True)
        p_ours = p_data.clone().requires_grad_(True)
        result_ours = SO3_Jinvp.apply(so3_ours, p_ours)
        result_ours.tensor().sum().backward()

        so3_ref = so3_data.clone().requires_grad_(True)
        p_ref = p_data.clone().requires_grad_(True)
        result_ref = pypose_jinvp(so3_ref, p_ref)
        result_ref.sum().backward()

        assert so3_ours.grad.shape == so3_ref.grad.shape == (3, 1, 5, 4)
        assert p_ours.grad.shape == p_ref.grad.shape == (1, 4, 1, 3)
        torch.testing.assert_close(so3_ours.grad, so3_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_Jinvp))
        torch.testing.assert_close(p_ours.grad, p_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_Jinvp))

    def test_broadcast_different_ndim(self, device, dtype_bwd):
        """Test backward with different number of dimensions."""
        so3_data = pp.randn_SO3(5, device=device, dtype=dtype_bwd)  # shape (5, 4)
        p_data = torch.randn(3, 1, 3, device=device, dtype=dtype_bwd)  # shape (3, 1, 3)

        so3_ours = so3_data.clone().requires_grad_(True)
        p_ours = p_data.clone().requires_grad_(True)
        result_ours = SO3_Jinvp.apply(so3_ours, p_ours)
        result_ours.tensor().sum().backward()

        so3_ref = so3_data.clone().requires_grad_(True)
        p_ref = p_data.clone().requires_grad_(True)
        result_ref = pypose_jinvp(so3_ref, p_ref)
        result_ref.sum().backward()

        assert so3_ours.grad.shape == so3_ref.grad.shape == (5, 4)
        assert p_ours.grad.shape == p_ref.grad.shape == (3, 1, 3)
        torch.testing.assert_close(so3_ours.grad, so3_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_Jinvp))
        torch.testing.assert_close(p_ours.grad, p_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_Jinvp))


class TestSO3_Jinvp_Bwd_Precision:
    """Test SO3_Jinvp backward precision handling."""

    def test_fp32_precision(self, device):
        """Test backward float32 precision."""
        dtype = torch.float32
        so3_data = pp.randn_SO3(10, device=device, dtype=dtype)
        p_data = torch.randn(10, 3, device=device, dtype=dtype)

        so3_ours = so3_data.clone().requires_grad_(True)
        p_ours = p_data.clone().requires_grad_(True)
        result_ours = SO3_Jinvp.apply(so3_ours, p_ours)
        result_ours.tensor().sum().backward()

        so3_ref = so3_data.clone().requires_grad_(True)
        p_ref = p_data.clone().requires_grad_(True)
        result_ref = pypose_jinvp(so3_ref, p_ref)
        result_ref.sum().backward()

        torch.testing.assert_close(so3_ours.grad, so3_ref.grad, **get_bwd_tolerances(dtype, Operator.SO3_Jinvp))
        torch.testing.assert_close(p_ours.grad, p_ref.grad, **get_bwd_tolerances(dtype, Operator.SO3_Jinvp))

    def test_fp64_precision(self, device):
        """Test backward float64 precision."""
        dtype = torch.float64
        so3_data = pp.randn_SO3(10, device=device, dtype=dtype)
        p_data = torch.randn(10, 3, device=device, dtype=dtype)

        so3_ours = so3_data.clone().requires_grad_(True)
        p_ours = p_data.clone().requires_grad_(True)
        result_ours = SO3_Jinvp.apply(so3_ours, p_ours)
        result_ours.tensor().sum().backward()

        so3_ref = so3_data.clone().requires_grad_(True)
        p_ref = p_data.clone().requires_grad_(True)
        result_ref = pypose_jinvp(so3_ref, p_ref)
        result_ref.sum().backward()

        torch.testing.assert_close(so3_ours.grad, so3_ref.grad, **get_bwd_tolerances(dtype, Operator.SO3_Jinvp))
        torch.testing.assert_close(p_ours.grad, p_ref.grad, **get_bwd_tolerances(dtype, Operator.SO3_Jinvp))

    def test_grad_dtype_preserved(self, device, dtype_bwd):
        """Test that gradient dtype matches input dtype."""
        so3 = pp.randn_SO3(5, device=device, dtype=dtype_bwd)
        so3.requires_grad_(True)
        p = torch.randn(5, 3, device=device, dtype=dtype_bwd, requires_grad=True)

        result = SO3_Jinvp.apply(so3, p)
        result.tensor().sum().backward()

        assert so3.grad.dtype == dtype_bwd
        assert p.grad.dtype == dtype_bwd


class TestSO3_Jinvp_Bwd_EdgeCases:
    """Test SO3_Jinvp backward edge cases."""

    def test_quaternion_w_component_zero(self, device, dtype_bwd):
        """Test that quaternion gradient w component is always zero."""
        so3 = pp.randn_SO3(5, device=device, dtype=dtype_bwd)
        so3.requires_grad_(True)
        p = torch.randn(5, 3, device=device, dtype=dtype_bwd, requires_grad=True)

        result = SO3_Jinvp.apply(so3, p)
        result.tensor().sum().backward()

        # The w component of quaternion gradient should always be 0
        assert torch.allclose(so3.grad[..., 3], torch.zeros_like(so3.grad[..., 3]))

    def test_identity_rotation_gradient(self, device):
        """Test gradient through identity rotation."""
        dtype = torch.float32
        so3_data = pp.identity_SO3(5, device=device, dtype=dtype)
        p_data = torch.randn(5, 3, device=device, dtype=dtype)

        so3_ours = so3_data.clone().requires_grad_(True)
        p_ours = p_data.clone().requires_grad_(True)
        result_ours = SO3_Jinvp.apply(so3_ours, p_ours)
        result_ours.tensor().sum().backward()

        so3_ref = so3_data.clone().requires_grad_(True)
        p_ref = p_data.clone().requires_grad_(True)
        result_ref = pypose_jinvp(so3_ref, p_ref)
        result_ref.sum().backward()

        # For identity, p gradient should be I^T @ grad_out = grad_out = all 1s for sum loss
        expected_p_grad = torch.ones_like(p_data)
        torch.testing.assert_close(p_ours.grad, expected_p_grad, **get_bwd_tolerances(dtype, Operator.SO3_Jinvp))
        torch.testing.assert_close(so3_ours.grad, so3_ref.grad, **get_bwd_tolerances(dtype, Operator.SO3_Jinvp))

    def test_zero_tangent_vector_gradient(self, device):
        """Test gradient with zero tangent vector."""
        dtype = torch.float32
        so3_data = pp.randn_SO3(5, device=device, dtype=dtype)
        p_data = torch.zeros(5, 3, device=device, dtype=dtype)

        so3_ours = so3_data.clone().requires_grad_(True)
        p_ours = p_data.clone().requires_grad_(True)
        result_ours = SO3_Jinvp.apply(so3_ours, p_ours)
        result_ours.tensor().sum().backward()

        so3_ref = so3_data.clone().requires_grad_(True)
        p_ref = p_data.clone().requires_grad_(True)
        result_ref = pypose_jinvp(so3_ref, p_ref)
        result_ref.sum().backward()

        torch.testing.assert_close(so3_ours.grad, so3_ref.grad, **get_bwd_tolerances(dtype, Operator.SO3_Jinvp))
        torch.testing.assert_close(p_ours.grad, p_ref.grad, **get_bwd_tolerances(dtype, Operator.SO3_Jinvp))

    def test_large_batch_gradient(self, device):
        """Test gradient with large batch."""
        dtype = torch.float32
        so3_data = pp.randn_SO3(1000, device=device, dtype=dtype)
        p_data = torch.randn(1000, 3, device=device, dtype=dtype)

        so3_ours = so3_data.clone().requires_grad_(True)
        p_ours = p_data.clone().requires_grad_(True)
        result_ours = SO3_Jinvp.apply(so3_ours, p_ours)
        result_ours.tensor().sum().backward()

        so3_ref = so3_data.clone().requires_grad_(True)
        p_ref = p_data.clone().requires_grad_(True)
        result_ref = pypose_jinvp(so3_ref, p_ref)
        result_ref.sum().backward()

        torch.testing.assert_close(so3_ours.grad, so3_ref.grad, **get_bwd_tolerances(dtype, Operator.SO3_Jinvp))
        torch.testing.assert_close(p_ours.grad, p_ref.grad, **get_bwd_tolerances(dtype, Operator.SO3_Jinvp))

    def test_grad_device(self, device):
        """Test that gradients are on the correct device."""
        dtype = torch.float32
        so3 = pp.randn_SO3(5, device=device, dtype=dtype)
        so3.requires_grad_(True)
        p = torch.randn(5, 3, device=device, dtype=dtype, requires_grad=True)

        result = SO3_Jinvp.apply(so3, p)
        result.tensor().sum().backward()

        assert so3.grad.device == so3.device
        assert p.grad.device == p.device


class TestSO3_Jinvp_Bwd_Gradcheck:
    """Test SO3_Jinvp gradient w.r.t p using torch.autograd.gradcheck."""

    def test_gradcheck_p_1d(self, device):
        """Test gradcheck for p with 1D batch."""
        dtype = torch.float64  # gradcheck requires float64
        so3 = pp.randn_SO3(3, device=device, dtype=dtype)
        p = torch.randn(3, 3, device=device, dtype=dtype, requires_grad=True)

        def func(p_input):
            return SO3_Jinvp.apply(so3, p_input).tensor()

        assert torch.autograd.gradcheck(func, (p,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gradcheck_p_2d(self, device):
        """Test gradcheck for p with 2D batch."""
        dtype = torch.float64
        so3 = pp.randn_SO3(2, 3, device=device, dtype=dtype)
        p = torch.randn(2, 3, 3, device=device, dtype=dtype, requires_grad=True)

        def func(p_input):
            return SO3_Jinvp.apply(so3, p_input).tensor()

        assert torch.autograd.gradcheck(func, (p,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gradcheck_p_scalar(self, device):
        """Test gradcheck for p with scalar input."""
        dtype = torch.float64
        so3 = pp.randn_SO3(device=device, dtype=dtype)
        p = torch.randn(3, device=device, dtype=dtype, requires_grad=True)

        def func(p_input):
            return SO3_Jinvp.apply(so3, p_input).tensor()

        assert torch.autograd.gradcheck(func, (p,), eps=1e-6, atol=1e-4, rtol=1e-3)
