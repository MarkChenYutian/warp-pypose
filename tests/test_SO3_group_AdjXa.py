"""Unit tests for SO3_AdjXa forward and backward."""
import pytest
import torch
import pypose as pp

from pypose_warp.ltype.SO3_group import SO3_AdjXa, SO3_AdjXa_fwd
from conftest import get_fwd_tolerances, get_bwd_tolerances, Operator


class TestSO3AdjXaBatchDimensions:
    """Test SO3_AdjXa_fwd with various batch dimensions."""

    def test_1d_batch(self, device, dtype):
        """Test with 1D batch dimension."""
        X = pp.randn_SO3(5, device=device, dtype=dtype)
        a = pp.randn_so3(5, device=device, dtype=dtype)

        result = SO3_AdjXa_fwd(X, a)
        expected = X.Adj(a)

        assert result.shape == expected.shape == (5, 3)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SO3_AdjXa))

    def test_2d_batch(self, device, dtype):
        """Test with 2D batch dimensions."""
        X = pp.randn_SO3(3, 4, device=device, dtype=dtype)
        a = pp.randn_so3(3, 4, device=device, dtype=dtype)

        result = SO3_AdjXa_fwd(X, a)
        expected = X.Adj(a)

        assert result.shape == expected.shape == (3, 4, 3)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SO3_AdjXa))

    def test_3d_batch(self, device, dtype):
        """Test with 3D batch dimensions."""
        X = pp.randn_SO3(2, 3, 4, device=device, dtype=dtype)
        a = pp.randn_so3(2, 3, 4, device=device, dtype=dtype)

        result = SO3_AdjXa_fwd(X, a)
        expected = X.Adj(a)

        assert result.shape == expected.shape == (2, 3, 4, 3)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SO3_AdjXa))

    def test_4d_batch(self, device, dtype):
        """Test with 4D batch dimensions."""
        X = pp.randn_SO3(2, 3, 4, 5, device=device, dtype=dtype)
        a = pp.randn_so3(2, 3, 4, 5, device=device, dtype=dtype)

        result = SO3_AdjXa_fwd(X, a)
        expected = X.Adj(a)

        assert result.shape == expected.shape == (2, 3, 4, 5, 3)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SO3_AdjXa))

    def test_scalar_no_batch(self, device, dtype):
        """Test with no batch dimensions (single element)."""
        X = pp.randn_SO3(device=device, dtype=dtype)
        a = pp.randn_so3(device=device, dtype=dtype)

        result = SO3_AdjXa_fwd(X, a)
        expected = X.Adj(a)

        assert result.shape == expected.shape == (3,)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SO3_AdjXa))


class TestSO3AdjXaBroadcasting:
    """Test SO3_AdjXa_fwd broadcasting."""

    def test_broadcast_1d(self, device, dtype):
        """Test broadcasting with 1D batch."""
        X = pp.randn_SO3(5, device=device, dtype=dtype)
        a = pp.randn_so3(1, device=device, dtype=dtype)

        result = SO3_AdjXa_fwd(X, a)
        expected = X.Adj(a)

        assert result.shape == expected.shape == (5, 3)
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SO3_AdjXa))

    def test_broadcast_2d(self, device, dtype):
        """Test broadcasting with 2D batch."""
        X = pp.randn_SO3(5, 1, device=device, dtype=dtype)
        a = pp.randn_so3(1, 4, device=device, dtype=dtype)

        result = SO3_AdjXa_fwd(X, a)
        expected = X.Adj(a)

        assert result.shape == expected.shape == (5, 4, 3)
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SO3_AdjXa))

    def test_broadcast_3d(self, device, dtype):
        """Test broadcasting with 3D batch."""
        X = pp.randn_SO3(2, 1, 4, device=device, dtype=dtype)
        a = pp.randn_so3(1, 3, 1, device=device, dtype=dtype)

        result = SO3_AdjXa_fwd(X, a)
        expected = X.Adj(a)

        assert result.shape == expected.shape == (2, 3, 4, 3)
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SO3_AdjXa))


class TestSO3AdjXaPrecision:
    """Test SO3_AdjXa_fwd precision handling."""

    def test_fp32_precision(self, device):
        """Test float32 precision and accuracy."""
        dtype = torch.float32
        X = pp.randn_SO3(10, device=device, dtype=dtype)
        a = pp.randn_so3(10, device=device, dtype=dtype)

        result = SO3_AdjXa_fwd(X, a)
        expected = X.Adj(a)

        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SO3_AdjXa))

    def test_fp64_precision(self, device):
        """Test float64 precision and accuracy."""
        dtype = torch.float64
        X = pp.randn_SO3(10, device=device, dtype=dtype)
        a = pp.randn_so3(10, device=device, dtype=dtype)

        result = SO3_AdjXa_fwd(X, a)
        expected = X.Adj(a)

        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SO3_AdjXa))

    def test_fp16_precision(self, device):
        """Test float16 precision and accuracy."""
        dtype = torch.float16
        X = pp.randn_SO3(10, device=device, dtype=dtype)
        a = pp.randn_so3(10, device=device, dtype=dtype)

        result = SO3_AdjXa_fwd(X, a)
        expected = X.Adj(a)

        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SO3_AdjXa))

    def test_output_dtype_preserved(self, device, dtype):
        """Test that output dtype matches input dtype."""
        X = pp.randn_SO3(5, device=device, dtype=dtype)
        a = pp.randn_so3(5, device=device, dtype=dtype)

        result = SO3_AdjXa_fwd(X, a)

        assert result.dtype == dtype


class TestSO3AdjXaEdgeCases:
    """Test SO3_AdjXa_fwd edge cases."""

    def test_identity_rotation(self, device, dtype):
        """Test that identity rotation leaves vector unchanged."""
        X = pp.identity_SO3(5, device=device, dtype=dtype)
        a = pp.randn_so3(5, device=device, dtype=dtype)

        result = SO3_AdjXa_fwd(X, a)

        # R @ a = I @ a = a for identity rotation
        torch.testing.assert_close(result, a, **get_fwd_tolerances(dtype, Operator.SO3_AdjXa))

    def test_zero_vector(self, device, dtype):
        """Test that rotating zero vector gives zero."""
        X = pp.randn_SO3(5, device=device, dtype=dtype)
        a = pp.identity_so3(5, device=device, dtype=dtype)

        result = SO3_AdjXa_fwd(X, a)

        torch.testing.assert_close(result, a, **get_fwd_tolerances(dtype, Operator.SO3_AdjXa))

    def test_single_element_batch(self, device):
        """Test with batch size of 1."""
        dtype = torch.float32
        X = pp.randn_SO3(1, device=device, dtype=dtype)
        a = pp.randn_so3(1, device=device, dtype=dtype)

        result = SO3_AdjXa_fwd(X, a)
        expected = X.Adj(a)

        assert result.shape == (1, 3)
        torch.testing.assert_close(result, expected)

    def test_large_batch(self, device):
        """Test with large batch size."""
        dtype = torch.float32
        X = pp.randn_SO3(1000, device=device, dtype=dtype)
        a = pp.randn_so3(1000, device=device, dtype=dtype)

        result = SO3_AdjXa_fwd(X, a)
        expected = X.Adj(a)

        assert result.shape == (1000, 3)
        torch.testing.assert_close(result, expected)

    def test_output_device(self, device):
        """Test that output is on the same device as input."""
        dtype = torch.float32
        X = pp.randn_SO3(5, device=device, dtype=dtype)
        a = pp.randn_so3(5, device=device, dtype=dtype)

        result = SO3_AdjXa_fwd(X, a)

        assert result.device == X.device


class TestSO3AdjXaErrors:
    """Test SO3_AdjXa_fwd error handling."""

    def test_5d_batch_raises(self, device):
        """Test that 5D batch dimensions raise NotImplementedError."""
        dtype = torch.float32
        X = pp.randn_SO3(2, 2, 2, 2, 2, device=device, dtype=dtype)
        a = pp.randn_so3(2, 2, 2, 2, 2, device=device, dtype=dtype)

        with pytest.raises(NotImplementedError, match="Batch dimensions > 4"):
            SO3_AdjXa_fwd(X, a)

    def test_incompatible_shapes_raises(self, device):
        """Test that incompatible shapes raise ValueError."""
        dtype = torch.float32
        X = pp.randn_SO3(5, device=device, dtype=dtype)
        a = pp.randn_so3(3, device=device, dtype=dtype)

        with pytest.raises(ValueError, match="not broadcastable"):
            SO3_AdjXa_fwd(X, a)


class TestSO3AdjXaAutogradFunction:
    """Test SO3_AdjXa autograd function wrapper."""

    def test_apply_matches_fwd(self, device, dtype):
        """Test that SO3_AdjXa.apply matches SO3_AdjXa_fwd."""
        X = pp.randn_SO3(5, device=device, dtype=dtype)
        a = pp.randn_so3(5, device=device, dtype=dtype)

        result_apply = SO3_AdjXa.apply(X, a)
        result_fwd = SO3_AdjXa_fwd(X, a)

        torch.testing.assert_close(result_apply, result_fwd)

    def test_apply_1d_batch(self, device, dtype):
        """Test SO3_AdjXa.apply with 1D batch."""
        X = pp.randn_SO3(5, device=device, dtype=dtype)
        a = pp.randn_so3(5, device=device, dtype=dtype)

        result = SO3_AdjXa.apply(X, a)
        expected = X.Adj(a)

        assert result.shape == expected.shape == (5, 3)
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SO3_AdjXa))

    def test_apply_2d_batch(self, device, dtype):
        """Test SO3_AdjXa.apply with 2D batch."""
        X = pp.randn_SO3(3, 4, device=device, dtype=dtype)
        a = pp.randn_so3(3, 4, device=device, dtype=dtype)

        result = SO3_AdjXa.apply(X, a)
        expected = X.Adj(a)

        assert result.shape == expected.shape == (3, 4, 3)
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SO3_AdjXa))


# =============================================================================
# Backward Pass Tests
# =============================================================================

class TestSO3AdjXaBwdBatchDimensions:
    """Test SO3_AdjXa backward with various batch dimensions."""

    def test_1d_batch(self, device, dtype_bwd):
        """Test backward with 1D batch dimension."""
        X_data = pp.randn_SO3(5, device=device, dtype=dtype_bwd)
        a_data = pp.randn_so3(5, device=device, dtype=dtype_bwd)

        X_ours = X_data.clone().requires_grad_(True)
        a_ours = a_data.clone().requires_grad_(True)
        result_ours = SO3_AdjXa.apply(X_ours, a_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        a_ref = a_data.clone().requires_grad_(True)
        result_ref = X_ref.Adj(a_ref)
        result_ref.sum().backward()

        assert X_ours.grad.shape == X_ref.grad.shape == (5, 4)
        assert a_ours.grad.shape == a_ref.grad.shape == (5, 3)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_AdjXa))
        torch.testing.assert_close(a_ours.grad, a_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_AdjXa))

    def test_2d_batch(self, device, dtype_bwd):
        """Test backward with 2D batch dimensions."""
        X_data = pp.randn_SO3(3, 4, device=device, dtype=dtype_bwd)
        a_data = pp.randn_so3(3, 4, device=device, dtype=dtype_bwd)

        X_ours = X_data.clone().requires_grad_(True)
        a_ours = a_data.clone().requires_grad_(True)
        result_ours = SO3_AdjXa.apply(X_ours, a_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        a_ref = a_data.clone().requires_grad_(True)
        result_ref = X_ref.Adj(a_ref)
        result_ref.sum().backward()

        assert X_ours.grad.shape == X_ref.grad.shape == (3, 4, 4)
        assert a_ours.grad.shape == a_ref.grad.shape == (3, 4, 3)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_AdjXa))
        torch.testing.assert_close(a_ours.grad, a_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_AdjXa))

    def test_3d_batch(self, device, dtype_bwd):
        """Test backward with 3D batch dimensions."""
        X_data = pp.randn_SO3(2, 3, 4, device=device, dtype=dtype_bwd)
        a_data = pp.randn_so3(2, 3, 4, device=device, dtype=dtype_bwd)

        X_ours = X_data.clone().requires_grad_(True)
        a_ours = a_data.clone().requires_grad_(True)
        result_ours = SO3_AdjXa.apply(X_ours, a_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        a_ref = a_data.clone().requires_grad_(True)
        result_ref = X_ref.Adj(a_ref)
        result_ref.sum().backward()

        assert X_ours.grad.shape == X_ref.grad.shape == (2, 3, 4, 4)
        assert a_ours.grad.shape == a_ref.grad.shape == (2, 3, 4, 3)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_AdjXa))
        torch.testing.assert_close(a_ours.grad, a_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_AdjXa))

    def test_4d_batch(self, device, dtype_bwd):
        """Test backward with 4D batch dimensions."""
        X_data = pp.randn_SO3(2, 2, 3, 4, device=device, dtype=dtype_bwd)
        a_data = pp.randn_so3(2, 2, 3, 4, device=device, dtype=dtype_bwd)

        X_ours = X_data.clone().requires_grad_(True)
        a_ours = a_data.clone().requires_grad_(True)
        result_ours = SO3_AdjXa.apply(X_ours, a_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        a_ref = a_data.clone().requires_grad_(True)
        result_ref = X_ref.Adj(a_ref)
        result_ref.sum().backward()

        assert X_ours.grad.shape == X_ref.grad.shape == (2, 2, 3, 4, 4)
        assert a_ours.grad.shape == a_ref.grad.shape == (2, 2, 3, 4, 3)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_AdjXa))
        torch.testing.assert_close(a_ours.grad, a_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_AdjXa))

    def test_scalar_no_batch(self, device, dtype_bwd):
        """Test backward with no batch dimensions."""
        X_data = pp.randn_SO3(device=device, dtype=dtype_bwd)
        a_data = pp.randn_so3(device=device, dtype=dtype_bwd)

        X_ours = X_data.clone().requires_grad_(True)
        a_ours = a_data.clone().requires_grad_(True)
        result_ours = SO3_AdjXa.apply(X_ours, a_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        a_ref = a_data.clone().requires_grad_(True)
        result_ref = X_ref.Adj(a_ref)
        result_ref.sum().backward()

        assert X_ours.grad.shape == X_ref.grad.shape == (4,)
        assert a_ours.grad.shape == a_ref.grad.shape == (3,)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_AdjXa))
        torch.testing.assert_close(a_ours.grad, a_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_AdjXa))


class TestSO3AdjXaBwdBroadcasting:
    """Test SO3_AdjXa backward with broadcasting."""

    def test_broadcast_X_singleton(self, device, dtype_bwd):
        """Test backward with X broadcast from singleton."""
        X_data = pp.randn_SO3(1, device=device, dtype=dtype_bwd)
        a_data = pp.randn_so3(5, device=device, dtype=dtype_bwd)

        X_ours = X_data.clone().requires_grad_(True)
        a_ours = a_data.clone().requires_grad_(True)
        result_ours = SO3_AdjXa.apply(X_ours, a_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        a_ref = a_data.clone().requires_grad_(True)
        result_ref = X_ref.Adj(a_ref)
        result_ref.sum().backward()

        assert X_ours.grad.shape == X_ref.grad.shape == (1, 4)
        assert a_ours.grad.shape == a_ref.grad.shape == (5, 3)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_AdjXa))
        torch.testing.assert_close(a_ours.grad, a_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_AdjXa))

    def test_broadcast_a_singleton(self, device, dtype_bwd):
        """Test backward with a broadcast from singleton."""
        X_data = pp.randn_SO3(5, device=device, dtype=dtype_bwd)
        a_data = pp.randn_so3(1, device=device, dtype=dtype_bwd)

        X_ours = X_data.clone().requires_grad_(True)
        a_ours = a_data.clone().requires_grad_(True)
        result_ours = SO3_AdjXa.apply(X_ours, a_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        a_ref = a_data.clone().requires_grad_(True)
        result_ref = X_ref.Adj(a_ref)
        result_ref.sum().backward()

        assert X_ours.grad.shape == X_ref.grad.shape == (5, 4)
        assert a_ours.grad.shape == a_ref.grad.shape == (1, 3)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_AdjXa))
        torch.testing.assert_close(a_ours.grad, a_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_AdjXa))

    def test_broadcast_2d_cross(self, device, dtype_bwd):
        """Test backward with 2D cross broadcast."""
        X_data = pp.randn_SO3(5, 1, device=device, dtype=dtype_bwd)
        a_data = pp.randn_so3(1, 4, device=device, dtype=dtype_bwd)

        X_ours = X_data.clone().requires_grad_(True)
        a_ours = a_data.clone().requires_grad_(True)
        result_ours = SO3_AdjXa.apply(X_ours, a_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        a_ref = a_data.clone().requires_grad_(True)
        result_ref = X_ref.Adj(a_ref)
        result_ref.sum().backward()

        assert X_ours.grad.shape == X_ref.grad.shape == (5, 1, 4)
        assert a_ours.grad.shape == a_ref.grad.shape == (1, 4, 3)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_AdjXa))
        torch.testing.assert_close(a_ours.grad, a_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_AdjXa))

    def test_broadcast_different_ndim(self, device, dtype_bwd):
        """Test backward with different number of batch dimensions."""
        X_data = pp.randn_SO3(5, device=device, dtype=dtype_bwd)  # shape (5, 4)
        a_data = pp.randn_so3(3, 1, device=device, dtype=dtype_bwd)  # shape (3, 1, 3)

        X_ours = X_data.clone().requires_grad_(True)
        a_ours = a_data.clone().requires_grad_(True)
        result_ours = SO3_AdjXa.apply(X_ours, a_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        a_ref = a_data.clone().requires_grad_(True)
        result_ref = X_ref.Adj(a_ref)
        result_ref.sum().backward()

        assert X_ours.grad.shape == X_ref.grad.shape == (5, 4)
        assert a_ours.grad.shape == a_ref.grad.shape == (3, 1, 3)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_AdjXa))
        torch.testing.assert_close(a_ours.grad, a_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_AdjXa))


class TestSO3AdjXaBwdPrecision:
    """Test SO3_AdjXa backward precision handling."""

    def test_precision(self, device, dtype_bwd):
        """Test backward precision."""
        X_data = pp.randn_SO3(10, device=device, dtype=dtype_bwd)
        a_data = pp.randn_so3(10, device=device, dtype=dtype_bwd)

        X_ours = X_data.clone().requires_grad_(True)
        a_ours = a_data.clone().requires_grad_(True)
        result_ours = SO3_AdjXa.apply(X_ours, a_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        a_ref = a_data.clone().requires_grad_(True)
        result_ref = X_ref.Adj(a_ref)
        result_ref.sum().backward()

        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_AdjXa))
        torch.testing.assert_close(a_ours.grad, a_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_AdjXa))

    def test_grad_dtype_preserved(self, device, dtype_bwd):
        """Test that gradient dtype matches input dtype."""
        X = pp.randn_SO3(5, device=device, dtype=dtype_bwd).requires_grad_(True)
        a = pp.randn_so3(5, device=device, dtype=dtype_bwd).requires_grad_(True)

        result = SO3_AdjXa.apply(X, a)
        result.sum().backward()

        assert X.grad.dtype == dtype_bwd
        assert a.grad.dtype == dtype_bwd


class TestSO3AdjXaBwdEdgeCases:
    """Test SO3_AdjXa backward edge cases."""

    def test_quaternion_w_component_zero(self, device, dtype_bwd):
        """Test that quaternion gradient w component is always zero."""
        X = pp.randn_SO3(5, device=device, dtype=dtype_bwd).requires_grad_(True)
        a = pp.randn_so3(5, device=device, dtype=dtype_bwd).requires_grad_(True)

        result = SO3_AdjXa.apply(X, a)
        result.sum().backward()

        # The w component of quaternion gradient should always be 0
        assert torch.allclose(X.grad[..., 3], torch.zeros_like(X.grad[..., 3]))

    def test_large_batch_gradient(self, device):
        """Test gradient with large batch."""
        dtype = torch.float32
        X_data = pp.randn_SO3(1000, device=device, dtype=dtype)
        a_data = pp.randn_so3(1000, device=device, dtype=dtype)

        X_ours = X_data.clone().requires_grad_(True)
        a_ours = a_data.clone().requires_grad_(True)
        result_ours = SO3_AdjXa.apply(X_ours, a_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        a_ref = a_data.clone().requires_grad_(True)
        result_ref = X_ref.Adj(a_ref)
        result_ref.sum().backward()

        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype, Operator.SO3_AdjXa))
        torch.testing.assert_close(a_ours.grad, a_ref.grad, **get_bwd_tolerances(dtype, Operator.SO3_AdjXa))

    def test_grad_device(self, device):
        """Test that gradients are on the correct device."""
        dtype = torch.float32
        X = pp.randn_SO3(5, device=device, dtype=dtype).requires_grad_(True)
        a = pp.randn_so3(5, device=device, dtype=dtype).requires_grad_(True)

        result = SO3_AdjXa.apply(X, a)
        result.sum().backward()

        assert X.grad.device == X.device
        assert a.grad.device == a.device
