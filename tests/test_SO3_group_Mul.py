"""Unit tests for SO3_Mul forward and backward."""
import pytest
import torch
import pypose as pp

from pypose_warp import to_pypose_backend
from pypose_warp.ltype.SO3_group.Mul import SO3_Mul
from pypose_warp.ltype.SO3_group.Mul.fwd import SO3_Mul_fwd
from conftest import get_fwd_tolerances, get_bwd_tolerances, Operator


class TestSO3MulBatchDimensions:
    """Test SO3_Mul_fwd with various batch dimensions."""

    def test_1d_batch(self, device, dtype):
        """Test with 1D batch dimension."""
        X = pp.randn_SO3(5, device=device, dtype=dtype)
        Y = pp.randn_SO3(5, device=device, dtype=dtype)

        result = SO3_Mul_fwd(X, Y)
        expected = X @ Y

        assert result.shape == expected.shape == (5, 4)
        assert to_pypose_backend(result).ltype == expected.ltype == pp.SO3_type
        assert result.dtype == dtype
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.SO3_Mul))

    def test_2d_batch(self, device, dtype):
        """Test with 2D batch dimensions."""
        X = pp.randn_SO3(3, 4, device=device, dtype=dtype)
        Y = pp.randn_SO3(3, 4, device=device, dtype=dtype)

        result = SO3_Mul_fwd(X, Y)
        expected = X @ Y

        assert result.shape == expected.shape == (3, 4, 4)
        assert to_pypose_backend(result).ltype == pp.SO3_type
        assert result.dtype == dtype
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.SO3_Mul))

    def test_3d_batch(self, device, dtype):
        """Test with 3D batch dimensions."""
        X = pp.randn_SO3(2, 3, 4, device=device, dtype=dtype)
        Y = pp.randn_SO3(2, 3, 4, device=device, dtype=dtype)

        result = SO3_Mul_fwd(X, Y)
        expected = X @ Y

        assert result.shape == expected.shape == (2, 3, 4, 4)
        assert to_pypose_backend(result).ltype == pp.SO3_type
        assert result.dtype == dtype
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.SO3_Mul))

    def test_4d_batch(self, device, dtype):
        """Test with 4D batch dimensions."""
        X = pp.randn_SO3(2, 3, 4, 5, device=device, dtype=dtype)
        Y = pp.randn_SO3(2, 3, 4, 5, device=device, dtype=dtype)

        result = SO3_Mul_fwd(X, Y)
        expected = X @ Y

        assert result.shape == expected.shape == (2, 3, 4, 5, 4)
        assert to_pypose_backend(result).ltype == pp.SO3_type
        assert result.dtype == dtype
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.SO3_Mul))

    def test_scalar_no_batch(self, device, dtype):
        """Test with no batch dimensions (single rotation)."""
        X = pp.randn_SO3(device=device, dtype=dtype)
        Y = pp.randn_SO3(device=device, dtype=dtype)

        result = SO3_Mul_fwd(X, Y)
        expected = X @ Y

        assert result.shape == expected.shape == (4,)
        assert to_pypose_backend(result).ltype == pp.SO3_type
        assert result.dtype == dtype
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.SO3_Mul))


class TestSO3MulBroadcasting:
    """Test SO3_Mul_fwd broadcasting."""

    def test_broadcast_1d(self, device, dtype):
        """Test broadcasting with 1D batch."""
        X = pp.randn_SO3(5, device=device, dtype=dtype)
        Y = pp.randn_SO3(1, device=device, dtype=dtype)

        result = SO3_Mul_fwd(X, Y)
        expected = X @ Y

        assert result.shape == expected.shape == (5, 4)
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.SO3_Mul))

    def test_broadcast_2d(self, device, dtype):
        """Test broadcasting with 2D batch."""
        X = pp.randn_SO3(5, 1, device=device, dtype=dtype)
        Y = pp.randn_SO3(1, 4, device=device, dtype=dtype)

        result = SO3_Mul_fwd(X, Y)
        expected = X @ Y

        assert result.shape == expected.shape == (5, 4, 4)
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.SO3_Mul))

    def test_broadcast_3d(self, device, dtype):
        """Test broadcasting with 3D batch."""
        X = pp.randn_SO3(2, 1, 4, device=device, dtype=dtype)
        Y = pp.randn_SO3(1, 3, 1, device=device, dtype=dtype)

        result = SO3_Mul_fwd(X, Y)
        expected = X @ Y

        assert result.shape == expected.shape == (2, 3, 4, 4)
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.SO3_Mul))


class TestSO3MulPrecision:
    """Test SO3_Mul_fwd precision handling."""

    def test_fp32_precision(self, device):
        """Test float32 precision and accuracy."""
        dtype = torch.float32
        X = pp.randn_SO3(10, device=device, dtype=dtype)
        Y = pp.randn_SO3(10, device=device, dtype=dtype)

        result = SO3_Mul_fwd(X, Y)
        expected = X @ Y

        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.SO3_Mul))

    def test_fp64_precision(self, device):
        """Test float64 precision and accuracy."""
        dtype = torch.float64
        X = pp.randn_SO3(10, device=device, dtype=dtype)
        Y = pp.randn_SO3(10, device=device, dtype=dtype)

        result = SO3_Mul_fwd(X, Y)
        expected = X @ Y

        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.SO3_Mul))

    def test_fp16_precision(self, device):
        """Test float16 precision and accuracy."""
        dtype = torch.float16
        X = pp.randn_SO3(10, device=device, dtype=dtype)
        Y = pp.randn_SO3(10, device=device, dtype=dtype)

        result = SO3_Mul_fwd(X, Y)
        expected = X @ Y

        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.SO3_Mul))

    def test_output_dtype_preserved(self, device, dtype):
        """Test that output dtype matches input dtype."""
        X = pp.randn_SO3(5, device=device, dtype=dtype)
        Y = pp.randn_SO3(5, device=device, dtype=dtype)

        result = SO3_Mul_fwd(X, Y)

        assert result.dtype == dtype


class TestSO3MulEdgeCases:
    """Test SO3_Mul_fwd edge cases."""

    def test_identity_right(self, device, dtype):
        """Test that X @ identity = X."""
        X = pp.randn_SO3(5, device=device, dtype=dtype)
        I = pp.identity_SO3(5, device=device, dtype=dtype)

        result = SO3_Mul_fwd(X, I)

        torch.testing.assert_close(result.tensor(), X.tensor(), **get_fwd_tolerances(dtype, Operator.SO3_Mul))

    def test_identity_left(self, device, dtype):
        """Test that identity @ Y = Y."""
        I = pp.identity_SO3(5, device=device, dtype=dtype)
        Y = pp.randn_SO3(5, device=device, dtype=dtype)

        result = SO3_Mul_fwd(I, Y)

        torch.testing.assert_close(result.tensor(), Y.tensor(), **get_fwd_tolerances(dtype, Operator.SO3_Mul))

    def test_inverse_product(self, device, dtype):
        """Test that X @ X.Inv() = identity."""
        X = pp.randn_SO3(5, device=device, dtype=dtype)
        X_inv = X.Inv()

        result = SO3_Mul_fwd(X, X_inv)
        expected = pp.identity_SO3(5, device=device, dtype=dtype)

        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.SO3_Mul))

    def test_single_element_batch(self, device):
        """Test with batch size of 1."""
        dtype = torch.float32
        X = pp.randn_SO3(1, device=device, dtype=dtype)
        Y = pp.randn_SO3(1, device=device, dtype=dtype)

        result = SO3_Mul_fwd(X, Y)
        expected = X @ Y

        assert result.shape == (1, 4)
        torch.testing.assert_close(result.tensor(), expected.tensor())

    def test_large_batch(self, device):
        """Test with large batch size."""
        dtype = torch.float32
        X = pp.randn_SO3(1000, device=device, dtype=dtype)
        Y = pp.randn_SO3(1000, device=device, dtype=dtype)

        result = SO3_Mul_fwd(X, Y)
        expected = X @ Y

        assert result.shape == (1000, 4)
        torch.testing.assert_close(result.tensor(), expected.tensor())

    def test_output_device(self, device):
        """Test that output is on the same device as input."""
        dtype = torch.float32
        X = pp.randn_SO3(5, device=device, dtype=dtype)
        Y = pp.randn_SO3(5, device=device, dtype=dtype)

        result = SO3_Mul_fwd(X, Y)

        assert result.device == X.device

    def test_output_ltype(self, device):
        """Test that output is SO3 LieTensor."""
        dtype = torch.float32
        X = pp.randn_SO3(5, device=device, dtype=dtype)
        Y = pp.randn_SO3(5, device=device, dtype=dtype)

        result = SO3_Mul_fwd(X, Y)

        assert isinstance(result, pp.LieTensor)
        assert to_pypose_backend(result).ltype == pp.SO3_type


class TestSO3MulErrors:
    """Test SO3_Mul_fwd error handling."""

    def test_5d_batch_raises(self, device):
        """Test that 5D batch dimensions raise NotImplementedError."""
        dtype = torch.float32
        X = pp.randn_SO3(2, 2, 2, 2, 2, device=device, dtype=dtype)
        Y = pp.randn_SO3(2, 2, 2, 2, 2, device=device, dtype=dtype)

        with pytest.raises(NotImplementedError, match="Batch dimensions > 4"):
            SO3_Mul_fwd(X, Y)

    def test_incompatible_shapes_raises(self, device):
        """Test that incompatible shapes raise ValueError."""
        dtype = torch.float32
        X = pp.randn_SO3(5, device=device, dtype=dtype)
        Y = pp.randn_SO3(3, device=device, dtype=dtype)

        with pytest.raises(ValueError, match="not broadcastable"):
            SO3_Mul_fwd(X, Y)


class TestSO3MulAutogradFunction:
    """Test SO3_Mul autograd function wrapper."""

    def test_apply_matches_fwd(self, device, dtype):
        """Test that SO3_Mul.apply matches SO3_Mul_fwd."""
        X = pp.randn_SO3(5, device=device, dtype=dtype)
        Y = pp.randn_SO3(5, device=device, dtype=dtype)

        result_apply = SO3_Mul.apply(X, Y)
        result_fwd = SO3_Mul_fwd(X, Y)

        torch.testing.assert_close(result_apply.tensor(), result_fwd.tensor())

    def test_apply_1d_batch(self, device, dtype):
        """Test SO3_Mul.apply with 1D batch."""
        X = pp.randn_SO3(5, device=device, dtype=dtype)
        Y = pp.randn_SO3(5, device=device, dtype=dtype)

        result = SO3_Mul.apply(X, Y)
        expected = X @ Y

        assert result.shape == expected.shape == (5, 4)
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.SO3_Mul))

    def test_apply_2d_batch(self, device, dtype):
        """Test SO3_Mul.apply with 2D batch."""
        X = pp.randn_SO3(3, 4, device=device, dtype=dtype)
        Y = pp.randn_SO3(3, 4, device=device, dtype=dtype)

        result = SO3_Mul.apply(X, Y)
        expected = X @ Y

        assert result.shape == expected.shape == (3, 4, 4)
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.SO3_Mul))


# =============================================================================
# Backward Pass Tests
# =============================================================================

class TestSO3MulBwdBatchDimensions:
    """Test SO3_Mul backward with various batch dimensions."""

    def test_1d_batch(self, device, dtype_bwd):
        """Test backward with 1D batch dimension."""
        X_data = pp.randn_SO3(5, device=device, dtype=dtype_bwd)
        Y_data = pp.randn_SO3(5, device=device, dtype=dtype_bwd)

        X_ours = X_data.clone().requires_grad_(True)
        Y_ours = Y_data.clone().requires_grad_(True)
        result_ours = SO3_Mul.apply(X_ours, Y_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        Y_ref = Y_data.clone().requires_grad_(True)
        result_ref = X_ref @ Y_ref
        result_ref.sum().backward()

        assert X_ours.grad.shape == X_ref.grad.shape == (5, 4)
        assert Y_ours.grad.shape == Y_ref.grad.shape == (5, 4)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_Mul))
        torch.testing.assert_close(Y_ours.grad, Y_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_Mul))

    def test_2d_batch(self, device, dtype_bwd):
        """Test backward with 2D batch dimensions."""
        X_data = pp.randn_SO3(3, 4, device=device, dtype=dtype_bwd)
        Y_data = pp.randn_SO3(3, 4, device=device, dtype=dtype_bwd)

        X_ours = X_data.clone().requires_grad_(True)
        Y_ours = Y_data.clone().requires_grad_(True)
        result_ours = SO3_Mul.apply(X_ours, Y_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        Y_ref = Y_data.clone().requires_grad_(True)
        result_ref = X_ref @ Y_ref
        result_ref.sum().backward()

        assert X_ours.grad.shape == X_ref.grad.shape == (3, 4, 4)
        assert Y_ours.grad.shape == Y_ref.grad.shape == (3, 4, 4)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_Mul))
        torch.testing.assert_close(Y_ours.grad, Y_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_Mul))

    def test_3d_batch(self, device, dtype_bwd):
        """Test backward with 3D batch dimensions."""
        X_data = pp.randn_SO3(2, 3, 4, device=device, dtype=dtype_bwd)
        Y_data = pp.randn_SO3(2, 3, 4, device=device, dtype=dtype_bwd)

        X_ours = X_data.clone().requires_grad_(True)
        Y_ours = Y_data.clone().requires_grad_(True)
        result_ours = SO3_Mul.apply(X_ours, Y_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        Y_ref = Y_data.clone().requires_grad_(True)
        result_ref = X_ref @ Y_ref
        result_ref.sum().backward()

        assert X_ours.grad.shape == X_ref.grad.shape == (2, 3, 4, 4)
        assert Y_ours.grad.shape == Y_ref.grad.shape == (2, 3, 4, 4)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_Mul))
        torch.testing.assert_close(Y_ours.grad, Y_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_Mul))

    def test_4d_batch(self, device, dtype_bwd):
        """Test backward with 4D batch dimensions."""
        X_data = pp.randn_SO3(2, 2, 3, 4, device=device, dtype=dtype_bwd)
        Y_data = pp.randn_SO3(2, 2, 3, 4, device=device, dtype=dtype_bwd)

        X_ours = X_data.clone().requires_grad_(True)
        Y_ours = Y_data.clone().requires_grad_(True)
        result_ours = SO3_Mul.apply(X_ours, Y_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        Y_ref = Y_data.clone().requires_grad_(True)
        result_ref = X_ref @ Y_ref
        result_ref.sum().backward()

        assert X_ours.grad.shape == X_ref.grad.shape == (2, 2, 3, 4, 4)
        assert Y_ours.grad.shape == Y_ref.grad.shape == (2, 2, 3, 4, 4)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_Mul))
        torch.testing.assert_close(Y_ours.grad, Y_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_Mul))

    def test_scalar_no_batch(self, device, dtype_bwd):
        """Test backward with no batch dimensions."""
        X_data = pp.randn_SO3(device=device, dtype=dtype_bwd)
        Y_data = pp.randn_SO3(device=device, dtype=dtype_bwd)

        X_ours = X_data.clone().requires_grad_(True)
        Y_ours = Y_data.clone().requires_grad_(True)
        result_ours = SO3_Mul.apply(X_ours, Y_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        Y_ref = Y_data.clone().requires_grad_(True)
        result_ref = X_ref @ Y_ref
        result_ref.sum().backward()

        assert X_ours.grad.shape == X_ref.grad.shape == (4,)
        assert Y_ours.grad.shape == Y_ref.grad.shape == (4,)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_Mul))
        torch.testing.assert_close(Y_ours.grad, Y_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_Mul))


class TestSO3MulBwdBroadcasting:
    """Test SO3_Mul backward with broadcasting."""

    def test_broadcast_1d(self, device, dtype_bwd):
        """Test backward with 1D broadcast."""
        X_data = pp.randn_SO3(5, device=device, dtype=dtype_bwd)
        Y_data = pp.randn_SO3(1, device=device, dtype=dtype_bwd)

        X_ours = X_data.clone().requires_grad_(True)
        Y_ours = Y_data.clone().requires_grad_(True)
        result_ours = SO3_Mul.apply(X_ours, Y_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        Y_ref = Y_data.clone().requires_grad_(True)
        result_ref = X_ref @ Y_ref
        result_ref.sum().backward()

        assert X_ours.grad.shape == X_ref.grad.shape == (5, 4)
        assert Y_ours.grad.shape == Y_ref.grad.shape == (1, 4)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_Mul))
        torch.testing.assert_close(Y_ours.grad, Y_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_Mul))

    def test_broadcast_2d(self, device, dtype_bwd):
        """Test backward with 2D broadcast."""
        X_data = pp.randn_SO3(5, 1, device=device, dtype=dtype_bwd)
        Y_data = pp.randn_SO3(1, 4, device=device, dtype=dtype_bwd)

        X_ours = X_data.clone().requires_grad_(True)
        Y_ours = Y_data.clone().requires_grad_(True)
        result_ours = SO3_Mul.apply(X_ours, Y_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        Y_ref = Y_data.clone().requires_grad_(True)
        result_ref = X_ref @ Y_ref
        result_ref.sum().backward()

        assert X_ours.grad.shape == X_ref.grad.shape == (5, 1, 4)
        assert Y_ours.grad.shape == Y_ref.grad.shape == (1, 4, 4)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_Mul))
        torch.testing.assert_close(Y_ours.grad, Y_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_Mul))


class TestSO3MulBwdPrecision:
    """Test SO3_Mul backward precision handling."""

    def test_fp32_precision(self, device):
        """Test backward float32 precision."""
        dtype = torch.float32
        X_data = pp.randn_SO3(10, device=device, dtype=dtype)
        Y_data = pp.randn_SO3(10, device=device, dtype=dtype)

        X_ours = X_data.clone().requires_grad_(True)
        Y_ours = Y_data.clone().requires_grad_(True)
        result_ours = SO3_Mul.apply(X_ours, Y_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        Y_ref = Y_data.clone().requires_grad_(True)
        result_ref = X_ref @ Y_ref
        result_ref.sum().backward()

        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype, Operator.SO3_Mul))
        torch.testing.assert_close(Y_ours.grad, Y_ref.grad, **get_bwd_tolerances(dtype, Operator.SO3_Mul))

    def test_precision(self, device, dtype_bwd):
        """Test backward precision."""
        X_data = pp.randn_SO3(10, device=device, dtype=dtype_bwd)
        Y_data = pp.randn_SO3(10, device=device, dtype=dtype_bwd)

        X_ours = X_data.clone().requires_grad_(True)
        Y_ours = Y_data.clone().requires_grad_(True)
        result_ours = SO3_Mul.apply(X_ours, Y_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        Y_ref = Y_data.clone().requires_grad_(True)
        result_ref = X_ref @ Y_ref
        result_ref.sum().backward()

        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_Mul))
        torch.testing.assert_close(Y_ours.grad, Y_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SO3_Mul))

    def test_grad_dtype_preserved(self, device, dtype_bwd):
        """Test that gradient dtype matches input dtype."""
        X = pp.randn_SO3(5, device=device, dtype=dtype_bwd).requires_grad_(True)
        Y = pp.randn_SO3(5, device=device, dtype=dtype_bwd).requires_grad_(True)

        result = SO3_Mul.apply(X, Y)
        result.sum().backward()

        assert X.grad.dtype == dtype_bwd
        assert Y.grad.dtype == dtype_bwd


class TestSO3MulBwdEdgeCases:
    """Test SO3_Mul backward edge cases."""

    def test_quaternion_w_component_zero(self, device, dtype_bwd):
        """Test that quaternion gradient w component is always zero."""
        X = pp.randn_SO3(5, device=device, dtype=dtype_bwd).requires_grad_(True)
        Y = pp.randn_SO3(5, device=device, dtype=dtype_bwd).requires_grad_(True)

        result = SO3_Mul.apply(X, Y)
        result.sum().backward()

        # The w component of quaternion gradient should always be 0
        assert torch.allclose(X.grad[..., 3], torch.zeros_like(X.grad[..., 3]))
        assert torch.allclose(Y.grad[..., 3], torch.zeros_like(Y.grad[..., 3]))

    def test_large_batch_gradient(self, device):
        """Test gradient with large batch."""
        dtype = torch.float32
        X_data = pp.randn_SO3(1000, device=device, dtype=dtype)
        Y_data = pp.randn_SO3(1000, device=device, dtype=dtype)

        X_ours = X_data.clone().requires_grad_(True)
        Y_ours = Y_data.clone().requires_grad_(True)
        result_ours = SO3_Mul.apply(X_ours, Y_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        Y_ref = Y_data.clone().requires_grad_(True)
        result_ref = X_ref @ Y_ref
        result_ref.sum().backward()

        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype, Operator.SO3_Mul))
        torch.testing.assert_close(Y_ours.grad, Y_ref.grad, **get_bwd_tolerances(dtype, Operator.SO3_Mul))

    def test_grad_device(self, device):
        """Test that gradients are on the correct device."""
        dtype = torch.float32
        X = pp.randn_SO3(5, device=device, dtype=dtype).requires_grad_(True)
        Y = pp.randn_SO3(5, device=device, dtype=dtype).requires_grad_(True)

        result = SO3_Mul.apply(X, Y)
        result.sum().backward()

        assert X.grad.device == X.device
        assert Y.grad.device == Y.device

