"""Unit tests for SE3_Mul forward and backward."""
import pytest
import torch
import pypose as pp
from pypose_warp.ltype.SE3_group import SE3_Mul, SE3_Mul_fwd
from conftest import get_fwd_tolerances, get_bwd_tolerances, Operator


class TestSE3MulBatchDimensions:
    """Test SE3_Mul_fwd with various batch dimensions."""

    def test_1d_batch(self, device, dtype):
        """Test with 1D batch dimension."""
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        Y = pp.randn_SE3(5, device=device, dtype=dtype)

        result = SE3_Mul_fwd(X, Y)
        expected = X @ Y

        assert result.shape == expected.shape == (5, 7)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_Mul))

    def test_2d_batch(self, device, dtype):
        """Test with 2D batch dimensions."""
        X = pp.randn_SE3(3, 4, device=device, dtype=dtype)
        Y = pp.randn_SE3(3, 4, device=device, dtype=dtype)

        result = SE3_Mul_fwd(X, Y)
        expected = X @ Y

        assert result.shape == expected.shape == (3, 4, 7)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_Mul))

    def test_3d_batch(self, device, dtype):
        """Test with 3D batch dimensions."""
        X = pp.randn_SE3(2, 3, 4, device=device, dtype=dtype)
        Y = pp.randn_SE3(2, 3, 4, device=device, dtype=dtype)

        result = SE3_Mul_fwd(X, Y)
        expected = X @ Y

        assert result.shape == expected.shape == (2, 3, 4, 7)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_Mul))

    def test_4d_batch(self, device, dtype):
        """Test with 4D batch dimensions."""
        X = pp.randn_SE3(2, 3, 4, 5, device=device, dtype=dtype)
        Y = pp.randn_SE3(2, 3, 4, 5, device=device, dtype=dtype)

        result = SE3_Mul_fwd(X, Y)
        expected = X @ Y

        assert result.shape == expected.shape == (2, 3, 4, 5, 7)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_Mul))

    def test_scalar_no_batch(self, device, dtype):
        """Test with no batch dimensions (single transform)."""
        X = pp.randn_SE3(device=device, dtype=dtype)
        Y = pp.randn_SE3(device=device, dtype=dtype)

        result = SE3_Mul_fwd(X, Y)
        expected = X @ Y

        assert result.shape == expected.shape == (7,)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_Mul))


class TestSE3MulBroadcasting:
    """Test SE3_Mul_fwd broadcasting behavior."""

    def test_broadcast_1d_to_2d(self, device, dtype):
        """Test broadcasting from 1D to 2D."""
        X = pp.randn_SE3(4, device=device, dtype=dtype)
        Y = pp.randn_SE3(3, 4, device=device, dtype=dtype)

        result = SE3_Mul_fwd(X, Y)
        expected = X @ Y

        assert result.shape == (3, 4, 7)
        torch.testing.assert_close(result, expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_Mul))

    def test_broadcast_scalar_to_batch(self, device, dtype):
        """Test broadcasting a single transform with batched transforms."""
        X = pp.randn_SE3(device=device, dtype=dtype)
        Y = pp.randn_SE3(5, device=device, dtype=dtype)

        result = SE3_Mul_fwd(X, Y)
        expected = X @ Y

        assert result.shape == (5, 7)
        torch.testing.assert_close(result, expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_Mul))

    def test_broadcast_different_batch_dims(self, device, dtype):
        """Test broadcasting with different batch dimensions."""
        X = pp.randn_SE3(1, 4, device=device, dtype=dtype)
        Y = pp.randn_SE3(3, 1, device=device, dtype=dtype)

        result = SE3_Mul_fwd(X, Y)
        expected = X @ Y

        assert result.shape == (3, 4, 7)
        torch.testing.assert_close(result, expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_Mul))


class TestSE3MulPrecision:
    """Test SE3_Mul_fwd precision handling."""

    def test_fp32_precision(self, device):
        """Test float32 precision and accuracy."""
        dtype = torch.float32
        X = pp.randn_SE3(10, device=device, dtype=dtype)
        Y = pp.randn_SE3(10, device=device, dtype=dtype)

        result = SE3_Mul_fwd(X, Y)
        expected = X @ Y

        torch.testing.assert_close(result, expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_Mul))

    def test_fp64_precision(self, device):
        """Test float64 precision and accuracy."""
        dtype = torch.float64
        X = pp.randn_SE3(10, device=device, dtype=dtype)
        Y = pp.randn_SE3(10, device=device, dtype=dtype)

        result = SE3_Mul_fwd(X, Y)
        expected = X @ Y

        torch.testing.assert_close(result, expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_Mul))

    def test_fp16_precision(self, device):
        """Test float16 precision and accuracy."""
        dtype = torch.float16
        X = pp.randn_SE3(10, device=device, dtype=dtype)
        Y = pp.randn_SE3(10, device=device, dtype=dtype)

        result = SE3_Mul_fwd(X, Y)
        expected = X @ Y

        torch.testing.assert_close(result, expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_Mul))

    def test_output_dtype_preserved(self, device, dtype):
        """Test that output dtype matches input dtype."""
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        Y = pp.randn_SE3(5, device=device, dtype=dtype)

        result = SE3_Mul_fwd(X, Y)

        assert result.dtype == dtype


class TestSE3MulEdgeCases:
    """Test SE3_Mul_fwd edge cases."""

    def test_identity_left(self, device):
        """Test that identity @ X = X."""
        dtype = torch.float32
        identity = pp.identity_SE3(5, device=device, dtype=dtype)
        X = pp.randn_SE3(5, device=device, dtype=dtype)

        result = SE3_Mul_fwd(identity, X)

        torch.testing.assert_close(result, X.tensor())

    def test_identity_right(self, device):
        """Test that X @ identity = X."""
        dtype = torch.float32
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        identity = pp.identity_SE3(5, device=device, dtype=dtype)

        result = SE3_Mul_fwd(X, identity)

        torch.testing.assert_close(result, X.tensor())

    def test_inverse_composition(self, device, dtype):
        """Test that X @ inv(X) = identity."""
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        X_inv = X.Inv()

        result = SE3_Mul_fwd(X, X_inv)
        identity = pp.identity_SE3(5, device=device, dtype=dtype)

        torch.testing.assert_close(result, identity.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_Mul))

    def test_single_element_batch(self, device):
        """Test with batch size of 1."""
        dtype = torch.float32
        X = pp.randn_SE3(1, device=device, dtype=dtype)
        Y = pp.randn_SE3(1, device=device, dtype=dtype)

        result = SE3_Mul_fwd(X, Y)
        expected = X @ Y

        assert result.shape == (1, 7)
        torch.testing.assert_close(result, expected.tensor())

    def test_large_batch(self, device):
        """Test with large batch size."""
        dtype = torch.float32
        X = pp.randn_SE3(1000, device=device, dtype=dtype)
        Y = pp.randn_SE3(1000, device=device, dtype=dtype)

        result = SE3_Mul_fwd(X, Y)
        expected = X @ Y

        assert result.shape == (1000, 7)
        torch.testing.assert_close(result, expected.tensor())

    def test_output_device(self, device):
        """Test that output is on the same device as input."""
        dtype = torch.float32
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        Y = pp.randn_SE3(5, device=device, dtype=dtype)

        result = SE3_Mul_fwd(X, Y)

        assert result.device == X.device


class TestSE3MulMathematicalProperties:
    """Test mathematical properties of SE3 multiplication."""

    def test_associativity(self, device, dtype):
        """Test that (X @ Y) @ Z = X @ (Y @ Z)."""
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        Y = pp.randn_SE3(5, device=device, dtype=dtype)
        Z = pp.randn_SE3(5, device=device, dtype=dtype)

        # (X @ Y) @ Z
        XY = SE3_Mul_fwd(X, Y)
        XY_lie = pp.SE3(XY)
        left = SE3_Mul_fwd(XY_lie, Z)

        # X @ (Y @ Z)
        YZ = SE3_Mul_fwd(Y, Z)
        YZ_lie = pp.SE3(YZ)
        right = SE3_Mul_fwd(X, YZ_lie)

        torch.testing.assert_close(left, right, **get_fwd_tolerances(dtype, Operator.SE3_Mul))

    def test_quaternion_norm_preserved(self, device, dtype):
        """Test that quaternion part has unit norm after multiplication."""
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        Y = pp.randn_SE3(5, device=device, dtype=dtype)

        result = SE3_Mul_fwd(X, Y)
        
        # Quaternion is components 3:7
        q = result[..., 3:]
        q_norm = torch.norm(q, dim=-1)
        
        # Use appropriate tolerance for dtype
        if dtype == torch.float16:
            torch.testing.assert_close(q_norm, torch.ones_like(q_norm), atol=1e-2, rtol=1e-2)
        else:
            torch.testing.assert_close(q_norm, torch.ones_like(q_norm), atol=1e-4, rtol=1e-4)


class TestSE3MulErrors:
    """Test SE3_Mul_fwd error handling."""

    def test_5d_batch_raises(self, device):
        """Test that 5D batch dimensions raise NotImplementedError."""
        dtype = torch.float32
        X = pp.randn_SE3(2, 2, 2, 2, 2, device=device, dtype=dtype)
        Y = pp.randn_SE3(2, 2, 2, 2, 2, device=device, dtype=dtype)

        with pytest.raises(NotImplementedError, match="Batch dimensions > 4"):
            SE3_Mul_fwd(X, Y)


# =============================================================================
# Backward Pass Tests
# =============================================================================

class TestSE3MulBwdBatchDimensions:
    """Test SE3_Mul backward with various batch dimensions."""

    def test_1d_batch(self, device, dtype_bwd):
        """Test backward with 1D batch dimension."""
        X_data = pp.randn_SE3(5, device=device, dtype=dtype_bwd)
        Y_data = pp.randn_SE3(5, device=device, dtype=dtype_bwd)

        # Our implementation
        X_ours = X_data.clone().requires_grad_(True)
        Y_ours = Y_data.clone().requires_grad_(True)
        result_ours = SE3_Mul.apply(X_ours, Y_ours)
        result_ours.sum().backward()

        # PyPose reference
        X_ref = X_data.clone().requires_grad_(True)
        Y_ref = Y_data.clone().requires_grad_(True)
        result_ref = X_ref @ Y_ref
        result_ref.sum().backward()

        assert X_ours.grad.shape == X_ref.grad.shape == (5, 7)
        assert Y_ours.grad.shape == Y_ref.grad.shape == (5, 7)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Mul))
        torch.testing.assert_close(Y_ours.grad, Y_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Mul))

    def test_2d_batch(self, device, dtype_bwd):
        """Test backward with 2D batch dimensions."""
        X_data = pp.randn_SE3(3, 4, device=device, dtype=dtype_bwd)
        Y_data = pp.randn_SE3(3, 4, device=device, dtype=dtype_bwd)

        X_ours = X_data.clone().requires_grad_(True)
        Y_ours = Y_data.clone().requires_grad_(True)
        result_ours = SE3_Mul.apply(X_ours, Y_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        Y_ref = Y_data.clone().requires_grad_(True)
        result_ref = X_ref @ Y_ref
        result_ref.sum().backward()

        assert X_ours.grad.shape == X_ref.grad.shape == (3, 4, 7)
        assert Y_ours.grad.shape == Y_ref.grad.shape == (3, 4, 7)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Mul))
        torch.testing.assert_close(Y_ours.grad, Y_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Mul))

    def test_3d_batch(self, device, dtype_bwd):
        """Test backward with 3D batch dimensions."""
        X_data = pp.randn_SE3(2, 3, 4, device=device, dtype=dtype_bwd)
        Y_data = pp.randn_SE3(2, 3, 4, device=device, dtype=dtype_bwd)

        X_ours = X_data.clone().requires_grad_(True)
        Y_ours = Y_data.clone().requires_grad_(True)
        result_ours = SE3_Mul.apply(X_ours, Y_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        Y_ref = Y_data.clone().requires_grad_(True)
        result_ref = X_ref @ Y_ref
        result_ref.sum().backward()

        assert X_ours.grad.shape == X_ref.grad.shape == (2, 3, 4, 7)
        assert Y_ours.grad.shape == Y_ref.grad.shape == (2, 3, 4, 7)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Mul))
        torch.testing.assert_close(Y_ours.grad, Y_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Mul))

    def test_4d_batch(self, device, dtype_bwd):
        """Test backward with 4D batch dimensions."""
        X_data = pp.randn_SE3(2, 2, 3, 4, device=device, dtype=dtype_bwd)
        Y_data = pp.randn_SE3(2, 2, 3, 4, device=device, dtype=dtype_bwd)

        X_ours = X_data.clone().requires_grad_(True)
        Y_ours = Y_data.clone().requires_grad_(True)
        result_ours = SE3_Mul.apply(X_ours, Y_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        Y_ref = Y_data.clone().requires_grad_(True)
        result_ref = X_ref @ Y_ref
        result_ref.sum().backward()

        assert X_ours.grad.shape == X_ref.grad.shape == (2, 2, 3, 4, 7)
        assert Y_ours.grad.shape == Y_ref.grad.shape == (2, 2, 3, 4, 7)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Mul))
        torch.testing.assert_close(Y_ours.grad, Y_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Mul))

    def test_scalar_no_batch(self, device, dtype_bwd):
        """Test backward with no batch dimensions."""
        X_data = pp.randn_SE3(device=device, dtype=dtype_bwd)
        Y_data = pp.randn_SE3(device=device, dtype=dtype_bwd)

        X_ours = X_data.clone().requires_grad_(True)
        Y_ours = Y_data.clone().requires_grad_(True)
        result_ours = SE3_Mul.apply(X_ours, Y_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        Y_ref = Y_data.clone().requires_grad_(True)
        result_ref = X_ref @ Y_ref
        result_ref.sum().backward()

        assert X_ours.grad.shape == X_ref.grad.shape == (7,)
        assert Y_ours.grad.shape == Y_ref.grad.shape == (7,)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Mul))
        torch.testing.assert_close(Y_ours.grad, Y_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Mul))


class TestSE3MulBwdBroadcasting:
    """Test SE3_Mul backward with broadcasting."""

    def test_broadcast_1d_to_2d(self, device, dtype_bwd):
        """Test backward with broadcasting from 1D to 2D."""
        X_data = pp.randn_SE3(4, device=device, dtype=dtype_bwd)
        Y_data = pp.randn_SE3(3, 4, device=device, dtype=dtype_bwd)

        X_ours = X_data.clone().requires_grad_(True)
        Y_ours = Y_data.clone().requires_grad_(True)
        result_ours = SE3_Mul.apply(X_ours, Y_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        Y_ref = Y_data.clone().requires_grad_(True)
        result_ref = X_ref @ Y_ref
        result_ref.sum().backward()

        assert X_ours.grad.shape == X_ref.grad.shape == (4, 7)
        assert Y_ours.grad.shape == Y_ref.grad.shape == (3, 4, 7)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Mul))
        torch.testing.assert_close(Y_ours.grad, Y_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Mul))

    def test_broadcast_different_dims(self, device, dtype_bwd):
        """Test backward with broadcasting across different dimensions."""
        X_data = pp.randn_SE3(1, 4, device=device, dtype=dtype_bwd)
        Y_data = pp.randn_SE3(3, 1, device=device, dtype=dtype_bwd)

        X_ours = X_data.clone().requires_grad_(True)
        Y_ours = Y_data.clone().requires_grad_(True)
        result_ours = SE3_Mul.apply(X_ours, Y_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        Y_ref = Y_data.clone().requires_grad_(True)
        result_ref = X_ref @ Y_ref
        result_ref.sum().backward()

        assert X_ours.grad.shape == X_ref.grad.shape == (1, 4, 7)
        assert Y_ours.grad.shape == Y_ref.grad.shape == (3, 1, 7)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Mul))
        torch.testing.assert_close(Y_ours.grad, Y_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Mul))


class TestSE3MulBwdPrecision:
    """Test SE3_Mul backward precision handling."""

    def test_precision(self, device, dtype_bwd):
        """Test backward precision."""
        X_data = pp.randn_SE3(10, device=device, dtype=dtype_bwd)
        Y_data = pp.randn_SE3(10, device=device, dtype=dtype_bwd)

        X_ours = X_data.clone().requires_grad_(True)
        Y_ours = Y_data.clone().requires_grad_(True)
        result_ours = SE3_Mul.apply(X_ours, Y_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        Y_ref = Y_data.clone().requires_grad_(True)
        result_ref = X_ref @ Y_ref
        result_ref.sum().backward()

        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Mul))
        torch.testing.assert_close(Y_ours.grad, Y_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Mul))

    def test_grad_dtype_preserved(self, device, dtype_bwd):
        """Test that gradient dtype matches input dtype."""
        X = pp.randn_SE3(5, device=device, dtype=dtype_bwd).requires_grad_(True)
        Y = pp.randn_SE3(5, device=device, dtype=dtype_bwd).requires_grad_(True)

        result = SE3_Mul.apply(X, Y)
        result.sum().backward()

        assert X.grad.dtype == dtype_bwd
        assert Y.grad.dtype == dtype_bwd


class TestSE3MulBwdEdgeCases:
    """Test SE3_Mul backward edge cases."""

    def test_quaternion_w_component_zero(self, device, dtype_bwd):
        """Test that quaternion gradient w component is always zero."""
        X = pp.randn_SE3(5, device=device, dtype=dtype_bwd).requires_grad_(True)
        Y = pp.randn_SE3(5, device=device, dtype=dtype_bwd).requires_grad_(True)

        result = SE3_Mul.apply(X, Y)
        result.sum().backward()

        # The w component of quaternion gradient should always be 0
        assert torch.allclose(X.grad[..., 6], torch.zeros_like(X.grad[..., 6]))
        assert torch.allclose(Y.grad[..., 6], torch.zeros_like(Y.grad[..., 6]))

    def test_identity_transform_gradient(self, device):
        """Test gradient through identity transformation."""
        dtype = torch.float32
        X_data = pp.identity_SE3(5, device=device, dtype=dtype)
        Y_data = pp.randn_SE3(5, device=device, dtype=dtype)

        X_ours = X_data.clone().requires_grad_(True)
        Y_ours = Y_data.clone().requires_grad_(True)
        result_ours = SE3_Mul.apply(X_ours, Y_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        Y_ref = Y_data.clone().requires_grad_(True)
        result_ref = X_ref @ Y_ref
        result_ref.sum().backward()

        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype, Operator.SE3_Mul))
        torch.testing.assert_close(Y_ours.grad, Y_ref.grad, **get_bwd_tolerances(dtype, Operator.SE3_Mul))

    def test_large_batch_gradient(self, device):
        """Test gradient with large batch."""
        dtype = torch.float32
        X_data = pp.randn_SE3(1000, device=device, dtype=dtype)
        Y_data = pp.randn_SE3(1000, device=device, dtype=dtype)

        X_ours = X_data.clone().requires_grad_(True)
        Y_ours = Y_data.clone().requires_grad_(True)
        result_ours = SE3_Mul.apply(X_ours, Y_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        Y_ref = Y_data.clone().requires_grad_(True)
        result_ref = X_ref @ Y_ref
        result_ref.sum().backward()

        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype, Operator.SE3_Mul))
        torch.testing.assert_close(Y_ours.grad, Y_ref.grad, **get_bwd_tolerances(dtype, Operator.SE3_Mul))

    def test_grad_device(self, device):
        """Test that gradients are on the correct device."""
        dtype = torch.float32
        X = pp.randn_SE3(5, device=device, dtype=dtype).requires_grad_(True)
        Y = pp.randn_SE3(5, device=device, dtype=dtype).requires_grad_(True)

        result = SE3_Mul.apply(X, Y)
        result.sum().backward()

        assert X.grad.device == X.device
        assert Y.grad.device == Y.device


class TestSE3MulWarpBackend:
    """Test SE3_Mul through the warp_SE3Type backend."""

    def test_warp_backend_integration(self, device, dtype):
        """Test that warp_SE3Type.Mul works correctly for SE3 @ SE3."""
        from pypose_warp import to_warp_backend
        
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        Y = pp.randn_SE3(5, device=device, dtype=dtype)
        
        X_warp = to_warp_backend(X)
        Y_warp = to_warp_backend(Y)

        result = X_warp @ Y_warp
        expected = X @ Y

        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_Mul))

    def test_warp_backend_gradient(self, device, dtype_bwd):
        """Test that warp_SE3Type.Mul gradients work correctly for SE3 @ SE3."""
        from pypose_warp import to_warp_backend
        
        X_data = pp.randn_SE3(5, device=device, dtype=dtype_bwd)
        Y_data = pp.randn_SE3(5, device=device, dtype=dtype_bwd)

        # Our warp implementation
        X_warp = to_warp_backend(X_data.clone()).requires_grad_(True)
        Y_warp = to_warp_backend(Y_data.clone()).requires_grad_(True)
        result_ours = X_warp @ Y_warp
        result_ours.sum().backward()

        # PyPose reference
        X_ref = X_data.clone().requires_grad_(True)
        Y_ref = Y_data.clone().requires_grad_(True)
        result_ref = X_ref @ Y_ref
        result_ref.sum().backward()

        torch.testing.assert_close(X_warp.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Mul))
        torch.testing.assert_close(Y_warp.grad, Y_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Mul))

    def test_warp_backend_act_dispatch(self, device, dtype):
        """Test that warp_SE3Type.Mul correctly dispatches to Act for Tensor operand."""
        from pypose_warp import to_warp_backend
        
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        X_warp = to_warp_backend(X)
        points = torch.randn(5, 3, device=device, dtype=dtype)

        # SE3 @ Tensor should dispatch to Act
        result = X_warp @ points
        expected = X @ points

        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SE3_Act))

