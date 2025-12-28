"""Unit tests for SE3_AdjXa forward and backward."""
import pytest
import torch
import pypose as pp
from pypose_warp.ltype.SE3_group import SE3_AdjXa, SE3_AdjXa_fwd
from conftest import get_fwd_tolerances, get_bwd_tolerances, Operator


class TestSE3AdjXaBatchDimensions:
    """Test SE3_AdjXa_fwd with various batch dimensions."""

    def test_1d_batch(self, device, dtype):
        """Test with 1D batch dimension."""
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        a = pp.randn_se3(5, device=device, dtype=dtype)

        result = SE3_AdjXa_fwd(X, a)
        expected = X.Adj(a)

        assert result.shape == expected.shape == (5, 6)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_AdjXa))

    def test_2d_batch(self, device, dtype):
        """Test with 2D batch dimensions."""
        X = pp.randn_SE3(3, 4, device=device, dtype=dtype)
        a = pp.randn_se3(3, 4, device=device, dtype=dtype)

        result = SE3_AdjXa_fwd(X, a)
        expected = X.Adj(a)

        assert result.shape == expected.shape == (3, 4, 6)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_AdjXa))

    def test_3d_batch(self, device, dtype):
        """Test with 3D batch dimensions."""
        X = pp.randn_SE3(2, 3, 4, device=device, dtype=dtype)
        a = pp.randn_se3(2, 3, 4, device=device, dtype=dtype)

        result = SE3_AdjXa_fwd(X, a)
        expected = X.Adj(a)

        assert result.shape == expected.shape == (2, 3, 4, 6)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_AdjXa))

    def test_4d_batch(self, device, dtype):
        """Test with 4D batch dimensions."""
        X = pp.randn_SE3(2, 3, 4, 5, device=device, dtype=dtype)
        a = pp.randn_se3(2, 3, 4, 5, device=device, dtype=dtype)

        result = SE3_AdjXa_fwd(X, a)
        expected = X.Adj(a)

        assert result.shape == expected.shape == (2, 3, 4, 5, 6)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_AdjXa))

    def test_scalar_no_batch(self, device, dtype):
        """Test with no batch dimensions (single transform)."""
        X = pp.randn_SE3(device=device, dtype=dtype)
        a = pp.randn_se3(device=device, dtype=dtype)

        result = SE3_AdjXa_fwd(X, a)
        expected = X.Adj(a)

        assert result.shape == expected.shape == (6,)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_AdjXa))


class TestSE3AdjXaBroadcasting:
    """Test SE3_AdjXa_fwd broadcasting behavior."""

    def test_broadcast_1d_to_2d(self, device, dtype):
        """Test broadcasting from 1D to 2D."""
        X = pp.randn_SE3(4, device=device, dtype=dtype)
        a = pp.randn_se3(3, 4, device=device, dtype=dtype)

        result = SE3_AdjXa_fwd(X, a)
        expected = X.Adj(a)

        assert result.shape == (3, 4, 6)
        torch.testing.assert_close(result, expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_AdjXa))

    def test_broadcast_scalar_to_batch(self, device, dtype):
        """Test broadcasting a single transform with batched algebra elements."""
        X = pp.randn_SE3(device=device, dtype=dtype)
        a = pp.randn_se3(5, device=device, dtype=dtype)

        result = SE3_AdjXa_fwd(X, a)
        expected = X.Adj(a)

        assert result.shape == (5, 6)
        torch.testing.assert_close(result, expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_AdjXa))

    def test_broadcast_different_batch_dims(self, device, dtype):
        """Test broadcasting with different batch dimensions."""
        X = pp.randn_SE3(1, 4, device=device, dtype=dtype)
        a = pp.randn_se3(3, 1, device=device, dtype=dtype)

        result = SE3_AdjXa_fwd(X, a)
        expected = X.Adj(a)

        assert result.shape == (3, 4, 6)
        torch.testing.assert_close(result, expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_AdjXa))


class TestSE3AdjXaPrecision:
    """Test SE3_AdjXa_fwd precision handling."""

    def test_fp32_precision(self, device):
        """Test float32 precision and accuracy."""
        dtype = torch.float32
        X = pp.randn_SE3(10, device=device, dtype=dtype)
        a = pp.randn_se3(10, device=device, dtype=dtype)

        result = SE3_AdjXa_fwd(X, a)
        expected = X.Adj(a)

        torch.testing.assert_close(result, expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_AdjXa))

    def test_fp64_precision(self, device):
        """Test float64 precision and accuracy."""
        dtype = torch.float64
        X = pp.randn_SE3(10, device=device, dtype=dtype)
        a = pp.randn_se3(10, device=device, dtype=dtype)

        result = SE3_AdjXa_fwd(X, a)
        expected = X.Adj(a)

        torch.testing.assert_close(result, expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_AdjXa))

    def test_fp16_precision(self, device):
        """Test float16 precision and accuracy."""
        dtype = torch.float16
        X = pp.randn_SE3(10, device=device, dtype=dtype)
        a = pp.randn_se3(10, device=device, dtype=dtype)

        result = SE3_AdjXa_fwd(X, a)
        expected = X.Adj(a)

        torch.testing.assert_close(result, expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_AdjXa))

    def test_output_dtype_preserved(self, device, dtype):
        """Test that output dtype matches input dtype."""
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        a = pp.randn_se3(5, device=device, dtype=dtype)

        result = SE3_AdjXa_fwd(X, a)

        assert result.dtype == dtype


class TestSE3AdjXaEdgeCases:
    """Test SE3_AdjXa_fwd edge cases."""

    def test_identity_transform(self, device):
        """Test adjoint with identity SE3 (should be identity adjoint)."""
        dtype = torch.float32
        identity = pp.identity_SE3(5, device=device, dtype=dtype)
        a = pp.randn_se3(5, device=device, dtype=dtype)

        result = SE3_AdjXa_fwd(identity, a)
        # For identity SE3, Adj is identity 6x6, so out = a
        torch.testing.assert_close(result, a.tensor())

    def test_zero_algebra_element(self, device):
        """Test adjoint with zero se3 element."""
        dtype = torch.float32
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        a = pp.identity_se3(5, device=device, dtype=dtype)  # Zero algebra element

        result = SE3_AdjXa_fwd(X, a)
        expected = torch.zeros_like(result)
        
        torch.testing.assert_close(result, expected)

    def test_single_element_batch(self, device):
        """Test with batch size of 1."""
        dtype = torch.float32
        X = pp.randn_SE3(1, device=device, dtype=dtype)
        a = pp.randn_se3(1, device=device, dtype=dtype)

        result = SE3_AdjXa_fwd(X, a)
        expected = X.Adj(a)

        assert result.shape == (1, 6)
        torch.testing.assert_close(result, expected.tensor())

    def test_large_batch(self, device):
        """Test with large batch size."""
        dtype = torch.float32
        X = pp.randn_SE3(1000, device=device, dtype=dtype)
        a = pp.randn_se3(1000, device=device, dtype=dtype)

        result = SE3_AdjXa_fwd(X, a)
        expected = X.Adj(a)

        assert result.shape == (1000, 6)
        torch.testing.assert_close(result, expected.tensor())

    def test_output_device(self, device):
        """Test that output is on the same device as input."""
        dtype = torch.float32
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        a = pp.randn_se3(5, device=device, dtype=dtype)

        result = SE3_AdjXa_fwd(X, a)

        assert result.device == X.device


class TestSE3AdjXaErrors:
    """Test SE3_AdjXa_fwd error handling."""

    def test_5d_batch_raises(self, device):
        """Test that 5D batch dimensions raise NotImplementedError."""
        dtype = torch.float32
        X = pp.randn_SE3(2, 2, 2, 2, 2, device=device, dtype=dtype)
        a = pp.randn_se3(2, 2, 2, 2, 2, device=device, dtype=dtype)

        with pytest.raises(NotImplementedError, match="Batch dimensions > 4"):
            SE3_AdjXa_fwd(X, a)


# =============================================================================
# Backward Pass Tests
# =============================================================================

class TestSE3AdjXaBwdBatchDimensions:
    """Test SE3_AdjXa backward with various batch dimensions."""

    def test_1d_batch(self, device, dtype_bwd):
        """Test backward with 1D batch dimension."""
        X_data = pp.randn_SE3(5, device=device, dtype=dtype_bwd)
        a_data = pp.randn_se3(5, device=device, dtype=dtype_bwd)

        # Our implementation
        X_ours = X_data.clone().requires_grad_(True)
        a_ours = a_data.clone().requires_grad_(True)
        result_ours = SE3_AdjXa.apply(X_ours, a_ours)
        result_ours.sum().backward()

        # PyPose reference
        X_ref = X_data.clone().requires_grad_(True)
        a_ref = a_data.clone().requires_grad_(True)
        result_ref = X_ref.Adj(a_ref)
        result_ref.sum().backward()

        assert X_ours.grad.shape == X_ref.grad.shape == (5, 7)
        assert a_ours.grad.shape == a_ref.grad.shape == (5, 6)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AdjXa))
        torch.testing.assert_close(a_ours.grad, a_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AdjXa))

    def test_2d_batch(self, device, dtype_bwd):
        """Test backward with 2D batch dimensions."""
        X_data = pp.randn_SE3(3, 4, device=device, dtype=dtype_bwd)
        a_data = pp.randn_se3(3, 4, device=device, dtype=dtype_bwd)

        X_ours = X_data.clone().requires_grad_(True)
        a_ours = a_data.clone().requires_grad_(True)
        result_ours = SE3_AdjXa.apply(X_ours, a_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        a_ref = a_data.clone().requires_grad_(True)
        result_ref = X_ref.Adj(a_ref)
        result_ref.sum().backward()

        assert X_ours.grad.shape == X_ref.grad.shape == (3, 4, 7)
        assert a_ours.grad.shape == a_ref.grad.shape == (3, 4, 6)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AdjXa))
        torch.testing.assert_close(a_ours.grad, a_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AdjXa))

    def test_3d_batch(self, device, dtype_bwd):
        """Test backward with 3D batch dimensions."""
        X_data = pp.randn_SE3(2, 3, 4, device=device, dtype=dtype_bwd)
        a_data = pp.randn_se3(2, 3, 4, device=device, dtype=dtype_bwd)

        X_ours = X_data.clone().requires_grad_(True)
        a_ours = a_data.clone().requires_grad_(True)
        result_ours = SE3_AdjXa.apply(X_ours, a_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        a_ref = a_data.clone().requires_grad_(True)
        result_ref = X_ref.Adj(a_ref)
        result_ref.sum().backward()

        assert X_ours.grad.shape == X_ref.grad.shape == (2, 3, 4, 7)
        assert a_ours.grad.shape == a_ref.grad.shape == (2, 3, 4, 6)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AdjXa))
        torch.testing.assert_close(a_ours.grad, a_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AdjXa))

    def test_4d_batch(self, device, dtype_bwd):
        """Test backward with 4D batch dimensions."""
        X_data = pp.randn_SE3(2, 2, 3, 4, device=device, dtype=dtype_bwd)
        a_data = pp.randn_se3(2, 2, 3, 4, device=device, dtype=dtype_bwd)

        X_ours = X_data.clone().requires_grad_(True)
        a_ours = a_data.clone().requires_grad_(True)
        result_ours = SE3_AdjXa.apply(X_ours, a_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        a_ref = a_data.clone().requires_grad_(True)
        result_ref = X_ref.Adj(a_ref)
        result_ref.sum().backward()

        assert X_ours.grad.shape == X_ref.grad.shape == (2, 2, 3, 4, 7)
        assert a_ours.grad.shape == a_ref.grad.shape == (2, 2, 3, 4, 6)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AdjXa))
        torch.testing.assert_close(a_ours.grad, a_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AdjXa))

    def test_scalar_no_batch(self, device, dtype_bwd):
        """Test backward with no batch dimensions."""
        X_data = pp.randn_SE3(device=device, dtype=dtype_bwd)
        a_data = pp.randn_se3(device=device, dtype=dtype_bwd)

        X_ours = X_data.clone().requires_grad_(True)
        a_ours = a_data.clone().requires_grad_(True)
        result_ours = SE3_AdjXa.apply(X_ours, a_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        a_ref = a_data.clone().requires_grad_(True)
        result_ref = X_ref.Adj(a_ref)
        result_ref.sum().backward()

        assert X_ours.grad.shape == X_ref.grad.shape == (7,)
        assert a_ours.grad.shape == a_ref.grad.shape == (6,)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AdjXa))
        torch.testing.assert_close(a_ours.grad, a_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AdjXa))


class TestSE3AdjXaBwdBroadcasting:
    """Test SE3_AdjXa backward with broadcasting."""

    def test_broadcast_1d_to_2d(self, device, dtype_bwd):
        """Test backward with broadcasting from 1D to 2D."""
        X_data = pp.randn_SE3(4, device=device, dtype=dtype_bwd)
        a_data = pp.randn_se3(3, 4, device=device, dtype=dtype_bwd)

        X_ours = X_data.clone().requires_grad_(True)
        a_ours = a_data.clone().requires_grad_(True)
        result_ours = SE3_AdjXa.apply(X_ours, a_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        a_ref = a_data.clone().requires_grad_(True)
        result_ref = X_ref.Adj(a_ref)
        result_ref.sum().backward()

        assert X_ours.grad.shape == X_ref.grad.shape == (4, 7)
        assert a_ours.grad.shape == a_ref.grad.shape == (3, 4, 6)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AdjXa))
        torch.testing.assert_close(a_ours.grad, a_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AdjXa))

    def test_broadcast_different_dims(self, device, dtype_bwd):
        """Test backward with broadcasting across different dimensions."""
        X_data = pp.randn_SE3(1, 4, device=device, dtype=dtype_bwd)
        a_data = pp.randn_se3(3, 1, device=device, dtype=dtype_bwd)

        X_ours = X_data.clone().requires_grad_(True)
        a_ours = a_data.clone().requires_grad_(True)
        result_ours = SE3_AdjXa.apply(X_ours, a_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        a_ref = a_data.clone().requires_grad_(True)
        result_ref = X_ref.Adj(a_ref)
        result_ref.sum().backward()

        assert X_ours.grad.shape == X_ref.grad.shape == (1, 4, 7)
        assert a_ours.grad.shape == a_ref.grad.shape == (3, 1, 6)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AdjXa))
        torch.testing.assert_close(a_ours.grad, a_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AdjXa))


class TestSE3AdjXaBwdPrecision:
    """Test SE3_AdjXa backward precision handling."""

    def test_precision(self, device, dtype_bwd):
        """Test backward precision."""
        X_data = pp.randn_SE3(10, device=device, dtype=dtype_bwd)
        a_data = pp.randn_se3(10, device=device, dtype=dtype_bwd)

        X_ours = X_data.clone().requires_grad_(True)
        a_ours = a_data.clone().requires_grad_(True)
        result_ours = SE3_AdjXa.apply(X_ours, a_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        a_ref = a_data.clone().requires_grad_(True)
        result_ref = X_ref.Adj(a_ref)
        result_ref.sum().backward()

        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AdjXa))
        torch.testing.assert_close(a_ours.grad, a_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AdjXa))

    def test_grad_dtype_preserved(self, device, dtype_bwd):
        """Test that gradient dtype matches input dtype."""
        X = pp.randn_SE3(5, device=device, dtype=dtype_bwd).requires_grad_(True)
        a = pp.randn_se3(5, device=device, dtype=dtype_bwd).requires_grad_(True)

        result = SE3_AdjXa.apply(X, a)
        result.sum().backward()

        assert X.grad.dtype == dtype_bwd
        assert a.grad.dtype == dtype_bwd


class TestSE3AdjXaBwdEdgeCases:
    """Test SE3_AdjXa backward edge cases."""

    def test_quaternion_w_component_zero(self, device, dtype_bwd):
        """Test that quaternion gradient w component is always zero."""
        X = pp.randn_SE3(5, device=device, dtype=dtype_bwd).requires_grad_(True)
        a = pp.randn_se3(5, device=device, dtype=dtype_bwd).requires_grad_(True)

        result = SE3_AdjXa.apply(X, a)
        result.sum().backward()

        # The w component of quaternion gradient should always be 0
        assert torch.allclose(X.grad[..., 6], torch.zeros_like(X.grad[..., 6]))

    def test_identity_transform_gradient(self, device):
        """Test gradient through identity transformation."""
        dtype = torch.float32
        X_data = pp.identity_SE3(5, device=device, dtype=dtype)
        a_data = pp.randn_se3(5, device=device, dtype=dtype)

        X_ours = X_data.clone().requires_grad_(True)
        a_ours = a_data.clone().requires_grad_(True)
        result_ours = SE3_AdjXa.apply(X_ours, a_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        a_ref = a_data.clone().requires_grad_(True)
        result_ref = X_ref.Adj(a_ref)
        result_ref.sum().backward()

        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype, Operator.SE3_AdjXa))
        torch.testing.assert_close(a_ours.grad, a_ref.grad, **get_bwd_tolerances(dtype, Operator.SE3_AdjXa))

    def test_large_batch_gradient(self, device):
        """Test gradient with large batch."""
        dtype = torch.float32
        X_data = pp.randn_SE3(1000, device=device, dtype=dtype)
        a_data = pp.randn_se3(1000, device=device, dtype=dtype)

        X_ours = X_data.clone().requires_grad_(True)
        a_ours = a_data.clone().requires_grad_(True)
        result_ours = SE3_AdjXa.apply(X_ours, a_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        a_ref = a_data.clone().requires_grad_(True)
        result_ref = X_ref.Adj(a_ref)
        result_ref.sum().backward()

        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype, Operator.SE3_AdjXa))
        torch.testing.assert_close(a_ours.grad, a_ref.grad, **get_bwd_tolerances(dtype, Operator.SE3_AdjXa))

    def test_grad_device(self, device):
        """Test that gradients are on the correct device."""
        dtype = torch.float32
        X = pp.randn_SE3(5, device=device, dtype=dtype).requires_grad_(True)
        a = pp.randn_se3(5, device=device, dtype=dtype).requires_grad_(True)

        result = SE3_AdjXa.apply(X, a)
        result.sum().backward()

        assert X.grad.device == X.device
        assert a.grad.device == a.device


class TestSE3AdjXaWarpBackend:
    """Test SE3_AdjXa through the warp_SE3Type backend."""

    def test_warp_backend_integration(self, device, dtype):
        """Test that warp_SE3Type.Adj works correctly."""
        from pypose_warp import to_warp_backend
        
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        a = pp.randn_se3(5, device=device, dtype=dtype)
        
        X_warp = to_warp_backend(X)

        result = X_warp.Adj(a)
        expected = X.Adj(a)

        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_AdjXa))

    def test_warp_backend_gradient(self, device, dtype_bwd):
        """Test that warp_SE3Type.Adj gradients work correctly."""
        from pypose_warp import to_warp_backend
        
        X_data = pp.randn_SE3(5, device=device, dtype=dtype_bwd)
        a_data = pp.randn_se3(5, device=device, dtype=dtype_bwd)

        # Our warp implementation
        X_warp = to_warp_backend(X_data.clone()).requires_grad_(True)
        a_warp = a_data.clone().requires_grad_(True)
        result_ours = X_warp.Adj(a_warp)
        result_ours.sum().backward()

        # PyPose reference
        X_ref = X_data.clone().requires_grad_(True)
        a_ref = a_data.clone().requires_grad_(True)
        result_ref = X_ref.Adj(a_ref)
        result_ref.sum().backward()

        torch.testing.assert_close(X_warp.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AdjXa))
        torch.testing.assert_close(a_warp.grad, a_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AdjXa))

