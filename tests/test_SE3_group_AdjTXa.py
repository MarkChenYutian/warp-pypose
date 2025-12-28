"""Unit tests for SE3_AdjTXa forward and backward."""
import pytest
import torch
import pypose as pp
from pypose_warp.ltype.SE3_group import SE3_AdjTXa, SE3_AdjTXa_fwd
from conftest import get_fwd_tolerances, get_bwd_tolerances, Operator, skip_if_nan_inputs, compute_reference_fp32


class TestSE3AdjTXaBatchDimensions:
    """Test SE3_AdjTXa_fwd with various batch dimensions."""

    def test_1d_batch(self, device, dtype):
        """Test with 1D batch dimension."""
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        a = pp.randn_se3(5, device=device, dtype=dtype)
        skip_if_nan_inputs(X, a)

        result = SE3_AdjTXa_fwd(X, a)
        expected = compute_reference_fp32(X, 'AdjT', a)

        assert result.shape == expected.shape == (5, 6)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SE3_AdjTXa))

    def test_2d_batch(self, device, dtype):
        """Test with 2D batch dimensions."""
        X = pp.randn_SE3(3, 4, device=device, dtype=dtype)
        a = pp.randn_se3(3, 4, device=device, dtype=dtype)
        skip_if_nan_inputs(X, a)

        result = SE3_AdjTXa_fwd(X, a)
        expected = compute_reference_fp32(X, 'AdjT', a)

        assert result.shape == expected.shape == (3, 4, 6)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SE3_AdjTXa))

    def test_3d_batch(self, device, dtype):
        """Test with 3D batch dimensions."""
        X = pp.randn_SE3(2, 3, 4, device=device, dtype=dtype)
        a = pp.randn_se3(2, 3, 4, device=device, dtype=dtype)
        skip_if_nan_inputs(X, a)

        result = SE3_AdjTXa_fwd(X, a)
        expected = compute_reference_fp32(X, 'AdjT', a)

        assert result.shape == expected.shape == (2, 3, 4, 6)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SE3_AdjTXa))

    def test_4d_batch(self, device, dtype):
        """Test with 4D batch dimensions."""
        X = pp.randn_SE3(2, 3, 4, 5, device=device, dtype=dtype)
        a = pp.randn_se3(2, 3, 4, 5, device=device, dtype=dtype)
        skip_if_nan_inputs(X, a)

        result = SE3_AdjTXa_fwd(X, a)
        expected = compute_reference_fp32(X, 'AdjT', a)

        assert result.shape == expected.shape == (2, 3, 4, 5, 6)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SE3_AdjTXa))

    def test_scalar_no_batch(self, device, dtype):
        """Test with no batch dimensions (single transform)."""
        X = pp.randn_SE3(device=device, dtype=dtype)
        a = pp.randn_se3(device=device, dtype=dtype)
        skip_if_nan_inputs(X, a)

        result = SE3_AdjTXa_fwd(X, a)
        expected = compute_reference_fp32(X, 'AdjT', a)

        assert result.shape == expected.shape == (6,)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SE3_AdjTXa))


class TestSE3AdjTXaBroadcasting:
    """Test SE3_AdjTXa_fwd broadcasting behavior."""

    def test_broadcast_1d_to_2d(self, device, dtype):
        """Test broadcasting from 1D to 2D."""
        X = pp.randn_SE3(4, device=device, dtype=dtype)
        a = pp.randn_se3(3, 4, device=device, dtype=dtype)
        skip_if_nan_inputs(X, a)

        result = SE3_AdjTXa_fwd(X, a)
        expected = compute_reference_fp32(X, 'AdjT', a)

        assert result.shape == (3, 4, 6)
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SE3_AdjTXa))

    def test_broadcast_scalar_to_batch(self, device, dtype):
        """Test broadcasting a single transform with batched algebra elements."""
        X = pp.randn_SE3(device=device, dtype=dtype)
        a = pp.randn_se3(5, device=device, dtype=dtype)
        skip_if_nan_inputs(X, a)

        result = SE3_AdjTXa_fwd(X, a)
        expected = compute_reference_fp32(X, 'AdjT', a)

        assert result.shape == (5, 6)
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SE3_AdjTXa))

    def test_broadcast_different_batch_dims(self, device, dtype):
        """Test broadcasting with different batch dimensions."""
        X = pp.randn_SE3(1, 4, device=device, dtype=dtype)
        a = pp.randn_se3(3, 1, device=device, dtype=dtype)
        skip_if_nan_inputs(X, a)

        result = SE3_AdjTXa_fwd(X, a)
        expected = compute_reference_fp32(X, 'AdjT', a)

        assert result.shape == (3, 4, 6)
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SE3_AdjTXa))


class TestSE3AdjTXaPrecision:
    """Test SE3_AdjTXa_fwd precision handling."""

    def test_fp32_precision(self, device):
        """Test float32 precision and accuracy."""
        dtype = torch.float32
        X = pp.randn_SE3(10, device=device, dtype=dtype)
        a = pp.randn_se3(10, device=device, dtype=dtype)

        result = SE3_AdjTXa_fwd(X, a)
        expected = X.AdjT(a)

        torch.testing.assert_close(result, expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_AdjTXa))

    def test_fp64_precision(self, device):
        """Test float64 precision and accuracy."""
        dtype = torch.float64
        X = pp.randn_SE3(10, device=device, dtype=dtype)
        a = pp.randn_se3(10, device=device, dtype=dtype)

        result = SE3_AdjTXa_fwd(X, a)
        expected = X.AdjT(a)

        torch.testing.assert_close(result, expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_AdjTXa))

    def test_fp16_precision(self, device):
        """Test float16 precision and accuracy."""
        dtype = torch.float16
        X = pp.randn_SE3(10, device=device, dtype=dtype)
        a = pp.randn_se3(10, device=device, dtype=dtype)
        skip_if_nan_inputs(X, a)

        result = SE3_AdjTXa_fwd(X, a)
        expected = compute_reference_fp32(X, 'AdjT', a)

        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SE3_AdjTXa))

    def test_output_dtype_preserved(self, device, dtype):
        """Test that output dtype matches input dtype."""
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        a = pp.randn_se3(5, device=device, dtype=dtype)
        skip_if_nan_inputs(X, a)

        result = SE3_AdjTXa_fwd(X, a)

        assert result.dtype == dtype


class TestSE3AdjTXaEdgeCases:
    """Test SE3_AdjTXa_fwd edge cases."""

    def test_identity_transform(self, device):
        """Test transpose adjoint with identity SE3 (should be identity)."""
        dtype = torch.float32
        identity = pp.identity_SE3(5, device=device, dtype=dtype)
        a = pp.randn_se3(5, device=device, dtype=dtype)

        result = SE3_AdjTXa_fwd(identity, a)
        # For identity SE3, Adj^T is also identity, so out = a
        torch.testing.assert_close(result, a.tensor())

    def test_zero_algebra_element(self, device):
        """Test transpose adjoint with zero se3 element."""
        dtype = torch.float32
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        a = pp.identity_se3(5, device=device, dtype=dtype)  # Zero algebra element

        result = SE3_AdjTXa_fwd(X, a)
        expected = torch.zeros_like(result)
        
        torch.testing.assert_close(result, expected)

    def test_single_element_batch(self, device):
        """Test with batch size of 1."""
        dtype = torch.float32
        X = pp.randn_SE3(1, device=device, dtype=dtype)
        a = pp.randn_se3(1, device=device, dtype=dtype)

        result = SE3_AdjTXa_fwd(X, a)
        expected = X.AdjT(a)

        assert result.shape == (1, 6)
        torch.testing.assert_close(result, expected.tensor())

    def test_large_batch(self, device):
        """Test with large batch size."""
        dtype = torch.float32
        X = pp.randn_SE3(1000, device=device, dtype=dtype)
        a = pp.randn_se3(1000, device=device, dtype=dtype)

        result = SE3_AdjTXa_fwd(X, a)
        expected = X.AdjT(a)

        assert result.shape == (1000, 6)
        torch.testing.assert_close(result, expected.tensor())

    def test_output_device(self, device):
        """Test that output is on the same device as input."""
        dtype = torch.float32
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        a = pp.randn_se3(5, device=device, dtype=dtype)

        result = SE3_AdjTXa_fwd(X, a)

        assert result.device == X.device


class TestSE3AdjTXaRelation:
    """Test relationship between AdjT and Adj."""

    def test_adjt_equals_adj_of_inverse(self, device):
        """Test that AdjT(X) @ a = Adj(X^{-1}) @ a."""
        dtype = torch.float32
        X = pp.randn_SE3(10, device=device, dtype=dtype)
        a = pp.randn_se3(10, device=device, dtype=dtype)

        result_adjt = X.AdjT(a)
        X_inv = X.Inv()
        result_adj_inv = X_inv.Adj(a)

        torch.testing.assert_close(result_adjt.tensor(), result_adj_inv.tensor())


class TestSE3AdjTXaErrors:
    """Test SE3_AdjTXa_fwd error handling."""

    def test_5d_batch_raises(self, device):
        """Test that 5D batch dimensions raise NotImplementedError."""
        dtype = torch.float32
        X = pp.randn_SE3(2, 2, 2, 2, 2, device=device, dtype=dtype)
        a = pp.randn_se3(2, 2, 2, 2, 2, device=device, dtype=dtype)

        with pytest.raises(NotImplementedError, match="Batch dimensions > 4"):
            SE3_AdjTXa_fwd(X, a)


# =============================================================================
# Backward Pass Tests
# =============================================================================

class TestSE3AdjTXaBwdBatchDimensions:
    """Test SE3_AdjTXa backward with various batch dimensions."""

    def test_1d_batch(self, device, dtype_bwd):
        """Test backward with 1D batch dimension."""
        X_data = pp.randn_SE3(5, device=device, dtype=dtype_bwd)
        a_data = pp.randn_se3(5, device=device, dtype=dtype_bwd)

        # Our implementation
        X_ours = X_data.clone().requires_grad_(True)
        a_ours = a_data.clone().requires_grad_(True)
        result_ours = SE3_AdjTXa.apply(X_ours, a_ours)
        result_ours.sum().backward()

        # PyPose reference
        X_ref = X_data.clone().requires_grad_(True)
        a_ref = a_data.clone().requires_grad_(True)
        result_ref = X_ref.AdjT(a_ref)
        result_ref.sum().backward()

        assert X_ours.grad.shape == X_ref.grad.shape == (5, 7)
        assert a_ours.grad.shape == a_ref.grad.shape == (5, 6)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AdjTXa))
        torch.testing.assert_close(a_ours.grad, a_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AdjTXa))

    def test_2d_batch(self, device, dtype_bwd):
        """Test backward with 2D batch dimensions."""
        X_data = pp.randn_SE3(3, 4, device=device, dtype=dtype_bwd)
        a_data = pp.randn_se3(3, 4, device=device, dtype=dtype_bwd)

        X_ours = X_data.clone().requires_grad_(True)
        a_ours = a_data.clone().requires_grad_(True)
        result_ours = SE3_AdjTXa.apply(X_ours, a_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        a_ref = a_data.clone().requires_grad_(True)
        result_ref = X_ref.AdjT(a_ref)
        result_ref.sum().backward()

        assert X_ours.grad.shape == X_ref.grad.shape == (3, 4, 7)
        assert a_ours.grad.shape == a_ref.grad.shape == (3, 4, 6)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AdjTXa))
        torch.testing.assert_close(a_ours.grad, a_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AdjTXa))

    def test_3d_batch(self, device, dtype_bwd):
        """Test backward with 3D batch dimensions."""
        X_data = pp.randn_SE3(2, 3, 4, device=device, dtype=dtype_bwd)
        a_data = pp.randn_se3(2, 3, 4, device=device, dtype=dtype_bwd)

        X_ours = X_data.clone().requires_grad_(True)
        a_ours = a_data.clone().requires_grad_(True)
        result_ours = SE3_AdjTXa.apply(X_ours, a_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        a_ref = a_data.clone().requires_grad_(True)
        result_ref = X_ref.AdjT(a_ref)
        result_ref.sum().backward()

        assert X_ours.grad.shape == X_ref.grad.shape == (2, 3, 4, 7)
        assert a_ours.grad.shape == a_ref.grad.shape == (2, 3, 4, 6)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AdjTXa))
        torch.testing.assert_close(a_ours.grad, a_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AdjTXa))

    def test_4d_batch(self, device, dtype_bwd):
        """Test backward with 4D batch dimensions."""
        X_data = pp.randn_SE3(2, 2, 3, 4, device=device, dtype=dtype_bwd)
        a_data = pp.randn_se3(2, 2, 3, 4, device=device, dtype=dtype_bwd)

        X_ours = X_data.clone().requires_grad_(True)
        a_ours = a_data.clone().requires_grad_(True)
        result_ours = SE3_AdjTXa.apply(X_ours, a_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        a_ref = a_data.clone().requires_grad_(True)
        result_ref = X_ref.AdjT(a_ref)
        result_ref.sum().backward()

        assert X_ours.grad.shape == X_ref.grad.shape == (2, 2, 3, 4, 7)
        assert a_ours.grad.shape == a_ref.grad.shape == (2, 2, 3, 4, 6)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AdjTXa))
        torch.testing.assert_close(a_ours.grad, a_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AdjTXa))

    def test_scalar_no_batch(self, device, dtype_bwd):
        """Test backward with no batch dimensions."""
        X_data = pp.randn_SE3(device=device, dtype=dtype_bwd)
        a_data = pp.randn_se3(device=device, dtype=dtype_bwd)

        X_ours = X_data.clone().requires_grad_(True)
        a_ours = a_data.clone().requires_grad_(True)
        result_ours = SE3_AdjTXa.apply(X_ours, a_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        a_ref = a_data.clone().requires_grad_(True)
        result_ref = X_ref.AdjT(a_ref)
        result_ref.sum().backward()

        assert X_ours.grad.shape == X_ref.grad.shape == (7,)
        assert a_ours.grad.shape == a_ref.grad.shape == (6,)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AdjTXa))
        torch.testing.assert_close(a_ours.grad, a_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AdjTXa))


class TestSE3AdjTXaBwdBroadcasting:
    """Test SE3_AdjTXa backward with broadcasting."""

    def test_broadcast_1d_to_2d(self, device, dtype_bwd):
        """Test backward with broadcasting from 1D to 2D."""
        X_data = pp.randn_SE3(4, device=device, dtype=dtype_bwd)
        a_data = pp.randn_se3(3, 4, device=device, dtype=dtype_bwd)

        X_ours = X_data.clone().requires_grad_(True)
        a_ours = a_data.clone().requires_grad_(True)
        result_ours = SE3_AdjTXa.apply(X_ours, a_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        a_ref = a_data.clone().requires_grad_(True)
        result_ref = X_ref.AdjT(a_ref)
        result_ref.sum().backward()

        assert X_ours.grad.shape == X_ref.grad.shape == (4, 7)
        assert a_ours.grad.shape == a_ref.grad.shape == (3, 4, 6)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AdjTXa))
        torch.testing.assert_close(a_ours.grad, a_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AdjTXa))

    def test_broadcast_different_dims(self, device, dtype_bwd):
        """Test backward with broadcasting across different dimensions."""
        X_data = pp.randn_SE3(1, 4, device=device, dtype=dtype_bwd)
        a_data = pp.randn_se3(3, 1, device=device, dtype=dtype_bwd)

        X_ours = X_data.clone().requires_grad_(True)
        a_ours = a_data.clone().requires_grad_(True)
        result_ours = SE3_AdjTXa.apply(X_ours, a_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        a_ref = a_data.clone().requires_grad_(True)
        result_ref = X_ref.AdjT(a_ref)
        result_ref.sum().backward()

        assert X_ours.grad.shape == X_ref.grad.shape == (1, 4, 7)
        assert a_ours.grad.shape == a_ref.grad.shape == (3, 1, 6)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AdjTXa))
        torch.testing.assert_close(a_ours.grad, a_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AdjTXa))


class TestSE3AdjTXaBwdPrecision:
    """Test SE3_AdjTXa backward precision handling."""

    def test_precision(self, device, dtype_bwd):
        """Test backward precision."""
        X_data = pp.randn_SE3(10, device=device, dtype=dtype_bwd)
        a_data = pp.randn_se3(10, device=device, dtype=dtype_bwd)

        X_ours = X_data.clone().requires_grad_(True)
        a_ours = a_data.clone().requires_grad_(True)
        result_ours = SE3_AdjTXa.apply(X_ours, a_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        a_ref = a_data.clone().requires_grad_(True)
        result_ref = X_ref.AdjT(a_ref)
        result_ref.sum().backward()

        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AdjTXa))
        torch.testing.assert_close(a_ours.grad, a_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AdjTXa))

    def test_grad_dtype_preserved(self, device, dtype_bwd):
        """Test that gradient dtype matches input dtype."""
        X = pp.randn_SE3(5, device=device, dtype=dtype_bwd).requires_grad_(True)
        a = pp.randn_se3(5, device=device, dtype=dtype_bwd).requires_grad_(True)

        result = SE3_AdjTXa.apply(X, a)
        result.sum().backward()

        assert X.grad.dtype == dtype_bwd
        assert a.grad.dtype == dtype_bwd


class TestSE3AdjTXaBwdEdgeCases:
    """Test SE3_AdjTXa backward edge cases."""

    def test_quaternion_w_component_zero(self, device, dtype_bwd):
        """Test that quaternion gradient w component is always zero."""
        X = pp.randn_SE3(5, device=device, dtype=dtype_bwd).requires_grad_(True)
        a = pp.randn_se3(5, device=device, dtype=dtype_bwd).requires_grad_(True)

        result = SE3_AdjTXa.apply(X, a)
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
        result_ours = SE3_AdjTXa.apply(X_ours, a_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        a_ref = a_data.clone().requires_grad_(True)
        result_ref = X_ref.AdjT(a_ref)
        result_ref.sum().backward()

        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype, Operator.SE3_AdjTXa))
        torch.testing.assert_close(a_ours.grad, a_ref.grad, **get_bwd_tolerances(dtype, Operator.SE3_AdjTXa))

    def test_large_batch_gradient(self, device):
        """Test gradient with large batch."""
        dtype = torch.float32
        X_data = pp.randn_SE3(1000, device=device, dtype=dtype)
        a_data = pp.randn_se3(1000, device=device, dtype=dtype)

        X_ours = X_data.clone().requires_grad_(True)
        a_ours = a_data.clone().requires_grad_(True)
        result_ours = SE3_AdjTXa.apply(X_ours, a_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        a_ref = a_data.clone().requires_grad_(True)
        result_ref = X_ref.AdjT(a_ref)
        result_ref.sum().backward()

        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype, Operator.SE3_AdjTXa))
        torch.testing.assert_close(a_ours.grad, a_ref.grad, **get_bwd_tolerances(dtype, Operator.SE3_AdjTXa))

    def test_grad_device(self, device):
        """Test that gradients are on the correct device."""
        dtype = torch.float32
        X = pp.randn_SE3(5, device=device, dtype=dtype).requires_grad_(True)
        a = pp.randn_se3(5, device=device, dtype=dtype).requires_grad_(True)

        result = SE3_AdjTXa.apply(X, a)
        result.sum().backward()

        assert X.grad.device == X.device
        assert a.grad.device == a.device


class TestSE3AdjTXaWarpBackend:
    """Test SE3_AdjTXa through the warp_SE3Type backend."""

    def test_warp_backend_integration(self, device, dtype):
        """Test that warp_SE3Type.AdjT works correctly."""
        from pypose_warp import to_warp_backend
        
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        a = pp.randn_se3(5, device=device, dtype=dtype)
        skip_if_nan_inputs(X, a)
        
        X_warp = to_warp_backend(X)

        result = X_warp.AdjT(a)
        expected = compute_reference_fp32(X, 'AdjT', a)

        torch.testing.assert_close(result.tensor(), expected, **get_fwd_tolerances(dtype, Operator.SE3_AdjTXa))

    def test_warp_backend_gradient(self, device, dtype_bwd):
        """Test that warp_SE3Type.AdjT gradients work correctly."""
        from pypose_warp import to_warp_backend
        
        X_data = pp.randn_SE3(5, device=device, dtype=dtype_bwd)
        a_data = pp.randn_se3(5, device=device, dtype=dtype_bwd)

        # Our warp implementation
        X_warp = to_warp_backend(X_data.clone()).requires_grad_(True)
        a_warp = a_data.clone().requires_grad_(True)
        result_ours = X_warp.AdjT(a_warp)
        result_ours.sum().backward()

        # PyPose reference
        X_ref = X_data.clone().requires_grad_(True)
        a_ref = a_data.clone().requires_grad_(True)
        result_ref = X_ref.AdjT(a_ref)
        result_ref.sum().backward()

        torch.testing.assert_close(X_warp.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AdjTXa))
        torch.testing.assert_close(a_warp.grad, a_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AdjTXa))

