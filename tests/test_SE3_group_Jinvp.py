"""Unit tests for SE3_Jinvp forward and backward."""
import pytest
import torch
import pypose as pp
from pypose.lietensor.operation import se3_Jl_inv, SE3_Log
from pypose_warp.ltype.SE3_group import SE3_Jinvp, SE3_Jinvp_fwd
from conftest import get_fwd_tolerances, get_bwd_tolerances, Operator, skip_if_nan_inputs, compute_reference_fp32


def pypose_jinvp(X, p):
    """Reference implementation using PyPose operations.
    
    For FP16, computes in FP32 then downcasts to avoid PyPose numerical instability.
    """
    X_tensor = X.tensor() if isinstance(X, pp.LieTensor) else X
    p_tensor = p.tensor() if isinstance(p, pp.LieTensor) else p
    original_dtype = X_tensor.dtype
    
    if original_dtype == torch.float16:
        # Compute in FP32 for numerical stability
        X_fp32 = X_tensor.float()
        p_fp32 = p_tensor.float()
        result_fp32 = (se3_Jl_inv(SE3_Log.apply(X_fp32)) @ p_fp32.unsqueeze(-1)).squeeze(-1)
        return result_fp32.half()
    else:
        return (se3_Jl_inv(SE3_Log.apply(X_tensor)) @ p_tensor.unsqueeze(-1)).squeeze(-1)


# =============================================================================
# Forward Pass Tests
# =============================================================================


class TestSE3JinvpFwdBatchDimensions:
    """Test SE3_Jinvp_fwd with various batch dimensions."""

    def test_1d_batch(self, device, dtype):
        """Test with 1D batch dimension."""
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        p = pp.randn_se3(5, device=device, dtype=dtype)
        skip_if_nan_inputs(X, p)

        result = SE3_Jinvp_fwd(X, p)
        expected = pypose_jinvp(X, p)

        assert result.shape == expected.shape == (5, 6)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SE3_Jinvp))

    def test_2d_batch(self, device, dtype):
        """Test with 2D batch dimensions."""
        X = pp.randn_SE3(3, 4, device=device, dtype=dtype)
        p = pp.randn_se3(3, 4, device=device, dtype=dtype)
        skip_if_nan_inputs(X, p)

        result = SE3_Jinvp_fwd(X, p)
        expected = pypose_jinvp(X, p)

        assert result.shape == expected.shape == (3, 4, 6)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SE3_Jinvp))

    def test_3d_batch(self, device, dtype):
        """Test with 3D batch dimensions."""
        X = pp.randn_SE3(2, 3, 4, device=device, dtype=dtype)
        p = pp.randn_se3(2, 3, 4, device=device, dtype=dtype)
        skip_if_nan_inputs(X, p)

        result = SE3_Jinvp_fwd(X, p)
        expected = pypose_jinvp(X, p)

        assert result.shape == expected.shape == (2, 3, 4, 6)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SE3_Jinvp))

    def test_4d_batch(self, device, dtype):
        """Test with 4D batch dimensions."""
        X = pp.randn_SE3(2, 3, 4, 5, device=device, dtype=dtype)
        p = pp.randn_se3(2, 3, 4, 5, device=device, dtype=dtype)
        skip_if_nan_inputs(X, p)

        result = SE3_Jinvp_fwd(X, p)
        expected = pypose_jinvp(X, p)

        assert result.shape == expected.shape == (2, 3, 4, 5, 6)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SE3_Jinvp))

    def test_scalar_no_batch(self, device, dtype):
        """Test with no batch dimensions (single transform)."""
        X = pp.randn_SE3(device=device, dtype=dtype)
        p = pp.randn_se3(device=device, dtype=dtype)
        skip_if_nan_inputs(X, p)

        result = SE3_Jinvp_fwd(X, p)
        expected = pypose_jinvp(X, p)

        assert result.shape == expected.shape == (6,)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SE3_Jinvp))


class TestSE3JinvpFwdBroadcasting:
    """Test SE3_Jinvp_fwd broadcasting behavior."""

    def test_broadcast_1d_to_2d(self, device, dtype):
        """Test broadcasting from 1D to 2D."""
        X = pp.randn_SE3(4, device=device, dtype=dtype)
        p = pp.randn_se3(3, 4, device=device, dtype=dtype)
        skip_if_nan_inputs(X, p)

        result = SE3_Jinvp_fwd(X, p)
        expected = pypose_jinvp(X, p)

        assert result.shape == (3, 4, 6)
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SE3_Jinvp))

    def test_broadcast_scalar_to_batch(self, device, dtype):
        """Test broadcasting a single transform with batched tangent vectors."""
        X = pp.randn_SE3(device=device, dtype=dtype)
        p = pp.randn_se3(5, device=device, dtype=dtype)
        skip_if_nan_inputs(X, p)

        result = SE3_Jinvp_fwd(X, p)
        expected = pypose_jinvp(X, p)

        assert result.shape == (5, 6)
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SE3_Jinvp))

    def test_broadcast_different_batch_dims(self, device, dtype):
        """Test broadcasting with different batch dimensions."""
        X = pp.randn_SE3(1, 4, device=device, dtype=dtype)
        p = pp.randn_se3(3, 1, device=device, dtype=dtype)
        skip_if_nan_inputs(X, p)

        result = SE3_Jinvp_fwd(X, p)
        expected = pypose_jinvp(X, p)

        assert result.shape == (3, 4, 6)
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SE3_Jinvp))


class TestSE3JinvpFwdPrecision:
    """Test SE3_Jinvp_fwd precision handling."""

    def test_fp32_precision(self, device):
        """Test float32 precision and accuracy."""
        dtype = torch.float32
        X = pp.randn_SE3(10, device=device, dtype=dtype)
        p = pp.randn_se3(10, device=device, dtype=dtype)

        result = SE3_Jinvp_fwd(X, p)
        expected = pypose_jinvp(X, p)

        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SE3_Jinvp))

    def test_fp64_precision(self, device):
        """Test float64 precision and accuracy."""
        dtype = torch.float64
        X = pp.randn_SE3(10, device=device, dtype=dtype)
        p = pp.randn_se3(10, device=device, dtype=dtype)

        result = SE3_Jinvp_fwd(X, p)
        expected = pypose_jinvp(X, p)

        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SE3_Jinvp))

    def test_fp16_precision(self, device):
        """Test float16 precision and accuracy."""
        dtype = torch.float16
        X = pp.randn_SE3(10, device=device, dtype=dtype)
        p = pp.randn_se3(10, device=device, dtype=dtype)
        skip_if_nan_inputs(X, p)

        result = SE3_Jinvp_fwd(X, p)
        expected = pypose_jinvp(X, p)

        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SE3_Jinvp))

    def test_output_dtype_preserved(self, device, dtype):
        """Test that output dtype matches input dtype."""
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        p = pp.randn_se3(5, device=device, dtype=dtype)
        skip_if_nan_inputs(X, p)

        result = SE3_Jinvp_fwd(X, p)

        assert result.dtype == dtype


class TestSE3JinvpFwdEdgeCases:
    """Test SE3_Jinvp_fwd edge cases."""

    def test_identity_transform(self, device):
        """Test Jinvp with identity SE3."""
        dtype = torch.float32
        identity = pp.identity_SE3(5, device=device, dtype=dtype)
        p = pp.randn_se3(5, device=device, dtype=dtype)

        result = SE3_Jinvp_fwd(identity, p)
        expected = pypose_jinvp(identity, p)
        
        torch.testing.assert_close(result, expected)

    def test_zero_tangent(self, device):
        """Test Jinvp with zero tangent vector."""
        dtype = torch.float32
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        p = pp.identity_se3(5, device=device, dtype=dtype)  # Zero tangent

        result = SE3_Jinvp_fwd(X, p)
        expected = torch.zeros_like(result)
        
        torch.testing.assert_close(result, expected)

    def test_single_element_batch(self, device):
        """Test with batch size of 1."""
        dtype = torch.float32
        X = pp.randn_SE3(1, device=device, dtype=dtype)
        p = pp.randn_se3(1, device=device, dtype=dtype)

        result = SE3_Jinvp_fwd(X, p)
        expected = pypose_jinvp(X, p)

        assert result.shape == (1, 6)
        torch.testing.assert_close(result, expected)

    def test_large_batch(self, device):
        """Test with large batch size."""
        dtype = torch.float32
        X = pp.randn_SE3(1000, device=device, dtype=dtype)
        p = pp.randn_se3(1000, device=device, dtype=dtype)

        result = SE3_Jinvp_fwd(X, p)
        expected = pypose_jinvp(X, p)

        assert result.shape == (1000, 6)
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SE3_Jinvp))

    def test_output_device(self, device):
        """Test that output is on the same device as input."""
        dtype = torch.float32
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        p = pp.randn_se3(5, device=device, dtype=dtype)

        result = SE3_Jinvp_fwd(X, p)

        assert result.device == X.device


class TestSE3JinvpFwdErrors:
    """Test SE3_Jinvp_fwd error handling."""

    def test_5d_batch_raises(self, device):
        """Test that 5D batch dimensions raise NotImplementedError."""
        dtype = torch.float32
        X = pp.randn_SE3(2, 2, 2, 2, 2, device=device, dtype=dtype)
        p = pp.randn_se3(2, 2, 2, 2, 2, device=device, dtype=dtype)

        with pytest.raises(NotImplementedError, match="Batch dimensions > 4"):
            SE3_Jinvp_fwd(X, p)


# =============================================================================
# Backward Pass Tests
# =============================================================================


class TestSE3JinvpBwdBatchDimensions:
    """Test SE3_Jinvp backward with various batch dimensions."""

    def test_1d_batch(self, device, dtype_bwd):
        """Test backward with 1D batch dimension."""
        X_data = pp.randn_SE3(5, device=device, dtype=dtype_bwd)
        p_data = pp.randn_se3(5, device=device, dtype=dtype_bwd)

        # Our implementation
        X_ours = X_data.clone().requires_grad_(True)
        p_ours = p_data.clone().requires_grad_(True)
        result_ours = SE3_Jinvp.apply(X_ours, p_ours)
        result_ours.sum().backward()

        # PyPose reference
        X_ref = X_data.clone().requires_grad_(True)
        p_ref = p_data.clone().requires_grad_(True)
        result_ref = X_ref.Jinvp(p_ref)
        result_ref.sum().backward()

        assert X_ours.grad.shape == X_ref.grad.shape == (5, 7)
        assert p_ours.grad.shape == p_ref.grad.shape == (5, 6)
        # Backward is approximate, so use looser tolerances
        torch.testing.assert_close(p_ours.grad, p_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Jinvp))

    def test_2d_batch(self, device, dtype_bwd):
        """Test backward with 2D batch dimensions."""
        X_data = pp.randn_SE3(3, 4, device=device, dtype=dtype_bwd)
        p_data = pp.randn_se3(3, 4, device=device, dtype=dtype_bwd)

        X_ours = X_data.clone().requires_grad_(True)
        p_ours = p_data.clone().requires_grad_(True)
        result_ours = SE3_Jinvp.apply(X_ours, p_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        p_ref = p_data.clone().requires_grad_(True)
        result_ref = X_ref.Jinvp(p_ref)
        result_ref.sum().backward()

        assert X_ours.grad.shape == X_ref.grad.shape == (3, 4, 7)
        assert p_ours.grad.shape == p_ref.grad.shape == (3, 4, 6)
        torch.testing.assert_close(p_ours.grad, p_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Jinvp))

    def test_scalar_no_batch(self, device, dtype_bwd):
        """Test backward with no batch dimensions."""
        X_data = pp.randn_SE3(device=device, dtype=dtype_bwd)
        p_data = pp.randn_se3(device=device, dtype=dtype_bwd)

        X_ours = X_data.clone().requires_grad_(True)
        p_ours = p_data.clone().requires_grad_(True)
        result_ours = SE3_Jinvp.apply(X_ours, p_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        p_ref = p_data.clone().requires_grad_(True)
        result_ref = X_ref.Jinvp(p_ref)
        result_ref.sum().backward()

        assert X_ours.grad.shape == X_ref.grad.shape == (7,)
        assert p_ours.grad.shape == p_ref.grad.shape == (6,)
        torch.testing.assert_close(p_ours.grad, p_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Jinvp))


class TestSE3JinvpBwdPrecision:
    """Test SE3_Jinvp backward precision handling."""

    def test_precision(self, device, dtype_bwd):
        """Test backward precision."""
        X_data = pp.randn_SE3(10, device=device, dtype=dtype_bwd)
        p_data = pp.randn_se3(10, device=device, dtype=dtype_bwd)

        X_ours = X_data.clone().requires_grad_(True)
        p_ours = p_data.clone().requires_grad_(True)
        result_ours = SE3_Jinvp.apply(X_ours, p_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        p_ref = p_data.clone().requires_grad_(True)
        result_ref = X_ref.Jinvp(p_ref)
        result_ref.sum().backward()

        torch.testing.assert_close(p_ours.grad, p_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Jinvp))

    def test_grad_dtype_preserved(self, device, dtype_bwd):
        """Test that gradient dtype matches input dtype."""
        X = pp.randn_SE3(5, device=device, dtype=dtype_bwd).requires_grad_(True)
        p = pp.randn_se3(5, device=device, dtype=dtype_bwd).requires_grad_(True)

        result = SE3_Jinvp.apply(X, p)
        result.sum().backward()

        assert X.grad.dtype == dtype_bwd
        assert p.grad.dtype == dtype_bwd


class TestSE3JinvpBwdEdgeCases:
    """Test SE3_Jinvp backward edge cases."""

    def test_quaternion_w_component_zero(self, device, dtype_bwd):
        """Test that quaternion gradient w component is always zero."""
        X = pp.randn_SE3(5, device=device, dtype=dtype_bwd).requires_grad_(True)
        p = pp.randn_se3(5, device=device, dtype=dtype_bwd).requires_grad_(True)

        result = SE3_Jinvp.apply(X, p)
        result.sum().backward()

        # The w component of quaternion gradient should always be 0
        assert torch.allclose(X.grad[..., 6], torch.zeros_like(X.grad[..., 6]))

    def test_identity_transform_gradient(self, device):
        """Test gradient through identity transformation."""
        dtype = torch.float32
        X_data = pp.identity_SE3(5, device=device, dtype=dtype)
        p_data = pp.randn_se3(5, device=device, dtype=dtype)

        X_ours = X_data.clone().requires_grad_(True)
        p_ours = p_data.clone().requires_grad_(True)
        result_ours = SE3_Jinvp.apply(X_ours, p_ours)
        result_ours.sum().backward()

        X_ref = X_data.clone().requires_grad_(True)
        p_ref = p_data.clone().requires_grad_(True)
        result_ref = X_ref.Jinvp(p_ref)
        result_ref.sum().backward()

        torch.testing.assert_close(p_ours.grad, p_ref.grad, **get_bwd_tolerances(dtype, Operator.SE3_Jinvp))

    def test_grad_device(self, device):
        """Test that gradients are on the correct device."""
        dtype = torch.float32
        X = pp.randn_SE3(5, device=device, dtype=dtype).requires_grad_(True)
        p = pp.randn_se3(5, device=device, dtype=dtype).requires_grad_(True)

        result = SE3_Jinvp.apply(X, p)
        result.sum().backward()

        assert X.grad.device == X.device
        assert p.grad.device == p.device


class TestSE3JinvpWarpBackend:
    """Test SE3_Jinvp through the warp_SE3Type backend."""

    def test_warp_backend_integration(self, device, dtype):
        """Test that warp_SE3Type.Jinvp works correctly."""
        from pypose_warp import to_warp_backend
        
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        p = pp.randn_se3(5, device=device, dtype=dtype)
        skip_if_nan_inputs(X, p)
        
        X_warp = to_warp_backend(X)

        result = X_warp.Jinvp(p)
        # Use FP32 reference for FP16 to avoid PyPose numerical instability
        expected = compute_reference_fp32(X, 'Jinvp', p)

        torch.testing.assert_close(result.tensor(), expected, **get_fwd_tolerances(dtype, Operator.SE3_Jinvp))

    def test_warp_backend_gradient(self, device, dtype_bwd):
        """Test that warp_SE3Type.Jinvp gradients work correctly."""
        from pypose_warp import to_warp_backend
        
        X_data = pp.randn_SE3(5, device=device, dtype=dtype_bwd)
        p_data = pp.randn_se3(5, device=device, dtype=dtype_bwd)

        # Our warp implementation
        X_warp = to_warp_backend(X_data.clone()).requires_grad_(True)
        p_warp = p_data.clone().requires_grad_(True)
        result_ours = X_warp.Jinvp(p_warp)
        result_ours.sum().backward()

        # PyPose reference
        X_ref = X_data.clone().requires_grad_(True)
        p_ref = p_data.clone().requires_grad_(True)
        result_ref = X_ref.Jinvp(p_ref)
        result_ref.sum().backward()

        torch.testing.assert_close(p_warp.grad, p_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Jinvp))

