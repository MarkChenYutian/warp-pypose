"""Unit tests for SE3_Log forward and backward."""
import pytest
import torch
import pypose as pp
from pypose_warp import to_warp_backend, to_pypose_backend
from pypose_warp.ltype.SE3_group import SE3_Log, SE3_Log_fwd
from conftest import get_fwd_tolerances, get_bwd_tolerances, Operator, skip_if_nan_inputs


# =============================================================================
# Forward Pass Tests
# =============================================================================


class TestSE3LogBatchDimensions:
    """Test SE3_Log_fwd with various batch dimensions."""

    def test_1d_batch(self, device, dtype):
        """Test with 1D batch dimension."""
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        X_warp = to_warp_backend(X)
        
        result = SE3_Log_fwd(X_warp)
        expected = X.Log()
        
        assert result.shape == expected.shape == (5, 6)
        assert to_pypose_backend(result).ltype == expected.ltype == pp.se3_type
        assert result.dtype == dtype
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_Log))

    def test_2d_batch(self, device, dtype):
        """Test with 2D batch dimensions."""
        X = pp.randn_SE3(3, 4, device=device, dtype=dtype)
        skip_if_nan_inputs(X)
        X_warp = to_warp_backend(X)
        
        result = SE3_Log_fwd(X_warp)
        expected = X.Log()
        
        assert result.shape == expected.shape == (3, 4, 6)
        assert to_pypose_backend(result).ltype == pp.se3_type
        assert result.dtype == dtype
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_Log))

    def test_3d_batch(self, device, dtype):
        """Test with 3D batch dimensions."""
        X = pp.randn_SE3(2, 3, 4, device=device, dtype=dtype)
        skip_if_nan_inputs(X)
        X_warp = to_warp_backend(X)
        
        result = SE3_Log_fwd(X_warp)
        expected = X.Log()
        
        assert result.shape == expected.shape == (2, 3, 4, 6)
        assert to_pypose_backend(result).ltype == pp.se3_type
        assert result.dtype == dtype
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_Log))

    def test_4d_batch(self, device, dtype):
        """Test with 4D batch dimensions."""
        X = pp.randn_SE3(2, 2, 3, 4, device=device, dtype=dtype)
        skip_if_nan_inputs(X)
        X_warp = to_warp_backend(X)
        
        result = SE3_Log_fwd(X_warp)
        expected = X.Log()
        
        assert result.shape == expected.shape == (2, 2, 3, 4, 6)
        assert to_pypose_backend(result).ltype == pp.se3_type
        assert result.dtype == dtype
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_Log))

    def test_scalar_no_batch(self, device, dtype):
        """Test with no batch dimensions (single pose)."""
        X = pp.randn_SE3(device=device, dtype=dtype)
        X_warp = to_warp_backend(X)
        
        result = SE3_Log_fwd(X_warp)
        expected = X.Log()
        
        assert result.shape == expected.shape == (6,)
        assert to_pypose_backend(result).ltype == pp.se3_type
        assert result.dtype == dtype
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_Log))


class TestSE3LogPrecision:
    """Test SE3_Log_fwd precision handling."""

    def test_fp32_precision(self, device):
        """Test float32 precision and accuracy."""
        dtype = torch.float32
        X = pp.randn_SE3(10, device=device, dtype=dtype)
        X_warp = to_warp_backend(X)
        
        result = SE3_Log_fwd(X_warp)
        expected = X.Log()
        
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_Log))

    def test_fp64_precision(self, device):
        """Test float64 precision and accuracy."""
        dtype = torch.float64
        X = pp.randn_SE3(10, device=device, dtype=dtype)
        X_warp = to_warp_backend(X)
        
        result = SE3_Log_fwd(X_warp)
        expected = X.Log()
        
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_Log))

    def test_output_dtype_preserved(self, device, dtype):
        """Test that output dtype matches input dtype."""
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        X_warp = to_warp_backend(X)
        
        result = SE3_Log_fwd(X_warp)
        
        assert result.dtype == dtype


class TestSE3LogEdgeCases:
    """Test SE3_Log_fwd edge cases."""

    def test_identity_pose(self, device, dtype):
        """Test that identity pose maps to zero."""
        X = pp.identity_SE3(5, device=device, dtype=dtype)
        X_warp = to_warp_backend(X)
        
        result = SE3_Log_fwd(X_warp)
        
        # Log of identity should be zero vector
        expected = torch.zeros(5, 6, device=device, dtype=dtype)
        torch.testing.assert_close(result.tensor(), expected, atol=1e-5, rtol=1e-5)

    def test_pure_translation(self, device):
        """Test with pure translation (identity rotation)."""
        dtype = torch.float64
        # SE3: [tx, ty, tz, qx, qy, qz, qw] = [1, 2, 3, 0, 0, 0, 1]
        t = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]], device=device, dtype=dtype)
        X = pp.LieTensor(t, ltype=pp.SE3_type)
        X_warp = to_warp_backend(X)
        
        result = SE3_Log_fwd(X_warp)
        expected = X.Log()
        
        torch.testing.assert_close(result.tensor(), expected.tensor(), atol=1e-10, rtol=1e-10)

    def test_pure_rotation(self, device):
        """Test with pure rotation (zero translation)."""
        dtype = torch.float64
        import math
        # 90 degree rotation around z, no translation
        c = math.cos(math.pi / 4)  # cos(45°) for quat
        s = math.sin(math.pi / 4)  # sin(45°) for quat
        t = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, s, c]], device=device, dtype=dtype)
        X = pp.LieTensor(t, ltype=pp.SE3_type)
        X_warp = to_warp_backend(X)
        
        result = SE3_Log_fwd(X_warp)
        expected = X.Log()
        
        torch.testing.assert_close(result.tensor(), expected.tensor(), atol=1e-10, rtol=1e-10)

    def test_single_element_batch(self, device, dtype):
        """Test with batch size of 1."""
        X = pp.randn_SE3(1, device=device, dtype=dtype)
        X_warp = to_warp_backend(X)
        
        result = SE3_Log_fwd(X_warp)
        expected = X.Log()
        
        assert result.shape == (1, 6)
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_Log))

    def test_large_batch(self, device, dtype):
        """Test with large batch size."""
        X = pp.randn_SE3(1000, device=device, dtype=dtype)
        skip_if_nan_inputs(X)  # FP16 can produce NaN from PyPose's randn
        X_warp = to_warp_backend(X)
        
        result = SE3_Log_fwd(X_warp)
        expected = X.Log()
        
        assert result.shape == (1000, 6)
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_Log))

    def test_output_device(self, device, dtype):
        """Test that output is on the same device as input."""
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        X_warp = to_warp_backend(X)
        
        result = SE3_Log_fwd(X_warp)
        
        assert result.device == X.device

    def test_output_ltype(self, device, dtype):
        """Test that output is se3 LieTensor."""
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        X_warp = to_warp_backend(X)
        
        result = SE3_Log_fwd(X_warp)
        
        assert isinstance(result, pp.LieTensor)
        assert to_pypose_backend(result).ltype == pp.se3_type


class TestSE3LogErrors:
    """Test SE3_Log_fwd error handling."""

    def test_5d_batch_raises(self, device, dtype):
        """Test that 5D batch dimensions raise NotImplementedError."""
        X = pp.randn_SE3(2, 2, 2, 2, 2, device=device, dtype=dtype)
        X_warp = to_warp_backend(X)
        
        with pytest.raises(NotImplementedError, match="Batch dimensions > 4"):
            SE3_Log_fwd(X_warp)


class TestSE3LogAutogradFunction:
    """Test SE3_Log autograd function wrapper."""

    def test_apply_matches_fwd(self, device, dtype):
        """Test that SE3_Log.apply matches SE3_Log_fwd."""
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        X_warp = to_warp_backend(X)
        
        result_apply = SE3_Log.apply(X_warp)
        result_fwd = SE3_Log_fwd(X_warp)
        
        torch.testing.assert_close(result_apply.tensor(), result_fwd.tensor())

    def test_apply_1d_batch(self, device, dtype):
        """Test SE3_Log.apply with 1D batch."""
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        X_warp = to_warp_backend(X)
        
        result = SE3_Log.apply(X_warp)
        expected = X.Log()
        
        assert result.shape == expected.shape == (5, 6)
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_Log))

    def test_apply_2d_batch(self, device, dtype):
        """Test SE3_Log.apply with 2D batch."""
        X = pp.randn_SE3(3, 4, device=device, dtype=dtype)
        X_warp = to_warp_backend(X)
        
        result = SE3_Log.apply(X_warp)
        expected = X.Log()
        
        assert result.shape == expected.shape == (3, 4, 6)
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_Log))


class TestSE3LogWarpBackend:
    """Test SE3_Log through the warp backend interface."""

    def test_warp_backend_log(self, device, dtype):
        """Test Log through warp backend LieType."""
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        skip_if_nan_inputs(X)  # FP16 can produce NaN from PyPose's randn
        X_warp = to_warp_backend(X)
        
        result = X_warp.Log()
        expected = X.Log()
        
        assert result.shape == expected.shape == (5, 6)
        assert to_pypose_backend(result).ltype == pp.se3_type
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_Log))

    def test_warp_backend_log_2d(self, device, dtype):
        """Test Log through warp backend with 2D batch."""
        X = pp.randn_SE3(3, 4, device=device, dtype=dtype)
        skip_if_nan_inputs(X)
        X_warp = to_warp_backend(X)
        
        result = X_warp.Log()
        expected = X.Log()
        
        assert result.shape == expected.shape == (3, 4, 6)
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_Log))


# =============================================================================
# Backward Pass Tests
# =============================================================================


class TestSE3LogBwdBatchDimensions:
    """Test SE3_Log backward with various batch dimensions."""

    def test_1d_batch(self, device, dtype_bwd):
        """Test backward with 1D batch dimension."""
        X_data = pp.randn_SE3(5, device=device, dtype=dtype_bwd)
        
        X_ours = to_warp_backend(X_data).requires_grad_(True)
        result_ours = SE3_Log.apply(X_ours)
        result_ours.sum().backward()
        
        X_ref = X_data.clone().requires_grad_(True)
        result_ref = X_ref.Log()
        result_ref.sum().backward()
        
        assert X_ours.grad.shape == X_ref.grad.shape == (5, 7)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Log))

    def test_2d_batch(self, device, dtype_bwd):
        """Test backward with 2D batch dimensions."""
        X_data = pp.randn_SE3(3, 4, device=device, dtype=dtype_bwd)
        
        X_ours = to_warp_backend(X_data).requires_grad_(True)
        result_ours = SE3_Log.apply(X_ours)
        result_ours.sum().backward()
        
        X_ref = X_data.clone().requires_grad_(True)
        result_ref = X_ref.Log()
        result_ref.sum().backward()
        
        assert X_ours.grad.shape == X_ref.grad.shape == (3, 4, 7)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Log))

    def test_3d_batch(self, device, dtype_bwd):
        """Test backward with 3D batch dimensions."""
        X_data = pp.randn_SE3(2, 3, 4, device=device, dtype=dtype_bwd)
        
        X_ours = to_warp_backend(X_data).requires_grad_(True)
        result_ours = SE3_Log.apply(X_ours)
        result_ours.sum().backward()
        
        X_ref = X_data.clone().requires_grad_(True)
        result_ref = X_ref.Log()
        result_ref.sum().backward()
        
        assert X_ours.grad.shape == X_ref.grad.shape == (2, 3, 4, 7)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Log))

    def test_4d_batch(self, device, dtype_bwd):
        """Test backward with 4D batch dimensions."""
        X_data = pp.randn_SE3(2, 2, 3, 4, device=device, dtype=dtype_bwd)
        
        X_ours = to_warp_backend(X_data).requires_grad_(True)
        result_ours = SE3_Log.apply(X_ours)
        result_ours.sum().backward()
        
        X_ref = X_data.clone().requires_grad_(True)
        result_ref = X_ref.Log()
        result_ref.sum().backward()
        
        assert X_ours.grad.shape == X_ref.grad.shape == (2, 2, 3, 4, 7)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Log))

    def test_scalar_no_batch(self, device, dtype_bwd):
        """Test backward with no batch dimensions."""
        X_data = pp.randn_SE3(device=device, dtype=dtype_bwd)
        
        X_ours = to_warp_backend(X_data).requires_grad_(True)
        result_ours = SE3_Log.apply(X_ours)
        result_ours.sum().backward()
        
        X_ref = X_data.clone().requires_grad_(True)
        result_ref = X_ref.Log()
        result_ref.sum().backward()
        
        assert X_ours.grad.shape == X_ref.grad.shape == (7,)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Log))


class TestSE3LogBwdPrecision:
    """Test SE3_Log backward precision handling."""

    def test_precision(self, device, dtype_bwd):
        """Test backward precision."""
        X_data = pp.randn_SE3(10, device=device, dtype=dtype_bwd)
        
        X_ours = to_warp_backend(X_data).requires_grad_(True)
        result_ours = SE3_Log.apply(X_ours)
        result_ours.sum().backward()
        
        X_ref = X_data.clone().requires_grad_(True)
        result_ref = X_ref.Log()
        result_ref.sum().backward()
        
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Log))

    def test_grad_dtype_preserved(self, device, dtype_bwd):
        """Test that gradient dtype matches input dtype."""
        X = pp.randn_SE3(5, device=device, dtype=dtype_bwd)
        X_warp = to_warp_backend(X).requires_grad_(True)
        
        result = SE3_Log.apply(X_warp)
        result.sum().backward()
        
        assert X_warp.grad.dtype == dtype_bwd


class TestSE3LogBwdEdgeCases:
    """Test SE3_Log backward edge cases."""

    def test_quaternion_w_component_zero(self, device, dtype_bwd):
        """Test that quaternion gradient w component is always zero."""
        X = pp.randn_SE3(5, device=device, dtype=dtype_bwd)
        X_warp = to_warp_backend(X).requires_grad_(True)
        
        result = SE3_Log.apply(X_warp)
        result.sum().backward()
        
        # The w component of quaternion gradient (index 6) should always be 0
        assert torch.allclose(X_warp.grad[..., 6], torch.zeros_like(X_warp.grad[..., 6]))

    def test_large_batch_gradient(self, device, dtype_bwd):
        """Test gradient with large batch."""
        X_data = pp.randn_SE3(1000, device=device, dtype=dtype_bwd)
        
        X_ours = to_warp_backend(X_data).requires_grad_(True)
        result_ours = SE3_Log.apply(X_ours)
        result_ours.sum().backward()
        
        X_ref = X_data.clone().requires_grad_(True)
        result_ref = X_ref.Log()
        result_ref.sum().backward()
        
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Log))

    def test_grad_device(self, device, dtype_bwd):
        """Test that gradients are on the correct device."""
        X = pp.randn_SE3(5, device=device, dtype=dtype_bwd)
        X_warp = to_warp_backend(X).requires_grad_(True)
        
        result = SE3_Log.apply(X_warp)
        result.sum().backward()
        
        assert X_warp.grad.device == X_warp.device

    def test_identity_gradient(self, device, dtype_bwd):
        """Test backward with identity pose."""
        X_data = pp.identity_SE3(5, device=device, dtype=dtype_bwd)
        
        X_ours = to_warp_backend(X_data).requires_grad_(True)
        result_ours = SE3_Log.apply(X_ours)
        result_ours.sum().backward()
        
        X_ref = X_data.clone().requires_grad_(True)
        result_ref = X_ref.Log()
        result_ref.sum().backward()
        
        # Gradients should be finite
        assert not torch.isnan(X_ours.grad).any()
        assert not torch.isinf(X_ours.grad).any()
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Log))

