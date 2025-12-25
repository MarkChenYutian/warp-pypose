"""Unit tests for SO3_Log forward."""
import pytest
import torch
import pypose as pp

from pypose_warp.ltype.SO3_group import SO3_Log_fwd
from pypose_warp.ltype.SO3_group.Log import SO3_Log
from conftest import get_tolerances


class TestSO3LogBatchDimensions:
    """Test SO3_Log_fwd with various batch dimensions."""

    def test_1d_batch(self, device, dtype):
        """Test with 1D batch dimension."""
        so3 = pp.randn_SO3(5, device=device, dtype=dtype)

        result = SO3_Log_fwd(so3)
        expected = so3.Log()

        assert result.shape == expected.shape == (5, 3)
        assert result.ltype == expected.ltype == pp.so3_type
        assert result.dtype == dtype
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_tolerances(dtype))

    def test_2d_batch(self, device, dtype):
        """Test with 2D batch dimensions."""
        so3 = pp.randn_SO3(3, 4, device=device, dtype=dtype)

        result = SO3_Log_fwd(so3)
        expected = so3.Log()

        assert result.shape == expected.shape == (3, 4, 3)
        assert result.ltype == pp.so3_type
        assert result.dtype == dtype
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_tolerances(dtype))

    def test_3d_batch(self, device, dtype):
        """Test with 3D batch dimensions."""
        so3 = pp.randn_SO3(2, 3, 4, device=device, dtype=dtype)

        result = SO3_Log_fwd(so3)
        expected = so3.Log()

        assert result.shape == expected.shape == (2, 3, 4, 3)
        assert result.ltype == pp.so3_type
        assert result.dtype == dtype
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_tolerances(dtype))

    def test_4d_batch(self, device, dtype):
        """Test with 4D batch dimensions."""
        so3 = pp.randn_SO3(2, 3, 4, 5, device=device, dtype=dtype)

        result = SO3_Log_fwd(so3)
        expected = so3.Log()

        assert result.shape == expected.shape == (2, 3, 4, 5, 3)
        assert result.ltype == pp.so3_type
        assert result.dtype == dtype
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_tolerances(dtype))

    def test_scalar_no_batch(self, device, dtype):
        """Test with no batch dimensions (single rotation)."""
        so3 = pp.randn_SO3(device=device, dtype=dtype)

        result = SO3_Log_fwd(so3)
        expected = so3.Log()

        assert result.shape == expected.shape == (3,)
        assert result.ltype == pp.so3_type
        assert result.dtype == dtype
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_tolerances(dtype))


class TestSO3LogPrecision:
    """Test SO3_Log_fwd precision handling."""

    def test_fp32_precision(self, device):
        """Test float32 precision and accuracy."""
        dtype = torch.float32
        so3 = pp.randn_SO3(10, device=device, dtype=dtype)

        result = SO3_Log_fwd(so3)
        expected = so3.Log()

        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_tolerances(dtype))

    def test_fp64_precision(self, device):
        """Test float64 precision and accuracy."""
        dtype = torch.float64
        so3 = pp.randn_SO3(10, device=device, dtype=dtype)

        result = SO3_Log_fwd(so3)
        expected = so3.Log()

        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_tolerances(dtype))

    def test_fp16_precision(self, device):
        """Test float16 precision and accuracy."""
        dtype = torch.float16
        so3 = pp.randn_SO3(10, device=device, dtype=dtype)

        result = SO3_Log_fwd(so3)
        expected = so3.Log()

        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_tolerances(dtype))

    def test_output_dtype_preserved(self, device, dtype):
        """Test that output dtype matches input dtype."""
        so3 = pp.randn_SO3(5, device=device, dtype=dtype)

        result = SO3_Log_fwd(so3)

        assert result.dtype == dtype


class TestSO3LogEdgeCases:
    """Test SO3_Log_fwd edge cases."""

    def test_identity_rotation(self, device):
        """Test that identity rotation maps to zero."""
        dtype = torch.float32
        so3 = pp.identity_SO3(5, device=device, dtype=dtype)

        result = SO3_Log_fwd(so3)

        # Log of identity should be zero vector
        expected = torch.zeros(5, 3, device=device, dtype=dtype)
        torch.testing.assert_close(result.tensor(), expected, atol=1e-6, rtol=1e-6)

    def test_single_element_batch(self, device):
        """Test with batch size of 1."""
        dtype = torch.float32
        so3 = pp.randn_SO3(1, device=device, dtype=dtype)

        result = SO3_Log_fwd(so3)
        expected = so3.Log()

        assert result.shape == (1, 3)
        torch.testing.assert_close(result.tensor(), expected.tensor())

    def test_large_batch(self, device):
        """Test with large batch size."""
        dtype = torch.float32
        so3 = pp.randn_SO3(1000, device=device, dtype=dtype)

        result = SO3_Log_fwd(so3)
        expected = so3.Log()

        assert result.shape == (1000, 3)
        torch.testing.assert_close(result.tensor(), expected.tensor())

    def test_output_device(self, device):
        """Test that output is on the same device as input."""
        dtype = torch.float32
        so3 = pp.randn_SO3(5, device=device, dtype=dtype)

        result = SO3_Log_fwd(so3)

        assert result.device == so3.device

    def test_output_ltype(self, device):
        """Test that output is so3 LieTensor."""
        dtype = torch.float32
        so3 = pp.randn_SO3(5, device=device, dtype=dtype)

        result = SO3_Log_fwd(so3)

        assert isinstance(result, pp.LieTensor)
        assert result.ltype == pp.so3_type


class TestSO3LogErrors:
    """Test SO3_Log_fwd error handling."""

    def test_5d_batch_raises(self, device):
        """Test that 5D batch dimensions raise NotImplementedError."""
        dtype = torch.float32
        so3 = pp.randn_SO3(2, 2, 2, 2, 2, device=device, dtype=dtype)

        with pytest.raises(NotImplementedError, match="Batch dimensions > 4"):
            SO3_Log_fwd(so3)


class TestSO3LogAutogradFunction:
    """Test SO3_Log autograd function wrapper."""

    def test_apply_matches_fwd(self, device, dtype):
        """Test that SO3_Log.apply matches SO3_Log_fwd."""
        so3 = pp.randn_SO3(5, device=device, dtype=dtype)

        result_apply = SO3_Log.apply(so3)
        result_fwd = SO3_Log_fwd(so3)

        torch.testing.assert_close(result_apply.tensor(), result_fwd.tensor())

    def test_apply_1d_batch(self, device, dtype):
        """Test SO3_Log.apply with 1D batch."""
        so3 = pp.randn_SO3(5, device=device, dtype=dtype)

        result = SO3_Log.apply(so3)
        expected = so3.Log()

        assert result.shape == expected.shape == (5, 3)
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_tolerances(dtype))

    def test_apply_2d_batch(self, device, dtype):
        """Test SO3_Log.apply with 2D batch."""
        so3 = pp.randn_SO3(3, 4, device=device, dtype=dtype)

        result = SO3_Log.apply(so3)
        expected = so3.Log()

        assert result.shape == expected.shape == (3, 4, 3)
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_tolerances(dtype))


class TestSO3LogWarpBackend:
    """Test SO3_Log through the warp backend interface."""

    def test_warp_backend_log(self, device, dtype):
        """Test Log through warp backend LieType."""
        from pypose_warp import to_warp_backend

        so3 = pp.randn_SO3(5, device=device, dtype=dtype)
        wp_so3 = to_warp_backend(so3)

        result = wp_so3.Log()
        expected = so3.Log()

        assert result.shape == expected.shape == (5, 3)
        assert result.ltype == pp.so3_type
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_tolerances(dtype))

    def test_warp_backend_log_2d(self, device, dtype):
        """Test Log through warp backend with 2D batch."""
        from pypose_warp import to_warp_backend

        so3 = pp.randn_SO3(3, 4, device=device, dtype=dtype)
        wp_so3 = to_warp_backend(so3)

        result = wp_so3.Log()
        expected = so3.Log()

        assert result.shape == expected.shape == (3, 4, 3)
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_tolerances(dtype))


# =============================================================================
# Backward Pass Tests
# =============================================================================

class TestSO3LogBwdBatchDimensions:
    """Test SO3_Log backward with various batch dimensions."""

    def test_1d_batch(self, device, dtype):
        """Test backward with 1D batch dimension."""
        so3_data = pp.randn_SO3(5, device=device, dtype=dtype)

        so3_ours = so3_data.clone().requires_grad_(True)
        result_ours = SO3_Log.apply(so3_ours)
        result_ours.sum().backward()

        so3_ref = so3_data.clone().requires_grad_(True)
        result_ref = so3_ref.Log()
        result_ref.sum().backward()

        assert so3_ours.grad.shape == so3_ref.grad.shape == (5, 4)
        torch.testing.assert_close(so3_ours.grad, so3_ref.grad, **get_tolerances(dtype))

    def test_2d_batch(self, device, dtype):
        """Test backward with 2D batch dimensions."""
        so3_data = pp.randn_SO3(3, 4, device=device, dtype=dtype)

        so3_ours = so3_data.clone().requires_grad_(True)
        result_ours = SO3_Log.apply(so3_ours)
        result_ours.sum().backward()

        so3_ref = so3_data.clone().requires_grad_(True)
        result_ref = so3_ref.Log()
        result_ref.sum().backward()

        assert so3_ours.grad.shape == so3_ref.grad.shape == (3, 4, 4)
        torch.testing.assert_close(so3_ours.grad, so3_ref.grad, **get_tolerances(dtype))

    def test_3d_batch(self, device, dtype):
        """Test backward with 3D batch dimensions."""
        so3_data = pp.randn_SO3(2, 3, 4, device=device, dtype=dtype)

        so3_ours = so3_data.clone().requires_grad_(True)
        result_ours = SO3_Log.apply(so3_ours)
        result_ours.sum().backward()

        so3_ref = so3_data.clone().requires_grad_(True)
        result_ref = so3_ref.Log()
        result_ref.sum().backward()

        assert so3_ours.grad.shape == so3_ref.grad.shape == (2, 3, 4, 4)
        torch.testing.assert_close(so3_ours.grad, so3_ref.grad, **get_tolerances(dtype))

    def test_4d_batch(self, device, dtype):
        """Test backward with 4D batch dimensions."""
        so3_data = pp.randn_SO3(2, 2, 3, 4, device=device, dtype=dtype)

        so3_ours = so3_data.clone().requires_grad_(True)
        result_ours = SO3_Log.apply(so3_ours)
        result_ours.sum().backward()

        so3_ref = so3_data.clone().requires_grad_(True)
        result_ref = so3_ref.Log()
        result_ref.sum().backward()

        assert so3_ours.grad.shape == so3_ref.grad.shape == (2, 2, 3, 4, 4)
        torch.testing.assert_close(so3_ours.grad, so3_ref.grad, **get_tolerances(dtype))

    def test_scalar_no_batch(self, device, dtype):
        """Test backward with no batch dimensions."""
        so3_data = pp.randn_SO3(device=device, dtype=dtype)

        so3_ours = so3_data.clone().requires_grad_(True)
        result_ours = SO3_Log.apply(so3_ours)
        result_ours.sum().backward()

        so3_ref = so3_data.clone().requires_grad_(True)
        result_ref = so3_ref.Log()
        result_ref.sum().backward()

        assert so3_ours.grad.shape == so3_ref.grad.shape == (4,)
        torch.testing.assert_close(so3_ours.grad, so3_ref.grad, **get_tolerances(dtype))


class TestSO3LogBwdPrecision:
    """Test SO3_Log backward precision handling."""

    def test_fp32_precision(self, device):
        """Test backward float32 precision."""
        dtype = torch.float32
        so3_data = pp.randn_SO3(10, device=device, dtype=dtype)

        so3_ours = so3_data.clone().requires_grad_(True)
        result_ours = SO3_Log.apply(so3_ours)
        result_ours.sum().backward()

        so3_ref = so3_data.clone().requires_grad_(True)
        result_ref = so3_ref.Log()
        result_ref.sum().backward()

        torch.testing.assert_close(so3_ours.grad, so3_ref.grad, **get_tolerances(dtype))

    def test_fp64_precision(self, device):
        """Test backward float64 precision."""
        dtype = torch.float64
        so3_data = pp.randn_SO3(10, device=device, dtype=dtype)

        so3_ours = so3_data.clone().requires_grad_(True)
        result_ours = SO3_Log.apply(so3_ours)
        result_ours.sum().backward()

        so3_ref = so3_data.clone().requires_grad_(True)
        result_ref = so3_ref.Log()
        result_ref.sum().backward()

        torch.testing.assert_close(so3_ours.grad, so3_ref.grad, **get_tolerances(dtype))

    def test_fp16_precision(self, device):
        """Test backward float16 precision."""
        dtype = torch.float16
        so3_data = pp.randn_SO3(10, device=device, dtype=dtype)

        so3_ours = so3_data.clone().requires_grad_(True)
        result_ours = SO3_Log.apply(so3_ours)
        result_ours.sum().backward()

        so3_ref = so3_data.clone().requires_grad_(True)
        result_ref = so3_ref.Log()
        result_ref.sum().backward()

        torch.testing.assert_close(so3_ours.grad, so3_ref.grad, **get_tolerances(dtype))

    def test_grad_dtype_preserved(self, device, dtype):
        """Test that gradient dtype matches input dtype."""
        so3 = pp.randn_SO3(5, device=device, dtype=dtype)
        so3.requires_grad_(True)

        result = SO3_Log.apply(so3)
        result.sum().backward()

        assert so3.grad.dtype == dtype


class TestSO3LogBwdEdgeCases:
    """Test SO3_Log backward edge cases."""

    def test_quaternion_w_component_zero(self, device, dtype):
        """Test that quaternion gradient w component is always zero."""
        so3 = pp.randn_SO3(5, device=device, dtype=dtype)
        so3.requires_grad_(True)

        result = SO3_Log.apply(so3)
        result.sum().backward()

        # The w component of quaternion gradient should always be 0
        assert torch.allclose(so3.grad[..., 3], torch.zeros_like(so3.grad[..., 3]))

    def test_large_batch_gradient(self, device):
        """Test gradient with large batch."""
        dtype = torch.float32
        so3_data = pp.randn_SO3(1000, device=device, dtype=dtype)

        so3_ours = so3_data.clone().requires_grad_(True)
        result_ours = SO3_Log.apply(so3_ours)
        result_ours.sum().backward()

        so3_ref = so3_data.clone().requires_grad_(True)
        result_ref = so3_ref.Log()
        result_ref.sum().backward()

        torch.testing.assert_close(so3_ours.grad, so3_ref.grad, **get_tolerances(dtype))

    def test_grad_device(self, device):
        """Test that gradients are on the correct device."""
        dtype = torch.float32
        so3 = pp.randn_SO3(5, device=device, dtype=dtype)
        so3.requires_grad_(True)

        result = SO3_Log.apply(so3)
        result.sum().backward()

        assert so3.grad.device == so3.device
