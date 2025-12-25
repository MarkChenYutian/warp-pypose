"""Unit tests for SO3_Act forward and backward."""
import pytest
import torch
import pypose as pp

from pypose_warp.ltype.SO3_group import SO3_Act_fwd
from pypose_warp.ltype.SO3_group.Act import SO3_Act
from conftest import get_tolerances


class TestSO3ActBatchDimensions:
    """Test SO3_Act_fwd with various batch dimensions."""

    def test_1d_batch(self, device, dtype):
        """Test with 1D batch dimension."""
        so3 = pp.randn_SO3(5, device=device, dtype=dtype)
        points = torch.randn(5, 3, device=device, dtype=dtype)

        result = SO3_Act_fwd(so3, points)
        expected = so3.Act(points)

        assert result.shape == expected.shape == (5, 3)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_tolerances(dtype))

    def test_2d_batch(self, device, dtype):
        """Test with 2D batch dimensions."""
        so3 = pp.randn_SO3(3, 4, device=device, dtype=dtype)
        points = torch.randn(3, 4, 3, device=device, dtype=dtype)

        result = SO3_Act_fwd(so3, points)
        expected = so3.Act(points)

        assert result.shape == expected.shape == (3, 4, 3)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_tolerances(dtype))

    def test_3d_batch(self, device, dtype):
        """Test with 3D batch dimensions."""
        so3 = pp.randn_SO3(2, 3, 4, device=device, dtype=dtype)
        points = torch.randn(2, 3, 4, 3, device=device, dtype=dtype)

        result = SO3_Act_fwd(so3, points)
        expected = so3.Act(points)

        assert result.shape == expected.shape == (2, 3, 4, 3)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_tolerances(dtype))

    def test_4d_batch(self, device, dtype):
        """Test with 4D batch dimensions."""
        so3 = pp.randn_SO3(2, 3, 4, 5, device=device, dtype=dtype)
        points = torch.randn(2, 3, 4, 5, 3, device=device, dtype=dtype)

        result = SO3_Act_fwd(so3, points)
        expected = so3.Act(points)

        assert result.shape == expected.shape == (2, 3, 4, 5, 3)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_tolerances(dtype))

    def test_scalar_no_batch(self, device, dtype):
        """Test with no batch dimensions (single rotation and point)."""
        so3 = pp.randn_SO3(device=device, dtype=dtype)
        points = torch.randn(3, device=device, dtype=dtype)

        result = SO3_Act_fwd(so3, points)
        expected = so3.Act(points)

        assert result.shape == expected.shape == (3,)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_tolerances(dtype))


class TestSO3ActBroadcasting:
    """Test SO3_Act_fwd broadcasting behavior."""

    def test_broadcast_so3_singleton(self, device):
        """Test broadcasting single SO3 to multiple points."""
        dtype = torch.float32
        so3 = pp.randn_SO3(1, device=device, dtype=dtype)
        points = torch.randn(5, 3, device=device, dtype=dtype)

        result = SO3_Act_fwd(so3, points)
        expected = so3.Act(points)

        assert result.shape == expected.shape == (5, 3)
        torch.testing.assert_close(result, expected)

    def test_broadcast_points_singleton(self, device):
        """Test broadcasting single point to multiple SO3."""
        dtype = torch.float32
        so3 = pp.randn_SO3(5, device=device, dtype=dtype)
        points = torch.randn(1, 3, device=device, dtype=dtype)

        result = SO3_Act_fwd(so3, points)
        expected = so3.Act(points)

        assert result.shape == expected.shape == (5, 3)
        torch.testing.assert_close(result, expected)

    def test_broadcast_2d_cross(self, device):
        """Test 2D cross-broadcasting: (1, 5) SO3 with (4, 1) points."""
        dtype = torch.float32
        so3 = pp.randn_SO3(1, 5, device=device, dtype=dtype)
        points = torch.randn(4, 1, 3, device=device, dtype=dtype)

        result = SO3_Act_fwd(so3, points)
        expected = so3.Act(points)

        assert result.shape == expected.shape == (4, 5, 3)
        torch.testing.assert_close(result, expected)

    def test_broadcast_3d_complex(self, device):
        """Test complex 3D broadcasting: (3, 1, 5) SO3 with (1, 4, 1) points."""
        dtype = torch.float32
        so3 = pp.randn_SO3(3, 1, 5, device=device, dtype=dtype)
        points = torch.randn(1, 4, 1, 3, device=device, dtype=dtype)

        result = SO3_Act_fwd(so3, points)
        expected = so3.Act(points)

        assert result.shape == expected.shape == (3, 4, 5, 3)
        torch.testing.assert_close(result, expected)

    def test_broadcast_4d_complex(self, device):
        """Test complex 4D broadcasting."""
        dtype = torch.float32
        so3 = pp.randn_SO3(2, 1, 3, 1, device=device, dtype=dtype)
        points = torch.randn(1, 4, 1, 5, 3, device=device, dtype=dtype)

        result = SO3_Act_fwd(so3, points)
        expected = so3.Act(points)

        assert result.shape == expected.shape == (2, 4, 3, 5, 3)
        torch.testing.assert_close(result, expected)

    def test_broadcast_different_ndim(self, device):
        """Test broadcasting tensors with different number of dimensions."""
        dtype = torch.float32
        so3 = pp.randn_SO3(5, device=device, dtype=dtype)  # shape (5, 4)
        points = torch.randn(3, 1, 3, device=device, dtype=dtype)  # shape (3, 1, 3)

        result = SO3_Act_fwd(so3, points)
        expected = so3.Act(points)

        assert result.shape == expected.shape == (3, 5, 3)
        torch.testing.assert_close(result, expected)


class TestSO3ActPrecision:
    """Test SO3_Act_fwd precision handling."""

    def test_fp32_precision(self, device):
        """Test float32 precision and accuracy."""
        dtype = torch.float32
        so3 = pp.randn_SO3(10, device=device, dtype=dtype)
        points = torch.randn(10, 3, device=device, dtype=dtype)

        result = SO3_Act_fwd(so3, points)
        expected = so3.Act(points)

        torch.testing.assert_close(result, expected, **get_tolerances(dtype))

    def test_fp64_precision(self, device):
        """Test float64 precision and accuracy."""
        dtype = torch.float64
        so3 = pp.randn_SO3(10, device=device, dtype=dtype)
        points = torch.randn(10, 3, device=device, dtype=dtype)

        result = SO3_Act_fwd(so3, points)
        expected = so3.Act(points)

        torch.testing.assert_close(result, expected, **get_tolerances(dtype))

    def test_fp16_precision(self, device):
        """Test float16 precision and accuracy."""
        dtype = torch.float16
        so3 = pp.randn_SO3(10, device=device, dtype=dtype)
        points = torch.randn(10, 3, device=device, dtype=dtype)

        result = SO3_Act_fwd(so3, points)
        expected = so3.Act(points)

        torch.testing.assert_close(result, expected, **get_tolerances(dtype))

    def test_output_dtype_preserved(self, device, dtype):
        """Test that output dtype matches input dtype."""
        so3 = pp.randn_SO3(5, device=device, dtype=dtype)
        points = torch.randn(5, 3, device=device, dtype=dtype)

        result = SO3_Act_fwd(so3, points)

        assert result.dtype == dtype


class TestSO3ActEdgeCases:
    """Test SO3_Act_fwd edge cases."""

    def test_identity_rotation(self, device):
        """Test that identity rotation leaves points unchanged."""
        dtype = torch.float32
        # Identity quaternion: (x, y, z, w) = (0, 0, 0, 1)
        so3 = pp.identity_SO3(5, device=device, dtype=dtype)
        points = torch.randn(5, 3, device=device, dtype=dtype)

        result = SO3_Act_fwd(so3, points)

        torch.testing.assert_close(result, points)

    def test_single_element_batch(self, device):
        """Test with batch size of 1."""
        dtype = torch.float32
        so3 = pp.randn_SO3(1, device=device, dtype=dtype)
        points = torch.randn(1, 3, device=device, dtype=dtype)

        result = SO3_Act_fwd(so3, points)
        expected = so3.Act(points)

        assert result.shape == (1, 3)
        torch.testing.assert_close(result, expected)

    def test_large_batch(self, device):
        """Test with large batch size."""
        dtype = torch.float32
        so3 = pp.randn_SO3(1000, device=device, dtype=dtype)
        points = torch.randn(1000, 3, device=device, dtype=dtype)

        result = SO3_Act_fwd(so3, points)
        expected = so3.Act(points)

        assert result.shape == (1000, 3)
        torch.testing.assert_close(result, expected)

    def test_output_device(self, device):
        """Test that output is on the same device as input."""
        dtype = torch.float32
        so3 = pp.randn_SO3(5, device=device, dtype=dtype)
        points = torch.randn(5, 3, device=device, dtype=dtype)

        result = SO3_Act_fwd(so3, points)

        assert result.device == points.device == so3.device


class TestSO3ActErrors:
    """Test SO3_Act_fwd error handling."""

    def test_incompatible_shapes_raises(self, device):
        """Test that incompatible shapes raise ValueError."""
        dtype = torch.float32
        so3 = pp.randn_SO3(3, device=device, dtype=dtype)
        points = torch.randn(5, 3, device=device, dtype=dtype)  # Incompatible

        with pytest.raises(ValueError, match="not broadcastable"):
            SO3_Act_fwd(so3, points)

    def test_5d_batch_raises(self, device):
        """Test that 5D batch dimensions raise NotImplementedError."""
        dtype = torch.float32
        so3 = pp.randn_SO3(2, 2, 2, 2, 2, device=device, dtype=dtype)
        points = torch.randn(2, 2, 2, 2, 2, 3, device=device, dtype=dtype)

        with pytest.raises(NotImplementedError, match="Batch dimensions > 4"):
            SO3_Act_fwd(so3, points)


# =============================================================================
# Backward Pass Tests
# =============================================================================

class TestSO3ActBwdBatchDimensions:
    """Test SO3_Act backward with various batch dimensions."""

    def test_1d_batch(self, device, dtype):
        """Test backward with 1D batch dimension."""
        so3_data = pp.randn_SO3(5, device=device, dtype=dtype)
        points_data = torch.randn(5, 3, device=device, dtype=dtype)

        # Our implementation
        so3_ours = so3_data.clone().requires_grad_(True)
        points_ours = points_data.clone().requires_grad_(True)
        result_ours = SO3_Act.apply(so3_ours, points_ours)
        result_ours.sum().backward()

        # PyPose reference
        so3_ref = so3_data.clone().requires_grad_(True)
        points_ref = points_data.clone().requires_grad_(True)
        result_ref = so3_ref.Act(points_ref)
        result_ref.sum().backward()

        assert so3_ours.grad.shape == so3_ref.grad.shape == (5, 4)
        assert points_ours.grad.shape == points_ref.grad.shape == (5, 3)
        torch.testing.assert_close(so3_ours.grad, so3_ref.grad, **get_tolerances(dtype))
        torch.testing.assert_close(points_ours.grad, points_ref.grad, **get_tolerances(dtype))

    def test_2d_batch(self, device, dtype):
        """Test backward with 2D batch dimensions."""
        so3_data = pp.randn_SO3(3, 4, device=device, dtype=dtype)
        points_data = torch.randn(3, 4, 3, device=device, dtype=dtype)

        so3_ours = so3_data.clone().requires_grad_(True)
        points_ours = points_data.clone().requires_grad_(True)
        result_ours = SO3_Act.apply(so3_ours, points_ours)
        result_ours.sum().backward()

        so3_ref = so3_data.clone().requires_grad_(True)
        points_ref = points_data.clone().requires_grad_(True)
        result_ref = so3_ref.Act(points_ref)
        result_ref.sum().backward()

        assert so3_ours.grad.shape == so3_ref.grad.shape == (3, 4, 4)
        assert points_ours.grad.shape == points_ref.grad.shape == (3, 4, 3)
        torch.testing.assert_close(so3_ours.grad, so3_ref.grad, **get_tolerances(dtype))
        torch.testing.assert_close(points_ours.grad, points_ref.grad, **get_tolerances(dtype))

    def test_3d_batch(self, device, dtype):
        """Test backward with 3D batch dimensions."""
        so3_data = pp.randn_SO3(2, 3, 4, device=device, dtype=dtype)
        points_data = torch.randn(2, 3, 4, 3, device=device, dtype=dtype)

        so3_ours = so3_data.clone().requires_grad_(True)
        points_ours = points_data.clone().requires_grad_(True)
        result_ours = SO3_Act.apply(so3_ours, points_ours)
        result_ours.sum().backward()

        so3_ref = so3_data.clone().requires_grad_(True)
        points_ref = points_data.clone().requires_grad_(True)
        result_ref = so3_ref.Act(points_ref)
        result_ref.sum().backward()

        assert so3_ours.grad.shape == so3_ref.grad.shape == (2, 3, 4, 4)
        assert points_ours.grad.shape == points_ref.grad.shape == (2, 3, 4, 3)
        torch.testing.assert_close(so3_ours.grad, so3_ref.grad, **get_tolerances(dtype))
        torch.testing.assert_close(points_ours.grad, points_ref.grad, **get_tolerances(dtype))

    def test_4d_batch(self, device, dtype):
        """Test backward with 4D batch dimensions."""
        so3_data = pp.randn_SO3(2, 2, 3, 4, device=device, dtype=dtype)
        points_data = torch.randn(2, 2, 3, 4, 3, device=device, dtype=dtype)

        so3_ours = so3_data.clone().requires_grad_(True)
        points_ours = points_data.clone().requires_grad_(True)
        result_ours = SO3_Act.apply(so3_ours, points_ours)
        result_ours.sum().backward()

        so3_ref = so3_data.clone().requires_grad_(True)
        points_ref = points_data.clone().requires_grad_(True)
        result_ref = so3_ref.Act(points_ref)
        result_ref.sum().backward()

        assert so3_ours.grad.shape == so3_ref.grad.shape == (2, 2, 3, 4, 4)
        assert points_ours.grad.shape == points_ref.grad.shape == (2, 2, 3, 4, 3)
        torch.testing.assert_close(so3_ours.grad, so3_ref.grad, **get_tolerances(dtype))
        torch.testing.assert_close(points_ours.grad, points_ref.grad, **get_tolerances(dtype))

    def test_scalar_no_batch(self, device, dtype):
        """Test backward with no batch dimensions."""
        so3_data = pp.randn_SO3(device=device, dtype=dtype)
        points_data = torch.randn(3, device=device, dtype=dtype)

        so3_ours = so3_data.clone().requires_grad_(True)
        points_ours = points_data.clone().requires_grad_(True)
        result_ours = SO3_Act.apply(so3_ours, points_ours)
        result_ours.sum().backward()

        so3_ref = so3_data.clone().requires_grad_(True)
        points_ref = points_data.clone().requires_grad_(True)
        result_ref = so3_ref.Act(points_ref)
        result_ref.sum().backward()

        assert so3_ours.grad.shape == so3_ref.grad.shape == (4,)
        assert points_ours.grad.shape == points_ref.grad.shape == (3,)
        torch.testing.assert_close(so3_ours.grad, so3_ref.grad, **get_tolerances(dtype))
        torch.testing.assert_close(points_ours.grad, points_ref.grad, **get_tolerances(dtype))


class TestSO3ActBwdPrecision:
    """Test SO3_Act backward precision handling."""

    def test_fp32_precision(self, device):
        """Test backward float32 precision."""
        dtype = torch.float32
        so3_data = pp.randn_SO3(10, device=device, dtype=dtype)
        points_data = torch.randn(10, 3, device=device, dtype=dtype)

        so3_ours = so3_data.clone().requires_grad_(True)
        points_ours = points_data.clone().requires_grad_(True)
        result_ours = SO3_Act.apply(so3_ours, points_ours)
        result_ours.sum().backward()

        so3_ref = so3_data.clone().requires_grad_(True)
        points_ref = points_data.clone().requires_grad_(True)
        result_ref = so3_ref.Act(points_ref)
        result_ref.sum().backward()

        torch.testing.assert_close(so3_ours.grad, so3_ref.grad, **get_tolerances(dtype))
        torch.testing.assert_close(points_ours.grad, points_ref.grad, **get_tolerances(dtype))

    def test_fp64_precision(self, device):
        """Test backward float64 precision."""
        dtype = torch.float64
        so3_data = pp.randn_SO3(10, device=device, dtype=dtype)
        points_data = torch.randn(10, 3, device=device, dtype=dtype)

        so3_ours = so3_data.clone().requires_grad_(True)
        points_ours = points_data.clone().requires_grad_(True)
        result_ours = SO3_Act.apply(so3_ours, points_ours)
        result_ours.sum().backward()

        so3_ref = so3_data.clone().requires_grad_(True)
        points_ref = points_data.clone().requires_grad_(True)
        result_ref = so3_ref.Act(points_ref)
        result_ref.sum().backward()

        torch.testing.assert_close(so3_ours.grad, so3_ref.grad, **get_tolerances(dtype))
        torch.testing.assert_close(points_ours.grad, points_ref.grad, **get_tolerances(dtype))

    def test_fp16_precision(self, device):
        """Test backward float16 precision."""
        dtype = torch.float16
        so3_data = pp.randn_SO3(10, device=device, dtype=dtype)
        points_data = torch.randn(10, 3, device=device, dtype=dtype)

        so3_ours = so3_data.clone().requires_grad_(True)
        points_ours = points_data.clone().requires_grad_(True)
        result_ours = SO3_Act.apply(so3_ours, points_ours)
        result_ours.sum().backward()

        so3_ref = so3_data.clone().requires_grad_(True)
        points_ref = points_data.clone().requires_grad_(True)
        result_ref = so3_ref.Act(points_ref)
        result_ref.sum().backward()

        torch.testing.assert_close(so3_ours.grad, so3_ref.grad, **get_tolerances(dtype))
        torch.testing.assert_close(points_ours.grad, points_ref.grad, **get_tolerances(dtype))

    def test_grad_dtype_preserved(self, device, dtype):
        """Test that gradient dtype matches input dtype."""
        so3 = pp.randn_SO3(5, device=device, dtype=dtype)
        so3.requires_grad_(True)
        points = torch.randn(5, 3, device=device, dtype=dtype, requires_grad=True)

        result = SO3_Act.apply(so3, points)
        result.sum().backward()

        assert so3.grad.dtype == dtype
        assert points.grad.dtype == dtype


class TestSO3ActBwdEdgeCases:
    """Test SO3_Act backward edge cases."""

    def test_quaternion_w_component_zero(self, device, dtype):
        """Test that quaternion gradient w component is always zero."""
        so3 = pp.randn_SO3(5, device=device, dtype=dtype)
        so3.requires_grad_(True)
        points = torch.randn(5, 3, device=device, dtype=dtype, requires_grad=True)

        result = SO3_Act.apply(so3, points)
        result.sum().backward()

        # The w component of quaternion gradient should always be 0
        assert torch.allclose(so3.grad[..., 3], torch.zeros_like(so3.grad[..., 3]))

    def test_identity_rotation_gradient(self, device):
        """Test gradient through identity rotation."""
        dtype = torch.float32
        so3 = pp.identity_SO3(5, device=device, dtype=dtype)
        so3.requires_grad_(True)
        points = torch.randn(5, 3, device=device, dtype=dtype, requires_grad=True)

        result = SO3_Act.apply(so3, points)
        result.sum().backward()

        # For identity, point gradient should be identity transform (all 1s for sum loss)
        expected_points_grad = torch.ones_like(points)
        torch.testing.assert_close(points.grad, expected_points_grad)

    def test_large_batch_gradient(self, device):
        """Test gradient with large batch."""
        dtype = torch.float32
        so3_data = pp.randn_SO3(1000, device=device, dtype=dtype)
        points_data = torch.randn(1000, 3, device=device, dtype=dtype)

        so3_ours = so3_data.clone().requires_grad_(True)
        points_ours = points_data.clone().requires_grad_(True)
        result_ours = SO3_Act.apply(so3_ours, points_ours)
        result_ours.sum().backward()

        so3_ref = so3_data.clone().requires_grad_(True)
        points_ref = points_data.clone().requires_grad_(True)
        result_ref = so3_ref.Act(points_ref)
        result_ref.sum().backward()

        torch.testing.assert_close(so3_ours.grad, so3_ref.grad, **get_tolerances(dtype))
        torch.testing.assert_close(points_ours.grad, points_ref.grad, **get_tolerances(dtype))

    def test_grad_device(self, device):
        """Test that gradients are on the correct device."""
        dtype = torch.float32
        so3 = pp.randn_SO3(5, device=device, dtype=dtype)
        so3.requires_grad_(True)
        points = torch.randn(5, 3, device=device, dtype=dtype, requires_grad=True)

        result = SO3_Act.apply(so3, points)
        result.sum().backward()

        assert so3.grad.device == so3.device
        assert points.grad.device == points.device
