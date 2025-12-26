"""Unit tests for SE3_Act4 forward and backward (4D homogeneous points)."""
import pytest
import torch
import pypose as pp
from pypose_warp.ltype.SE3_group import SE3_Act4, SE3_Act4_fwd
from conftest import get_fwd_tolerances, get_bwd_tolerances, Operator


class TestSE3Act4BatchDimensions:
    """Test SE3_Act4_fwd with various batch dimensions."""

    def test_1d_batch(self, device, dtype):
        """Test with 1D batch dimension."""
        se3 = pp.randn_SE3(5, device=device, dtype=dtype)
        points = torch.randn(5, 4, device=device, dtype=dtype)

        result = SE3_Act4_fwd(se3, points)
        expected = se3.Act(points)

        assert result.shape == expected.shape == (5, 4)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SE3_Act4))

    def test_2d_batch(self, device, dtype):
        """Test with 2D batch dimensions."""
        se3 = pp.randn_SE3(3, 4, device=device, dtype=dtype)
        points = torch.randn(3, 4, 4, device=device, dtype=dtype)

        result = SE3_Act4_fwd(se3, points)
        expected = se3.Act(points)

        assert result.shape == expected.shape == (3, 4, 4)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SE3_Act4))

    def test_3d_batch(self, device, dtype):
        """Test with 3D batch dimensions."""
        se3 = pp.randn_SE3(2, 3, 4, device=device, dtype=dtype)
        points = torch.randn(2, 3, 4, 4, device=device, dtype=dtype)

        result = SE3_Act4_fwd(se3, points)
        expected = se3.Act(points)

        assert result.shape == expected.shape == (2, 3, 4, 4)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SE3_Act4))

    def test_4d_batch(self, device, dtype):
        """Test with 4D batch dimensions."""
        se3 = pp.randn_SE3(2, 3, 4, 5, device=device, dtype=dtype)
        points = torch.randn(2, 3, 4, 5, 4, device=device, dtype=dtype)

        result = SE3_Act4_fwd(se3, points)
        expected = se3.Act(points)

        assert result.shape == expected.shape == (2, 3, 4, 5, 4)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SE3_Act4))

    def test_scalar_no_batch(self, device, dtype):
        """Test with no batch dimensions (single transform and point)."""
        se3 = pp.randn_SE3(device=device, dtype=dtype)
        points = torch.randn(4, device=device, dtype=dtype)

        result = SE3_Act4_fwd(se3, points)
        expected = se3.Act(points)

        assert result.shape == expected.shape == (4,)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SE3_Act4))


class TestSE3Act4Broadcasting:
    """Test SE3_Act4_fwd broadcasting behavior."""

    def test_broadcast_se3_singleton(self, device):
        """Test broadcasting single SE3 to multiple points."""
        dtype = torch.float32
        se3 = pp.randn_SE3(1, device=device, dtype=dtype)
        points = torch.randn(5, 4, device=device, dtype=dtype)

        result = SE3_Act4_fwd(se3, points)
        expected = se3.Act(points)

        assert result.shape == expected.shape == (5, 4)
        torch.testing.assert_close(result, expected)

    def test_broadcast_points_singleton(self, device):
        """Test broadcasting single point to multiple SE3."""
        dtype = torch.float32
        se3 = pp.randn_SE3(5, device=device, dtype=dtype)
        points = torch.randn(1, 4, device=device, dtype=dtype)

        result = SE3_Act4_fwd(se3, points)
        expected = se3.Act(points)

        assert result.shape == expected.shape == (5, 4)
        torch.testing.assert_close(result, expected)

    def test_broadcast_2d_cross(self, device):
        """Test 2D cross-broadcasting: (1, 5) SE3 with (4, 1) points."""
        dtype = torch.float32
        se3 = pp.randn_SE3(1, 5, device=device, dtype=dtype)
        points = torch.randn(4, 1, 4, device=device, dtype=dtype)

        result = SE3_Act4_fwd(se3, points)
        expected = se3.Act(points)

        assert result.shape == expected.shape == (4, 5, 4)
        torch.testing.assert_close(result, expected)

    def test_broadcast_3d_complex(self, device):
        """Test complex 3D broadcasting: (3, 1, 5) SE3 with (1, 4, 1) points."""
        dtype = torch.float32
        se3 = pp.randn_SE3(3, 1, 5, device=device, dtype=dtype)
        points = torch.randn(1, 4, 1, 4, device=device, dtype=dtype)

        result = SE3_Act4_fwd(se3, points)
        expected = se3.Act(points)

        assert result.shape == expected.shape == (3, 4, 5, 4)
        torch.testing.assert_close(result, expected)

    def test_broadcast_4d_complex(self, device):
        """Test complex 4D broadcasting."""
        dtype = torch.float32
        se3 = pp.randn_SE3(2, 1, 3, 1, device=device, dtype=dtype)
        points = torch.randn(1, 4, 1, 5, 4, device=device, dtype=dtype)

        result = SE3_Act4_fwd(se3, points)
        expected = se3.Act(points)

        assert result.shape == expected.shape == (2, 4, 3, 5, 4)
        torch.testing.assert_close(result, expected)

    def test_broadcast_different_ndim(self, device):
        """Test broadcasting tensors with different number of dimensions."""
        dtype = torch.float32
        se3 = pp.randn_SE3(5, device=device, dtype=dtype)  # shape (5, 7)
        points = torch.randn(3, 1, 4, device=device, dtype=dtype)  # shape (3, 1, 4)

        result = SE3_Act4_fwd(se3, points)
        expected = se3.Act(points)

        assert result.shape == expected.shape == (3, 5, 4)
        torch.testing.assert_close(result, expected)


class TestSE3Act4Precision:
    """Test SE3_Act4_fwd precision handling."""

    def test_fp32_precision(self, device):
        """Test float32 precision and accuracy."""
        dtype = torch.float32
        se3 = pp.randn_SE3(10, device=device, dtype=dtype)
        points = torch.randn(10, 4, device=device, dtype=dtype)

        result = SE3_Act4_fwd(se3, points)
        expected = se3.Act(points)

        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SE3_Act4))

    def test_fp64_precision(self, device):
        """Test float64 precision and accuracy."""
        dtype = torch.float64
        se3 = pp.randn_SE3(10, device=device, dtype=dtype)
        points = torch.randn(10, 4, device=device, dtype=dtype)

        result = SE3_Act4_fwd(se3, points)
        expected = se3.Act(points)

        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SE3_Act4))

    def test_fp16_precision(self, device):
        """Test float16 precision and accuracy."""
        dtype = torch.float16
        se3 = pp.randn_SE3(10, device=device, dtype=dtype)
        points = torch.randn(10, 4, device=device, dtype=dtype)

        result = SE3_Act4_fwd(se3, points)
        expected = se3.Act(points)

        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SE3_Act4))

    def test_output_dtype_preserved(self, device, dtype):
        """Test that output dtype matches input dtype."""
        se3 = pp.randn_SE3(5, device=device, dtype=dtype)
        points = torch.randn(5, 4, device=device, dtype=dtype)

        result = SE3_Act4_fwd(se3, points)

        assert result.dtype == dtype


class TestSE3Act4EdgeCases:
    """Test SE3_Act4_fwd edge cases."""

    def test_identity_transform(self, device):
        """Test that identity transformation leaves points unchanged."""
        dtype = torch.float32
        # Identity SE3: (t_x, t_y, t_z, q_x, q_y, q_z, q_w) = (0, 0, 0, 0, 0, 0, 1)
        se3 = pp.identity_SE3(5, device=device, dtype=dtype)
        points = torch.randn(5, 4, device=device, dtype=dtype)

        result = SE3_Act4_fwd(se3, points)

        # For identity, out[:3] = p[:3] + 0*p[3] = p[:3], out[3] = p[3]
        torch.testing.assert_close(result, points)

    def test_pure_translation(self, device):
        """Test pure translation (identity rotation)."""
        dtype = torch.float32
        # Create SE3 with only translation (identity quaternion)
        translation = torch.randn(5, 3, device=device, dtype=dtype)
        identity_quat = torch.zeros(5, 4, device=device, dtype=dtype)
        identity_quat[..., 3] = 1.0  # w = 1
        se3_data = torch.cat([translation, identity_quat], dim=-1)
        se3 = pp.SE3(se3_data)
        
        points = torch.randn(5, 4, device=device, dtype=dtype)
        result = SE3_Act4_fwd(se3, points)
        
        # Pure translation: out[:3] = p[:3] + t * p[3], out[3] = p[3]
        expected_xyz = points[..., :3] + translation * points[..., 3:4]
        expected = torch.cat([expected_xyz, points[..., 3:]], dim=-1)
        torch.testing.assert_close(result, expected)

    def test_homogeneous_coord_preserved(self, device):
        """Test that the homogeneous coordinate (4th component) is preserved."""
        dtype = torch.float32
        se3 = pp.randn_SE3(5, device=device, dtype=dtype)
        points = torch.randn(5, 4, device=device, dtype=dtype)

        result = SE3_Act4_fwd(se3, points)

        # The 4th component should be unchanged
        torch.testing.assert_close(result[..., 3], points[..., 3])

    def test_homogeneous_coord_zero(self, device):
        """Test with homogeneous coordinate = 0 (direction vector)."""
        dtype = torch.float32
        se3 = pp.randn_SE3(5, device=device, dtype=dtype)
        points = torch.randn(5, 4, device=device, dtype=dtype)
        points[..., 3] = 0.0  # Set homogeneous coord to 0

        result = SE3_Act4_fwd(se3, points)
        expected = se3.Act(points)

        torch.testing.assert_close(result, expected)
        # Direction vectors should only be rotated, not translated
        assert torch.allclose(result[..., 3], torch.zeros_like(result[..., 3]))

    def test_homogeneous_coord_one(self, device):
        """Test with homogeneous coordinate = 1 (standard point)."""
        dtype = torch.float32
        se3 = pp.randn_SE3(5, device=device, dtype=dtype)
        points = torch.randn(5, 4, device=device, dtype=dtype)
        points[..., 3] = 1.0  # Set homogeneous coord to 1

        result = SE3_Act4_fwd(se3, points)
        expected = se3.Act(points)

        torch.testing.assert_close(result, expected)

    def test_single_element_batch(self, device):
        """Test with batch size of 1."""
        dtype = torch.float32
        se3 = pp.randn_SE3(1, device=device, dtype=dtype)
        points = torch.randn(1, 4, device=device, dtype=dtype)

        result = SE3_Act4_fwd(se3, points)
        expected = se3.Act(points)

        assert result.shape == (1, 4)
        torch.testing.assert_close(result, expected)

    def test_large_batch(self, device):
        """Test with large batch size."""
        dtype = torch.float32
        se3 = pp.randn_SE3(1000, device=device, dtype=dtype)
        points = torch.randn(1000, 4, device=device, dtype=dtype)

        result = SE3_Act4_fwd(se3, points)
        expected = se3.Act(points)

        assert result.shape == (1000, 4)
        torch.testing.assert_close(result, expected)

    def test_output_device(self, device):
        """Test that output is on the same device as input."""
        dtype = torch.float32
        se3 = pp.randn_SE3(5, device=device, dtype=dtype)
        points = torch.randn(5, 4, device=device, dtype=dtype)

        result = SE3_Act4_fwd(se3, points)

        assert result.device == points.device == se3.device


class TestSE3Act4Errors:
    """Test SE3_Act4_fwd error handling."""

    def test_incompatible_shapes_raises(self, device):
        """Test that incompatible shapes raise ValueError."""
        dtype = torch.float32
        se3 = pp.randn_SE3(3, device=device, dtype=dtype)
        points = torch.randn(5, 4, device=device, dtype=dtype)  # Incompatible

        with pytest.raises(ValueError, match="not broadcastable"):
            SE3_Act4_fwd(se3, points)

    def test_5d_batch_raises(self, device):
        """Test that 5D batch dimensions raise NotImplementedError."""
        dtype = torch.float32
        se3 = pp.randn_SE3(2, 2, 2, 2, 2, device=device, dtype=dtype)
        points = torch.randn(2, 2, 2, 2, 2, 4, device=device, dtype=dtype)

        with pytest.raises(NotImplementedError, match="Batch dimensions > 4"):
            SE3_Act4_fwd(se3, points)


# =============================================================================
# Backward Pass Tests
# =============================================================================

class TestSE3Act4BwdBatchDimensions:
    """Test SE3_Act4 backward with various batch dimensions."""

    def test_1d_batch(self, device, dtype_bwd):
        """Test backward with 1D batch dimension."""
        se3_data = pp.randn_SE3(5, device=device, dtype=dtype_bwd)
        points_data = torch.randn(5, 4, device=device, dtype=dtype_bwd)

        # Our implementation
        se3_ours = se3_data.clone().requires_grad_(True)
        points_ours = points_data.clone().requires_grad_(True)
        result_ours = SE3_Act4.apply(se3_ours, points_ours)
        result_ours.sum().backward()

        # PyPose reference
        se3_ref = se3_data.clone().requires_grad_(True)
        points_ref = points_data.clone().requires_grad_(True)
        result_ref = se3_ref.Act(points_ref)
        result_ref.sum().backward()

        assert se3_ours.grad.shape == se3_ref.grad.shape == (5, 7)
        assert points_ours.grad.shape == points_ref.grad.shape == (5, 4)
        torch.testing.assert_close(se3_ours.grad, se3_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Act4))
        torch.testing.assert_close(points_ours.grad, points_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Act4))

    def test_2d_batch(self, device, dtype_bwd):
        """Test backward with 2D batch dimensions."""
        se3_data = pp.randn_SE3(3, 4, device=device, dtype=dtype_bwd)
        points_data = torch.randn(3, 4, 4, device=device, dtype=dtype_bwd)

        se3_ours = se3_data.clone().requires_grad_(True)
        points_ours = points_data.clone().requires_grad_(True)
        result_ours = SE3_Act4.apply(se3_ours, points_ours)
        result_ours.sum().backward()

        se3_ref = se3_data.clone().requires_grad_(True)
        points_ref = points_data.clone().requires_grad_(True)
        result_ref = se3_ref.Act(points_ref)
        result_ref.sum().backward()

        assert se3_ours.grad.shape == se3_ref.grad.shape == (3, 4, 7)
        assert points_ours.grad.shape == points_ref.grad.shape == (3, 4, 4)
        torch.testing.assert_close(se3_ours.grad, se3_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Act4))
        torch.testing.assert_close(points_ours.grad, points_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Act4))

    def test_3d_batch(self, device, dtype_bwd):
        """Test backward with 3D batch dimensions."""
        se3_data = pp.randn_SE3(2, 3, 4, device=device, dtype=dtype_bwd)
        points_data = torch.randn(2, 3, 4, 4, device=device, dtype=dtype_bwd)

        se3_ours = se3_data.clone().requires_grad_(True)
        points_ours = points_data.clone().requires_grad_(True)
        result_ours = SE3_Act4.apply(se3_ours, points_ours)
        result_ours.sum().backward()

        se3_ref = se3_data.clone().requires_grad_(True)
        points_ref = points_data.clone().requires_grad_(True)
        result_ref = se3_ref.Act(points_ref)
        result_ref.sum().backward()

        assert se3_ours.grad.shape == se3_ref.grad.shape == (2, 3, 4, 7)
        assert points_ours.grad.shape == points_ref.grad.shape == (2, 3, 4, 4)
        torch.testing.assert_close(se3_ours.grad, se3_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Act4))
        torch.testing.assert_close(points_ours.grad, points_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Act4))

    def test_4d_batch(self, device, dtype_bwd):
        """Test backward with 4D batch dimensions."""
        se3_data = pp.randn_SE3(2, 2, 3, 4, device=device, dtype=dtype_bwd)
        points_data = torch.randn(2, 2, 3, 4, 4, device=device, dtype=dtype_bwd)

        se3_ours = se3_data.clone().requires_grad_(True)
        points_ours = points_data.clone().requires_grad_(True)
        result_ours = SE3_Act4.apply(se3_ours, points_ours)
        result_ours.sum().backward()

        se3_ref = se3_data.clone().requires_grad_(True)
        points_ref = points_data.clone().requires_grad_(True)
        result_ref = se3_ref.Act(points_ref)
        result_ref.sum().backward()

        assert se3_ours.grad.shape == se3_ref.grad.shape == (2, 2, 3, 4, 7)
        assert points_ours.grad.shape == points_ref.grad.shape == (2, 2, 3, 4, 4)
        torch.testing.assert_close(se3_ours.grad, se3_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Act4))
        torch.testing.assert_close(points_ours.grad, points_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Act4))

    def test_scalar_no_batch(self, device, dtype_bwd):
        """Test backward with no batch dimensions."""
        se3_data = pp.randn_SE3(device=device, dtype=dtype_bwd)
        points_data = torch.randn(4, device=device, dtype=dtype_bwd)

        se3_ours = se3_data.clone().requires_grad_(True)
        points_ours = points_data.clone().requires_grad_(True)
        result_ours = SE3_Act4.apply(se3_ours, points_ours)
        result_ours.sum().backward()

        se3_ref = se3_data.clone().requires_grad_(True)
        points_ref = points_data.clone().requires_grad_(True)
        result_ref = se3_ref.Act(points_ref)
        result_ref.sum().backward()

        assert se3_ours.grad.shape == se3_ref.grad.shape == (7,)
        assert points_ours.grad.shape == points_ref.grad.shape == (4,)
        torch.testing.assert_close(se3_ours.grad, se3_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Act4))
        torch.testing.assert_close(points_ours.grad, points_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Act4))


class TestSE3Act4BwdBroadcasting:
    """Test SE3_Act4 backward with broadcasting."""

    def test_broadcast_se3_singleton(self, device, dtype_bwd):
        """Test backward with single SE3 broadcast to multiple points."""
        se3_data = pp.randn_SE3(1, device=device, dtype=dtype_bwd)
        points_data = torch.randn(5, 4, device=device, dtype=dtype_bwd)

        se3_ours = se3_data.clone().requires_grad_(True)
        points_ours = points_data.clone().requires_grad_(True)
        result_ours = SE3_Act4.apply(se3_ours, points_ours)
        result_ours.sum().backward()

        se3_ref = se3_data.clone().requires_grad_(True)
        points_ref = points_data.clone().requires_grad_(True)
        result_ref = se3_ref.Act(points_ref)
        result_ref.sum().backward()

        assert se3_ours.grad.shape == se3_ref.grad.shape == (1, 7)
        assert points_ours.grad.shape == points_ref.grad.shape == (5, 4)
        torch.testing.assert_close(se3_ours.grad, se3_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Act4))
        torch.testing.assert_close(points_ours.grad, points_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Act4))

    def test_broadcast_points_singleton(self, device, dtype_bwd):
        """Test backward with single point broadcast to multiple SE3."""
        se3_data = pp.randn_SE3(5, device=device, dtype=dtype_bwd)
        points_data = torch.randn(1, 4, device=device, dtype=dtype_bwd)

        se3_ours = se3_data.clone().requires_grad_(True)
        points_ours = points_data.clone().requires_grad_(True)
        result_ours = SE3_Act4.apply(se3_ours, points_ours)
        result_ours.sum().backward()

        se3_ref = se3_data.clone().requires_grad_(True)
        points_ref = points_data.clone().requires_grad_(True)
        result_ref = se3_ref.Act(points_ref)
        result_ref.sum().backward()

        assert se3_ours.grad.shape == se3_ref.grad.shape == (5, 7)
        assert points_ours.grad.shape == points_ref.grad.shape == (1, 4)
        torch.testing.assert_close(se3_ours.grad, se3_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Act4))
        torch.testing.assert_close(points_ours.grad, points_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Act4))

    def test_broadcast_2d_cross(self, device, dtype_bwd):
        """Test backward with 2D cross-broadcasting: (1, 5) SE3 with (4, 1) points."""
        se3_data = pp.randn_SE3(1, 5, device=device, dtype=dtype_bwd)
        points_data = torch.randn(4, 1, 4, device=device, dtype=dtype_bwd)

        se3_ours = se3_data.clone().requires_grad_(True)
        points_ours = points_data.clone().requires_grad_(True)
        result_ours = SE3_Act4.apply(se3_ours, points_ours)
        result_ours.sum().backward()

        se3_ref = se3_data.clone().requires_grad_(True)
        points_ref = points_data.clone().requires_grad_(True)
        result_ref = se3_ref.Act(points_ref)
        result_ref.sum().backward()

        assert se3_ours.grad.shape == se3_ref.grad.shape == (1, 5, 7)
        assert points_ours.grad.shape == points_ref.grad.shape == (4, 1, 4)
        torch.testing.assert_close(se3_ours.grad, se3_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Act4))
        torch.testing.assert_close(points_ours.grad, points_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Act4))

    def test_broadcast_3d_complex(self, device, dtype_bwd):
        """Test backward with complex 3D broadcasting."""
        se3_data = pp.randn_SE3(3, 1, 5, device=device, dtype=dtype_bwd)
        points_data = torch.randn(1, 4, 1, 4, device=device, dtype=dtype_bwd)

        se3_ours = se3_data.clone().requires_grad_(True)
        points_ours = points_data.clone().requires_grad_(True)
        result_ours = SE3_Act4.apply(se3_ours, points_ours)
        result_ours.sum().backward()

        se3_ref = se3_data.clone().requires_grad_(True)
        points_ref = points_data.clone().requires_grad_(True)
        result_ref = se3_ref.Act(points_ref)
        result_ref.sum().backward()

        assert se3_ours.grad.shape == se3_ref.grad.shape == (3, 1, 5, 7)
        assert points_ours.grad.shape == points_ref.grad.shape == (1, 4, 1, 4)
        torch.testing.assert_close(se3_ours.grad, se3_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Act4))
        torch.testing.assert_close(points_ours.grad, points_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Act4))

    def test_broadcast_different_ndim(self, device, dtype_bwd):
        """Test backward with different number of dimensions."""
        se3_data = pp.randn_SE3(5, device=device, dtype=dtype_bwd)  # shape (5, 7)
        points_data = torch.randn(3, 1, 4, device=device, dtype=dtype_bwd)  # shape (3, 1, 4)

        se3_ours = se3_data.clone().requires_grad_(True)
        points_ours = points_data.clone().requires_grad_(True)
        result_ours = SE3_Act4.apply(se3_ours, points_ours)
        result_ours.sum().backward()

        se3_ref = se3_data.clone().requires_grad_(True)
        points_ref = points_data.clone().requires_grad_(True)
        result_ref = se3_ref.Act(points_ref)
        result_ref.sum().backward()

        assert se3_ours.grad.shape == se3_ref.grad.shape == (5, 7)
        assert points_ours.grad.shape == points_ref.grad.shape == (3, 1, 4)
        torch.testing.assert_close(se3_ours.grad, se3_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Act4))
        torch.testing.assert_close(points_ours.grad, points_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Act4))


class TestSE3Act4BwdPrecision:
    """Test SE3_Act4 backward precision handling."""

    def test_precision(self, device, dtype_bwd):
        """Test backward precision."""
        se3_data = pp.randn_SE3(10, device=device, dtype=dtype_bwd)
        points_data = torch.randn(10, 4, device=device, dtype=dtype_bwd)

        se3_ours = se3_data.clone().requires_grad_(True)
        points_ours = points_data.clone().requires_grad_(True)
        result_ours = SE3_Act4.apply(se3_ours, points_ours)
        result_ours.sum().backward()

        se3_ref = se3_data.clone().requires_grad_(True)
        points_ref = points_data.clone().requires_grad_(True)
        result_ref = se3_ref.Act(points_ref)
        result_ref.sum().backward()

        torch.testing.assert_close(se3_ours.grad, se3_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Act4))
        torch.testing.assert_close(points_ours.grad, points_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Act4))

    def test_grad_dtype_preserved(self, device, dtype_bwd):
        """Test that gradient dtype matches input dtype."""
        se3 = pp.randn_SE3(5, device=device, dtype=dtype_bwd)
        se3.requires_grad_(True)
        points = torch.randn(5, 4, device=device, dtype=dtype_bwd, requires_grad=True)

        result = SE3_Act4.apply(se3, points)
        result.sum().backward()

        assert se3.grad.dtype == dtype_bwd
        assert points.grad.dtype == dtype_bwd


class TestSE3Act4BwdEdgeCases:
    """Test SE3_Act4 backward edge cases."""

    def test_quaternion_w_component_zero(self, device, dtype_bwd):
        """Test that quaternion gradient w component is always zero."""
        se3 = pp.randn_SE3(5, device=device, dtype=dtype_bwd)
        se3.requires_grad_(True)
        points = torch.randn(5, 4, device=device, dtype=dtype_bwd, requires_grad=True)

        result = SE3_Act4.apply(se3, points)
        result.sum().backward()

        # The w component of quaternion gradient should always be 0
        # For SE3, this is the 7th component (index 6)
        assert torch.allclose(se3.grad[..., 6], torch.zeros_like(se3.grad[..., 6]))

    def test_identity_transform_gradient(self, device):
        """Test gradient through identity transformation."""
        dtype = torch.float32
        se3 = pp.identity_SE3(5, device=device, dtype=dtype)
        se3.requires_grad_(True)
        points = torch.randn(5, 4, device=device, dtype=dtype, requires_grad=True)

        result = SE3_Act4.apply(se3, points)
        result.sum().backward()

        # For identity transform, point gradient for xyz should be all ones
        # and the w component gradient should be 1 as well
        expected_points_grad = torch.ones_like(points)
        torch.testing.assert_close(points.grad, expected_points_grad)

    def test_homogeneous_coord_zero_gradient(self, device):
        """Test gradient with homogeneous coordinate = 0 (direction vector)."""
        dtype = torch.float32
        se3_data = pp.randn_SE3(5, device=device, dtype=dtype)
        points_data = torch.randn(5, 4, device=device, dtype=dtype)
        points_data[..., 3] = 0.0  # Direction vector

        se3_ours = se3_data.clone().requires_grad_(True)
        points_ours = points_data.clone().requires_grad_(True)
        result_ours = SE3_Act4.apply(se3_ours, points_ours)
        result_ours.sum().backward()

        se3_ref = se3_data.clone().requires_grad_(True)
        points_ref = points_data.clone().requires_grad_(True)
        result_ref = se3_ref.Act(points_ref)
        result_ref.sum().backward()

        # For direction vectors, translation gradient should be 0
        torch.testing.assert_close(se3_ours.grad[..., :3], torch.zeros_like(se3_ours.grad[..., :3]))
        torch.testing.assert_close(se3_ours.grad, se3_ref.grad, **get_bwd_tolerances(dtype, Operator.SE3_Act4))
        torch.testing.assert_close(points_ours.grad, points_ref.grad, **get_bwd_tolerances(dtype, Operator.SE3_Act4))

    def test_large_batch_gradient(self, device):
        """Test gradient with large batch."""
        dtype = torch.float32
        se3_data = pp.randn_SE3(1000, device=device, dtype=dtype)
        points_data = torch.randn(1000, 4, device=device, dtype=dtype)

        se3_ours = se3_data.clone().requires_grad_(True)
        points_ours = points_data.clone().requires_grad_(True)
        result_ours = SE3_Act4.apply(se3_ours, points_ours)
        result_ours.sum().backward()

        se3_ref = se3_data.clone().requires_grad_(True)
        points_ref = points_data.clone().requires_grad_(True)
        result_ref = se3_ref.Act(points_ref)
        result_ref.sum().backward()

        torch.testing.assert_close(se3_ours.grad, se3_ref.grad, **get_bwd_tolerances(dtype, Operator.SE3_Act4))
        torch.testing.assert_close(points_ours.grad, points_ref.grad, **get_bwd_tolerances(dtype, Operator.SE3_Act4))

    def test_grad_device(self, device):
        """Test that gradients are on the correct device."""
        dtype = torch.float32
        se3 = pp.randn_SE3(5, device=device, dtype=dtype)
        se3.requires_grad_(True)
        points = torch.randn(5, 4, device=device, dtype=dtype, requires_grad=True)

        result = SE3_Act4.apply(se3, points)
        result.sum().backward()

        assert se3.grad.device == se3.device
        assert points.grad.device == points.device


class TestSE3Act4WarpBackend:
    """Test SE3_Act4 through the warp_SE3Type backend."""

    def test_warp_backend_integration(self, device, dtype):
        """Test that warp_SE3Type.Act works correctly with 4D points."""
        from pypose_warp import to_warp_backend
        
        se3 = pp.randn_SE3(5, device=device, dtype=dtype)
        se3_warp = to_warp_backend(se3)
        points = torch.randn(5, 4, device=device, dtype=dtype)

        result = se3_warp.Act(points)
        expected = se3.Act(points)

        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SE3_Act4))

    def test_warp_backend_gradient(self, device, dtype_bwd):
        """Test that warp_SE3Type.Act gradients work correctly with 4D points."""
        from pypose_warp import to_warp_backend
        
        se3_data = pp.randn_SE3(5, device=device, dtype=dtype_bwd)
        points_data = torch.randn(5, 4, device=device, dtype=dtype_bwd)

        # Our warp implementation
        se3_warp = to_warp_backend(se3_data.clone()).requires_grad_(True)
        points_ours = points_data.clone().requires_grad_(True)
        result_ours = se3_warp.Act(points_ours)
        result_ours.sum().backward()

        # PyPose reference
        se3_ref = se3_data.clone().requires_grad_(True)
        points_ref = points_data.clone().requires_grad_(True)
        result_ref = se3_ref.Act(points_ref)
        result_ref.sum().backward()

        torch.testing.assert_close(se3_warp.grad, se3_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Act4))
        torch.testing.assert_close(points_ours.grad, points_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Act4))

