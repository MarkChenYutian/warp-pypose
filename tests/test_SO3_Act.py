"""Unit tests for SO3_Act_fwd."""
import pytest
import torch
import pypose as pp

from pypose_warp.ltype.SO3_group import SO3_Act_fwd
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

        max_error = (result - expected).abs().max().item()
        assert max_error < 1e-5, f"Max error {max_error} exceeds threshold"

    def test_fp64_precision(self, device):
        """Test float64 precision and accuracy."""
        dtype = torch.float64
        so3 = pp.randn_SO3(10, device=device, dtype=dtype)
        points = torch.randn(10, 3, device=device, dtype=dtype)

        result = SO3_Act_fwd(so3, points)
        expected = so3.Act(points)

        max_error = (result - expected).abs().max().item()
        assert max_error < 1e-14, f"Max error {max_error} exceeds threshold"

    def test_fp16_precision(self, device):
        """Test float16 precision and accuracy."""
        dtype = torch.float16
        so3 = pp.randn_SO3(10, device=device, dtype=dtype)
        points = torch.randn(10, 3, device=device, dtype=dtype)

        result = SO3_Act_fwd(so3, points)
        expected = so3.Act(points)

        max_error = (result - expected).abs().max().item()
        assert max_error < 1e-2, f"Max error {max_error} exceeds threshold"

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
