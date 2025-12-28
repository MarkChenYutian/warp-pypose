"""Unit tests for SE3_Inv forward and backward."""
import pytest
import torch
import pypose as pp
from pypose_warp.ltype.SE3_group import SE3_Inv, SE3_Inv_fwd
from conftest import get_fwd_tolerances, get_bwd_tolerances, Operator, skip_if_nan_inputs, compute_reference_fp32


class TestSE3InvBatchDimensions:
    """Test SE3_Inv_fwd with various batch dimensions."""

    def test_1d_batch(self, device, dtype):
        """Test with 1D batch dimension."""
        se3 = pp.randn_SE3(5, device=device, dtype=dtype)
        skip_if_nan_inputs(se3)

        result = SE3_Inv_fwd(se3)
        expected = compute_reference_fp32(se3, 'Inv')

        assert result.shape == expected.shape == (5, 7)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SE3_Inv))

    def test_2d_batch(self, device, dtype):
        """Test with 2D batch dimensions."""
        se3 = pp.randn_SE3(3, 4, device=device, dtype=dtype)
        skip_if_nan_inputs(se3)

        result = SE3_Inv_fwd(se3)
        expected = compute_reference_fp32(se3, 'Inv')

        assert result.shape == expected.shape == (3, 4, 7)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SE3_Inv))

    def test_3d_batch(self, device, dtype):
        """Test with 3D batch dimensions."""
        se3 = pp.randn_SE3(2, 3, 4, device=device, dtype=dtype)
        skip_if_nan_inputs(se3)

        result = SE3_Inv_fwd(se3)
        expected = compute_reference_fp32(se3, 'Inv')

        assert result.shape == expected.shape == (2, 3, 4, 7)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SE3_Inv))

    def test_4d_batch(self, device, dtype):
        """Test with 4D batch dimensions."""
        se3 = pp.randn_SE3(2, 3, 4, 5, device=device, dtype=dtype)
        skip_if_nan_inputs(se3)

        result = SE3_Inv_fwd(se3)
        expected = compute_reference_fp32(se3, 'Inv')

        assert result.shape == expected.shape == (2, 3, 4, 5, 7)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SE3_Inv))

    def test_scalar_no_batch(self, device, dtype):
        """Test with no batch dimensions (single transform)."""
        se3 = pp.randn_SE3(device=device, dtype=dtype)
        skip_if_nan_inputs(se3)

        result = SE3_Inv_fwd(se3)
        expected = compute_reference_fp32(se3, 'Inv')

        assert result.shape == expected.shape == (7,)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SE3_Inv))


class TestSE3InvPrecision:
    """Test SE3_Inv_fwd precision handling."""

    def test_fp32_precision(self, device):
        """Test float32 precision and accuracy."""
        dtype = torch.float32
        se3 = pp.randn_SE3(10, device=device, dtype=dtype)

        result = SE3_Inv_fwd(se3)
        expected = se3.Inv()

        torch.testing.assert_close(result, expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_Inv))

    def test_fp64_precision(self, device):
        """Test float64 precision and accuracy."""
        dtype = torch.float64
        se3 = pp.randn_SE3(10, device=device, dtype=dtype)

        result = SE3_Inv_fwd(se3)
        expected = se3.Inv()

        torch.testing.assert_close(result, expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_Inv))

    def test_fp16_precision(self, device):
        """Test float16 precision and accuracy."""
        dtype = torch.float16
        se3 = pp.randn_SE3(10, device=device, dtype=dtype)
        skip_if_nan_inputs(se3)

        result = SE3_Inv_fwd(se3)
        expected = compute_reference_fp32(se3, 'Inv')

        torch.testing.assert_close(result, expected, **get_fwd_tolerances(dtype, Operator.SE3_Inv))

    def test_output_dtype_preserved(self, device, dtype):
        """Test that output dtype matches input dtype."""
        se3 = pp.randn_SE3(5, device=device, dtype=dtype)
        skip_if_nan_inputs(se3)

        result = SE3_Inv_fwd(se3)

        assert result.dtype == dtype


class TestSE3InvEdgeCases:
    """Test SE3_Inv_fwd edge cases."""

    def test_identity_transform(self, device):
        """Test that identity inverse is identity."""
        dtype = torch.float32
        se3 = pp.identity_SE3(5, device=device, dtype=dtype)

        result = SE3_Inv_fwd(se3)

        # Identity inverse is identity
        torch.testing.assert_close(result, se3.tensor())

    def test_inverse_of_inverse(self, device, dtype):
        """Test that inv(inv(X)) = X."""
        se3 = pp.randn_SE3(5, device=device, dtype=dtype)

        # Convert to warp backend for consistency
        from pypose_warp import to_warp_backend
        se3_warp = to_warp_backend(se3)
        
        inv_once = se3_warp.Inv()
        inv_twice = inv_once.Inv()

        torch.testing.assert_close(inv_twice.tensor(), se3.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_Inv))

    def test_pure_translation(self, device):
        """Test inverse of pure translation (identity rotation)."""
        dtype = torch.float32
        # Create SE3 with only translation
        translation = torch.randn(5, 3, device=device, dtype=dtype)
        identity_quat = torch.zeros(5, 4, device=device, dtype=dtype)
        identity_quat[..., 3] = 1.0  # w = 1
        se3_data = torch.cat([translation, identity_quat], dim=-1)
        se3 = pp.SE3(se3_data)
        
        result = SE3_Inv_fwd(se3)
        
        # Pure translation inverse: t_inv = -t, q_inv = q
        expected_t = -translation
        expected_q = identity_quat.clone()
        expected_q[..., :3] = -expected_q[..., :3]  # Conjugate (but identity quat xyz are 0)
        
        torch.testing.assert_close(result[..., :3], expected_t)
        torch.testing.assert_close(result[..., 3:], identity_quat)

    def test_single_element_batch(self, device):
        """Test with batch size of 1."""
        dtype = torch.float32
        se3 = pp.randn_SE3(1, device=device, dtype=dtype)

        result = SE3_Inv_fwd(se3)
        expected = se3.Inv()

        assert result.shape == (1, 7)
        torch.testing.assert_close(result, expected.tensor())

    def test_large_batch(self, device):
        """Test with large batch size."""
        dtype = torch.float32
        se3 = pp.randn_SE3(1000, device=device, dtype=dtype)

        result = SE3_Inv_fwd(se3)
        expected = se3.Inv()

        assert result.shape == (1000, 7)
        torch.testing.assert_close(result, expected.tensor())

    def test_output_device(self, device):
        """Test that output is on the same device as input."""
        dtype = torch.float32
        se3 = pp.randn_SE3(5, device=device, dtype=dtype)

        result = SE3_Inv_fwd(se3)

        assert result.device == se3.device


class TestSE3InvMathematicalProperties:
    """Test mathematical properties of SE3 inverse."""

    def test_inv_mul_equals_identity(self, device, dtype):
        """Test that X * inv(X) = identity."""
        se3 = pp.randn_SE3(5, device=device, dtype=dtype)

        result = SE3_Inv_fwd(se3)
        result_se3 = pp.SE3(result)
        
        # X * inv(X) should be identity
        product = se3 @ result_se3
        identity = pp.identity_SE3(5, device=device, dtype=dtype)

        torch.testing.assert_close(product.tensor(), identity.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_Inv))

    def test_quaternion_norm_preserved(self, device, dtype):
        """Test that quaternion part has unit norm after inversion."""
        se3 = pp.randn_SE3(5, device=device, dtype=dtype)

        result = SE3_Inv_fwd(se3)
        
        # Quaternion is components 3:7
        q_inv = result[..., 3:]
        q_norm = torch.norm(q_inv, dim=-1)
        
        # Use appropriate tolerance for dtype (fp16 has lower precision)
        if dtype == torch.float16:
            torch.testing.assert_close(q_norm, torch.ones_like(q_norm), atol=1e-2, rtol=1e-2)
        else:
            torch.testing.assert_close(q_norm, torch.ones_like(q_norm), atol=1e-4, rtol=1e-4)


class TestSE3InvErrors:
    """Test SE3_Inv_fwd error handling."""

    def test_5d_batch_raises(self, device):
        """Test that 5D batch dimensions raise NotImplementedError."""
        dtype = torch.float32
        se3 = pp.randn_SE3(2, 2, 2, 2, 2, device=device, dtype=dtype)

        with pytest.raises(NotImplementedError, match="Batch dimensions > 4"):
            SE3_Inv_fwd(se3)


# =============================================================================
# Backward Pass Tests
# =============================================================================

class TestSE3InvBwdBatchDimensions:
    """Test SE3_Inv backward with various batch dimensions."""

    def test_1d_batch(self, device, dtype_bwd):
        """Test backward with 1D batch dimension."""
        se3_data = pp.randn_SE3(5, device=device, dtype=dtype_bwd)

        # Our implementation
        se3_ours = se3_data.clone().requires_grad_(True)
        result_ours = SE3_Inv.apply(se3_ours)
        result_ours.sum().backward()

        # PyPose reference
        se3_ref = se3_data.clone().requires_grad_(True)
        result_ref = se3_ref.Inv()
        result_ref.sum().backward()

        assert se3_ours.grad.shape == se3_ref.grad.shape == (5, 7)
        torch.testing.assert_close(se3_ours.grad, se3_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Inv))

    def test_2d_batch(self, device, dtype_bwd):
        """Test backward with 2D batch dimensions."""
        se3_data = pp.randn_SE3(3, 4, device=device, dtype=dtype_bwd)

        se3_ours = se3_data.clone().requires_grad_(True)
        result_ours = SE3_Inv.apply(se3_ours)
        result_ours.sum().backward()

        se3_ref = se3_data.clone().requires_grad_(True)
        result_ref = se3_ref.Inv()
        result_ref.sum().backward()

        assert se3_ours.grad.shape == se3_ref.grad.shape == (3, 4, 7)
        torch.testing.assert_close(se3_ours.grad, se3_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Inv))

    def test_3d_batch(self, device, dtype_bwd):
        """Test backward with 3D batch dimensions."""
        se3_data = pp.randn_SE3(2, 3, 4, device=device, dtype=dtype_bwd)

        se3_ours = se3_data.clone().requires_grad_(True)
        result_ours = SE3_Inv.apply(se3_ours)
        result_ours.sum().backward()

        se3_ref = se3_data.clone().requires_grad_(True)
        result_ref = se3_ref.Inv()
        result_ref.sum().backward()

        assert se3_ours.grad.shape == se3_ref.grad.shape == (2, 3, 4, 7)
        torch.testing.assert_close(se3_ours.grad, se3_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Inv))

    def test_4d_batch(self, device, dtype_bwd):
        """Test backward with 4D batch dimensions."""
        se3_data = pp.randn_SE3(2, 2, 3, 4, device=device, dtype=dtype_bwd)

        se3_ours = se3_data.clone().requires_grad_(True)
        result_ours = SE3_Inv.apply(se3_ours)
        result_ours.sum().backward()

        se3_ref = se3_data.clone().requires_grad_(True)
        result_ref = se3_ref.Inv()
        result_ref.sum().backward()

        assert se3_ours.grad.shape == se3_ref.grad.shape == (2, 2, 3, 4, 7)
        torch.testing.assert_close(se3_ours.grad, se3_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Inv))

    def test_scalar_no_batch(self, device, dtype_bwd):
        """Test backward with no batch dimensions."""
        se3_data = pp.randn_SE3(device=device, dtype=dtype_bwd)

        se3_ours = se3_data.clone().requires_grad_(True)
        result_ours = SE3_Inv.apply(se3_ours)
        result_ours.sum().backward()

        se3_ref = se3_data.clone().requires_grad_(True)
        result_ref = se3_ref.Inv()
        result_ref.sum().backward()

        assert se3_ours.grad.shape == se3_ref.grad.shape == (7,)
        torch.testing.assert_close(se3_ours.grad, se3_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Inv))


class TestSE3InvBwdPrecision:
    """Test SE3_Inv backward precision handling."""

    def test_precision(self, device, dtype_bwd):
        """Test backward precision."""
        se3_data = pp.randn_SE3(10, device=device, dtype=dtype_bwd)

        se3_ours = se3_data.clone().requires_grad_(True)
        result_ours = SE3_Inv.apply(se3_ours)
        result_ours.sum().backward()

        se3_ref = se3_data.clone().requires_grad_(True)
        result_ref = se3_ref.Inv()
        result_ref.sum().backward()

        torch.testing.assert_close(se3_ours.grad, se3_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Inv))

    def test_grad_dtype_preserved(self, device, dtype_bwd):
        """Test that gradient dtype matches input dtype."""
        se3 = pp.randn_SE3(5, device=device, dtype=dtype_bwd)
        se3.requires_grad_(True)

        result = SE3_Inv.apply(se3)
        result.sum().backward()

        assert se3.grad.dtype == dtype_bwd


class TestSE3InvBwdEdgeCases:
    """Test SE3_Inv backward edge cases."""

    def test_quaternion_w_component_zero(self, device, dtype_bwd):
        """Test that quaternion gradient w component is always zero."""
        se3 = pp.randn_SE3(5, device=device, dtype=dtype_bwd)
        se3.requires_grad_(True)

        result = SE3_Inv.apply(se3)
        result.sum().backward()

        # The w component of quaternion gradient should always be 0
        assert torch.allclose(se3.grad[..., 6], torch.zeros_like(se3.grad[..., 6]))

    def test_identity_transform_gradient(self, device):
        """Test gradient through identity transformation."""
        dtype = torch.float32
        se3_data = pp.identity_SE3(5, device=device, dtype=dtype)

        se3_ours = se3_data.clone().requires_grad_(True)
        result_ours = SE3_Inv.apply(se3_ours)
        result_ours.sum().backward()

        se3_ref = se3_data.clone().requires_grad_(True)
        result_ref = se3_ref.Inv()
        result_ref.sum().backward()

        torch.testing.assert_close(se3_ours.grad, se3_ref.grad, **get_bwd_tolerances(dtype, Operator.SE3_Inv))

    def test_large_batch_gradient(self, device):
        """Test gradient with large batch."""
        dtype = torch.float32
        se3_data = pp.randn_SE3(1000, device=device, dtype=dtype)

        se3_ours = se3_data.clone().requires_grad_(True)
        result_ours = SE3_Inv.apply(se3_ours)
        result_ours.sum().backward()

        se3_ref = se3_data.clone().requires_grad_(True)
        result_ref = se3_ref.Inv()
        result_ref.sum().backward()

        torch.testing.assert_close(se3_ours.grad, se3_ref.grad, **get_bwd_tolerances(dtype, Operator.SE3_Inv))

    def test_grad_device(self, device):
        """Test that gradients are on the correct device."""
        dtype = torch.float32
        se3 = pp.randn_SE3(5, device=device, dtype=dtype)
        se3.requires_grad_(True)

        result = SE3_Inv.apply(se3)
        result.sum().backward()

        assert se3.grad.device == se3.device


class TestSE3InvWarpBackend:
    """Test SE3_Inv through the warp_SE3Type backend."""

    def test_warp_backend_integration(self, device, dtype):
        """Test that warp_SE3Type.Inv works correctly."""
        from pypose_warp import to_warp_backend
        
        se3 = pp.randn_SE3(5, device=device, dtype=dtype)
        se3_warp = to_warp_backend(se3)

        result = se3_warp.Inv()
        expected = se3.Inv()

        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_Inv))

    def test_warp_backend_gradient(self, device, dtype_bwd):
        """Test that warp_SE3Type.Inv gradients work correctly."""
        from pypose_warp import to_warp_backend
        
        se3_data = pp.randn_SE3(5, device=device, dtype=dtype_bwd)

        # Our warp implementation
        se3_warp = to_warp_backend(se3_data.clone()).requires_grad_(True)
        result_ours = se3_warp.Inv()
        result_ours.sum().backward()

        # PyPose reference
        se3_ref = se3_data.clone().requires_grad_(True)
        result_ref = se3_ref.Inv()
        result_ref.sum().backward()

        torch.testing.assert_close(se3_warp.grad, se3_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_Inv))

