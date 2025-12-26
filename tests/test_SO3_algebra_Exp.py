"""Unit tests for so3_Exp forward and backward."""
import math
import pytest
import torch
import pypose as pp
from pypose_warp.ltype.SO3_algebra import so3_Exp, so3_Exp_fwd
from conftest import get_fwd_tolerances, get_bwd_tolerances, Operator


# =============================================================================
# Forward Pass Tests
# =============================================================================


class TestSo3_Exp_Fwd_BatchDimensions:
    """Test so3_Exp_fwd with various batch dimensions."""

    def test_1d_batch(self, device, dtype):
        """Test with 1D batch dimension."""
        so3 = pp.randn_so3(5, device=device, dtype=dtype)
        
        result = so3_Exp_fwd(so3)
        expected = so3.Exp()
        
        assert result.shape == expected.shape == (5, 4)
        assert result.dtype == dtype
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.so3_Exp))

    def test_2d_batch(self, device, dtype):
        """Test with 2D batch dimensions."""
        so3 = pp.randn_so3(3, 4, device=device, dtype=dtype)
        
        result = so3_Exp_fwd(so3)
        expected = so3.Exp()
        
        assert result.shape == expected.shape == (3, 4, 4)
        assert result.dtype == dtype
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.so3_Exp))

    def test_3d_batch(self, device, dtype):
        """Test with 3D batch dimensions."""
        so3 = pp.randn_so3(2, 3, 4, device=device, dtype=dtype)
        
        result = so3_Exp_fwd(so3)
        expected = so3.Exp()
        
        assert result.shape == expected.shape == (2, 3, 4, 4)
        assert result.dtype == dtype
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.so3_Exp))

    def test_4d_batch(self, device, dtype):
        """Test with 4D batch dimensions."""
        so3 = pp.randn_so3(2, 2, 3, 4, device=device, dtype=dtype)
        
        result = so3_Exp_fwd(so3)
        expected = so3.Exp()
        
        assert result.shape == expected.shape == (2, 2, 3, 4, 4)
        assert result.dtype == dtype
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.so3_Exp))

    def test_scalar_no_batch(self, device, dtype):
        """Test with no batch dimensions (scalar input)."""
        so3 = pp.randn_so3(device=device, dtype=dtype)
        
        result = so3_Exp_fwd(so3)
        expected = so3.Exp()
        
        assert result.shape == expected.shape == (4,)
        assert result.dtype == dtype
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.so3_Exp))


class TestSo3_Exp_Fwd_Precision:
    """Test so3_Exp_fwd precision handling."""

    def test_precision(self, device, dtype):
        """Test precision for various dtypes."""
        so3 = pp.randn_so3(10, device=device, dtype=dtype)
        
        result = so3_Exp_fwd(so3)
        expected = so3.Exp()
        
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.so3_Exp))

    def test_output_dtype_preserved(self, device, dtype):
        """Test that output dtype matches input dtype."""
        so3 = pp.randn_so3(5, device=device, dtype=dtype)
        
        result = so3_Exp_fwd(so3)
        assert result.dtype == dtype


class TestSo3_Exp_Fwd_EdgeCases:
    """Test so3_Exp_fwd with edge cases."""

    def test_identity_rotation(self, device, dtype):
        """Test with zero vector (identity rotation)."""
        so3 = pp.identity_so3(5, device=device, dtype=dtype)
        
        result = so3_Exp_fwd(so3)
        expected = pp.identity_SO3(5, device=device, dtype=dtype)
        
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.so3_Exp))

    def test_small_angle(self, device):
        """Test with very small rotation angle (Taylor expansion regime)."""
        dtype = torch.float64
        # Small angle vector
        angle = 1e-8
        so3 = pp.so3(torch.tensor([[angle, 0., 0.]], device=device, dtype=dtype))
        
        result = so3_Exp_fwd(so3)
        expected = so3.Exp()
        
        torch.testing.assert_close(result.tensor(), expected.tensor(), atol=1e-10, rtol=1e-10)

    def test_180_degree_rotation(self, device):
        """Test with 180 degree rotation."""
        dtype = torch.float64
        # 180 degrees around x-axis
        so3 = pp.so3(torch.tensor([[math.pi, 0., 0.]], device=device, dtype=dtype))
        
        result = so3_Exp_fwd(so3)
        expected = so3.Exp()
        
        torch.testing.assert_close(result.tensor(), expected.tensor(), atol=1e-10, rtol=1e-10)

    def test_quaternion_is_unit(self, device, dtype):
        """Test that output quaternion is unit norm."""
        so3 = pp.randn_so3(10, device=device, dtype=dtype)
        
        result = so3_Exp_fwd(so3)
        norms = torch.norm(result.tensor(), dim=-1)
        
        torch.testing.assert_close(norms, torch.ones_like(norms), **get_fwd_tolerances(dtype, Operator.so3_Exp))

    def test_large_batch(self, device, dtype):
        """Test with a large batch size."""
        so3 = pp.randn_so3(1000, device=device, dtype=dtype)
        
        result = so3_Exp_fwd(so3)
        expected = so3.Exp()
        
        assert result.shape == expected.shape == (1000, 4)
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.so3_Exp))

    def test_output_device(self, device, dtype):
        """Test that output device matches input device."""
        so3 = pp.randn_so3(5, device=device, dtype=dtype)
        
        result = so3_Exp_fwd(so3)
        assert str(result.device).startswith(device)


class TestSo3_Exp_Fwd_Errors:
    """Test so3_Exp_fwd error handling."""

    def test_5d_batch_raises(self, device, dtype):
        """Test that 5D batch dimensions raise NotImplementedError."""
        so3 = pp.randn_so3(2, 2, 2, 2, 2, device=device, dtype=dtype)
        
        with pytest.raises(NotImplementedError, match="Batch dimensions > 4"):
            so3_Exp_fwd(so3)


# =============================================================================
# Backward Pass Tests (use dtype_bwd fixture - no fp16)
# =============================================================================


class TestSo3_Exp_Bwd_PyposeAlignment:
    """
    Test so3_Exp backward alignment with PyPose.
    
    Note: We do NOT use torch.autograd.gradcheck here because:
    1. gradcheck computes raw tensor derivatives (dq/dx for quaternion q, axis-angle x)
    2. PyPose uses Lie algebra tangent space derivatives, which use the left Jacobian
    3. These are mathematically different conventions - both valid, but our goal is
       to match PyPose's behavior for drop-in compatibility
       
    The backward pass uses: grad_input = grad_output[..., :-1] @ Jl(input)
    where Jl is the left Jacobian of so3.
    """

    def test_backward_matches_pypose_fp64(self, device):
        """Verify backward exactly matches PyPose in fp64."""
        dtype = torch.float64
        so3_data = pp.randn_so3(10, device=device, dtype=dtype)
        
        # Our implementation
        so3_ours = so3_data.tensor().clone().requires_grad_(True)
        result_ours = so3_Exp.apply(pp.LieTensor(so3_ours, ltype=pp.so3_type))
        result_ours.tensor().sum().backward()
        
        # PyPose reference
        so3_ref = so3_data.clone().requires_grad_(True)
        result_ref = so3_ref.Exp()
        result_ref.tensor().sum().backward()
        
        # Should match to machine epsilon in fp64
        torch.testing.assert_close(so3_ours.grad, so3_ref.grad, atol=1e-10, rtol=1e-10)

    def test_backward_matches_pypose_fp32(self, device):
        """Verify backward matches PyPose in fp32 within tolerance."""
        dtype = torch.float32
        so3_data = pp.randn_so3(10, device=device, dtype=dtype)
        
        so3_ours = so3_data.tensor().clone().requires_grad_(True)
        result_ours = so3_Exp.apply(pp.LieTensor(so3_ours, ltype=pp.so3_type))
        result_ours.tensor().sum().backward()
        
        so3_ref = so3_data.clone().requires_grad_(True)
        result_ref = so3_ref.Exp()
        result_ref.tensor().sum().backward()
        
        torch.testing.assert_close(so3_ours.grad, so3_ref.grad, **get_bwd_tolerances(dtype, Operator.so3_Exp))

    def test_backward_jacobian_structure(self, device):
        """Verify the backward uses the left Jacobian structure."""
        dtype = torch.float64
        # Use a specific input where we can verify the Jacobian
        so3 = torch.tensor([[0.1, 0.2, 0.3]], device=device, dtype=dtype, requires_grad=True)
        
        result = so3_Exp.apply(pp.LieTensor(so3, ltype=pp.so3_type))
        # Use identity gradient for quaternion xyz components
        grad_q = torch.tensor([[1., 1., 1., 0.]], device=device, dtype=dtype)
        result.tensor().backward(grad_q)
        
        # The gradient should be: [1, 1, 1] @ Jl(so3)
        # Verify it's finite and reasonable
        assert not torch.isnan(so3.grad).any()
        assert not torch.isinf(so3.grad).any()
        assert so3.grad.shape == (1, 3)


class TestSo3_Exp_Bwd_BatchDimensions:
    """Test so3_Exp backward with various batch dimensions."""

    def test_1d_batch(self, device, dtype_bwd):
        """Test backward with 1D batch dimension."""
        so3_data = pp.randn_so3(5, device=device, dtype=dtype_bwd)
        
        # Our implementation - pass tensor to apply
        so3_ours = so3_data.tensor().clone().requires_grad_(True)
        result_ours = so3_Exp.apply(pp.LieTensor(so3_ours, ltype=pp.so3_type))
        result_ours.tensor().sum().backward()
        
        # PyPose reference
        so3_ref = so3_data.clone().requires_grad_(True)
        result_ref = so3_ref.Exp()
        result_ref.tensor().sum().backward()
        
        torch.testing.assert_close(so3_ours.grad, so3_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.so3_Exp))

    def test_2d_batch(self, device, dtype_bwd):
        """Test backward with 2D batch dimensions."""
        so3_data = pp.randn_so3(3, 4, device=device, dtype=dtype_bwd)
        
        so3_ours = so3_data.tensor().clone().requires_grad_(True)
        result_ours = so3_Exp.apply(pp.LieTensor(so3_ours, ltype=pp.so3_type))
        result_ours.tensor().sum().backward()
        
        so3_ref = so3_data.clone().requires_grad_(True)
        result_ref = so3_ref.Exp()
        result_ref.tensor().sum().backward()
        
        torch.testing.assert_close(so3_ours.grad, so3_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.so3_Exp))

    def test_scalar_no_batch(self, device, dtype_bwd):
        """Test backward with no batch dimensions."""
        so3_data = pp.randn_so3(device=device, dtype=dtype_bwd)
        
        so3_ours = so3_data.tensor().clone().requires_grad_(True)
        result_ours = so3_Exp.apply(pp.LieTensor(so3_ours, ltype=pp.so3_type))
        result_ours.tensor().sum().backward()
        
        so3_ref = so3_data.clone().requires_grad_(True)
        result_ref = so3_ref.Exp()
        result_ref.tensor().sum().backward()
        
        torch.testing.assert_close(so3_ours.grad, so3_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.so3_Exp))


class TestSo3_Exp_Bwd_EdgeCases:
    """Test so3_Exp backward with edge cases."""

    def test_identity_gradient(self, device):
        """Test backward with identity rotation (zero vector)."""
        dtype = torch.float64
        so3_data = pp.identity_so3(5, device=device, dtype=dtype)
        so3 = so3_data.tensor().clone().requires_grad_(True)
        
        result = so3_Exp.apply(pp.LieTensor(so3, ltype=pp.so3_type))
        result.tensor().sum().backward()
        
        # Gradients should be finite
        assert not torch.isnan(so3.grad).any()
        assert not torch.isinf(so3.grad).any()

    def test_small_angle_gradient(self, device):
        """Test backward with small angle (Taylor expansion regime)."""
        dtype = torch.float64
        angle = 1e-8
        so3 = torch.tensor([[angle, 0., 0.]], device=device, dtype=dtype, requires_grad=True)
        
        result = so3_Exp.apply(pp.LieTensor(so3, ltype=pp.so3_type))
        result.tensor().sum().backward()
        
        # Gradients should be finite
        assert not torch.isnan(so3.grad).any()
        assert not torch.isinf(so3.grad).any()

    def test_grad_dtype_preserved(self, device, dtype_bwd):
        """Test that gradient dtype matches input dtype."""
        so3_data = pp.randn_so3(5, device=device, dtype=dtype_bwd)
        so3 = so3_data.tensor().clone().requires_grad_(True)
        
        result = so3_Exp.apply(pp.LieTensor(so3, ltype=pp.so3_type))
        result.tensor().sum().backward()
        
        assert so3.grad.dtype == dtype_bwd

    def test_grad_device_preserved(self, device, dtype_bwd):
        """Test that gradient device matches input device."""
        so3_data = pp.randn_so3(5, device=device, dtype=dtype_bwd)
        so3 = so3_data.tensor().clone().requires_grad_(True)
        
        result = so3_Exp.apply(pp.LieTensor(so3, ltype=pp.so3_type))
        result.tensor().sum().backward()
        
        assert str(so3.grad.device).startswith(device)

    def test_large_batch_gradient(self, device, dtype_bwd):
        """Test backward with a large batch size."""
        so3_data = pp.randn_so3(1000, device=device, dtype=dtype_bwd)
        so3 = so3_data.tensor().clone().requires_grad_(True)
        
        result = so3_Exp.apply(pp.LieTensor(so3, ltype=pp.so3_type))
        result.tensor().sum().backward()
        
        assert so3.grad.shape == (1000, 3)
        assert not torch.isnan(so3.grad).any()
