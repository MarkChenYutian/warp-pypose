"""Unit tests for SO3_AddExp forward and backward."""
import math
import pytest
import torch
import pypose as pp
from pypose_warp.ltype.SO3_group import SO3_AddExp, SO3_AddExp_fwd, SO3_AddExp_bwd
from pypose_warp import to_warp_backend
from conftest import get_fwd_tolerances, get_bwd_tolerances, Operator


# =============================================================================
# Forward Pass Tests
# =============================================================================


class TestSO3_AddExp_Fwd_BatchDimensions:
    """Test SO3_AddExp_fwd with various batch dimensions."""

    def test_1d_batch(self, device, dtype):
        """Test with 1D batch dimension."""
        # Create SO3 element and tangent space delta
        X = pp.randn_SO3(5, device=device, dtype=dtype)
        delta = torch.randn(5, 3, device=device, dtype=dtype) * 0.1
        
        # Convert to warp backend
        X_warp = to_warp_backend(X)
        
        result = SO3_AddExp_fwd(delta, X_warp)
        
        # Reference: Exp(delta) * X using PyPose
        delta_exp = pp.so3(delta).Exp()
        expected = delta_exp @ X
        
        assert result.shape == expected.shape == (5, 4)
        assert result.dtype == dtype
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype))

    def test_2d_batch(self, device, dtype):
        """Test with 2D batch dimensions."""
        X = pp.randn_SO3(3, 4, device=device, dtype=dtype)
        delta = torch.randn(3, 4, 3, device=device, dtype=dtype) * 0.1
        
        X_warp = to_warp_backend(X)
        
        result = SO3_AddExp_fwd(delta, X_warp)
        
        delta_exp = pp.so3(delta).Exp()
        expected = delta_exp @ X
        
        assert result.shape == expected.shape == (3, 4, 4)
        assert result.dtype == dtype
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype))

    def test_3d_batch(self, device, dtype):
        """Test with 3D batch dimensions."""
        X = pp.randn_SO3(2, 3, 4, device=device, dtype=dtype)
        delta = torch.randn(2, 3, 4, 3, device=device, dtype=dtype) * 0.1
        
        X_warp = to_warp_backend(X)
        
        result = SO3_AddExp_fwd(delta, X_warp)
        
        delta_exp = pp.so3(delta).Exp()
        expected = delta_exp @ X
        
        assert result.shape == expected.shape == (2, 3, 4, 4)
        assert result.dtype == dtype
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype))

    def test_4d_batch(self, device, dtype):
        """Test with 4D batch dimensions."""
        X = pp.randn_SO3(2, 2, 3, 4, device=device, dtype=dtype)
        delta = torch.randn(2, 2, 3, 4, 3, device=device, dtype=dtype) * 0.1
        
        X_warp = to_warp_backend(X)
        
        result = SO3_AddExp_fwd(delta, X_warp)
        
        delta_exp = pp.so3(delta).Exp()
        expected = delta_exp @ X
        
        assert result.shape == expected.shape == (2, 2, 3, 4, 4)
        assert result.dtype == dtype
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype))

    def test_scalar_no_batch(self, device, dtype):
        """Test with no batch dimensions (scalar input)."""
        X = pp.randn_SO3(device=device, dtype=dtype)
        delta = torch.randn(3, device=device, dtype=dtype) * 0.1
        
        X_warp = to_warp_backend(X)
        
        result = SO3_AddExp_fwd(delta, X_warp)
        
        delta_exp = pp.so3(delta).Exp()
        expected = delta_exp @ X
        
        assert result.shape == expected.shape == (4,)
        assert result.dtype == dtype
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype))


class TestSO3_AddExp_Fwd_EdgeCases:
    """Test SO3_AddExp_fwd with edge cases."""

    def test_zero_delta(self, device, dtype):
        """Test with zero delta (should return X unchanged)."""
        X = pp.randn_SO3(10, device=device, dtype=dtype)
        delta = torch.zeros(10, 3, device=device, dtype=dtype)
        
        X_warp = to_warp_backend(X)
        
        result = SO3_AddExp_fwd(delta, X_warp)
        
        # Exp(0) = identity, so result should be X
        torch.testing.assert_close(result.tensor(), X.tensor(), **get_fwd_tolerances(dtype))

    def test_identity_X(self, device, dtype):
        """Test with identity X."""
        X = pp.identity_SO3(10, device=device, dtype=dtype)
        delta = torch.randn(10, 3, device=device, dtype=dtype) * 0.1
        
        X_warp = to_warp_backend(X)
        
        result = SO3_AddExp_fwd(delta, X_warp)
        
        # Result should be Exp(delta) * identity = Exp(delta)
        expected = pp.so3(delta).Exp()
        
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype))

    def test_small_delta(self, device):
        """Test with very small delta (Taylor expansion regime)."""
        dtype = torch.float64
        X = pp.randn_SO3(10, device=device, dtype=dtype)
        delta = torch.randn(10, 3, device=device, dtype=dtype) * 1e-8
        
        X_warp = to_warp_backend(X)
        
        result = SO3_AddExp_fwd(delta, X_warp)
        
        delta_exp = pp.so3(delta).Exp()
        expected = delta_exp @ X
        
        torch.testing.assert_close(result.tensor(), expected.tensor(), atol=1e-10, rtol=1e-10)

    def test_quaternion_is_unit(self, device, dtype):
        """Test that output quaternion is unit norm."""
        X = pp.randn_SO3(10, device=device, dtype=dtype)
        delta = torch.randn(10, 3, device=device, dtype=dtype) * 0.5
        
        X_warp = to_warp_backend(X)
        
        result = SO3_AddExp_fwd(delta, X_warp)
        norms = torch.norm(result.tensor(), dim=-1)
        
        torch.testing.assert_close(norms, torch.ones_like(norms), **get_fwd_tolerances(dtype))

    def test_large_batch(self, device, dtype):
        """Test with a large batch size."""
        X = pp.randn_SO3(1000, device=device, dtype=dtype)
        delta = torch.randn(1000, 3, device=device, dtype=dtype) * 0.1
        
        X_warp = to_warp_backend(X)
        
        result = SO3_AddExp_fwd(delta, X_warp)
        
        delta_exp = pp.so3(delta).Exp()
        expected = delta_exp @ X
        
        assert result.shape == expected.shape == (1000, 4)
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype))


class TestSO3_AddExp_Fwd_Errors:
    """Test SO3_AddExp_fwd error handling."""

    def test_5d_batch_raises(self, device, dtype):
        """Test that 5D batch dimensions raise NotImplementedError."""
        X = pp.randn_SO3(2, 2, 2, 2, 2, device=device, dtype=dtype)
        delta = torch.randn(2, 2, 2, 2, 2, 3, device=device, dtype=dtype)
        
        X_warp = to_warp_backend(X)
        
        with pytest.raises(NotImplementedError, match="Batch dimensions > 4"):
            SO3_AddExp_fwd(delta, X_warp)


# =============================================================================
# Backward Pass Tests (use dtype_bwd fixture - no fp16)
# =============================================================================


class TestSO3_AddExp_Bwd_PyposeAlignment:
    """
    Test SO3_AddExp backward alignment with PyPose.
    
    The backward should match the composition of Exp and Mul backward passes.
    """

    def test_backward_matches_pypose_fp64(self, device):
        """Verify backward matches PyPose (Exp then Mul) in fp64."""
        dtype = torch.float64
        X_data = pp.randn_SO3(10, device=device, dtype=dtype)
        delta_data = torch.randn(10, 3, device=device, dtype=dtype) * 0.1
        
        # Our implementation
        delta_ours = delta_data.clone().requires_grad_(True)
        X_ours = X_data.clone().requires_grad_(True)
        X_warp = to_warp_backend(X_ours)
        
        result_ours = SO3_AddExp.apply(delta_ours, X_warp)
        result_ours.tensor().sum().backward()
        
        # PyPose reference: Exp(delta) @ X
        delta_ref = delta_data.clone().requires_grad_(True)
        X_ref = X_data.clone().requires_grad_(True)
        delta_exp = pp.so3(delta_ref).Exp()
        result_ref = delta_exp @ X_ref
        result_ref.tensor().sum().backward()
        
        # Should match to good precision in fp64
        torch.testing.assert_close(delta_ours.grad, delta_ref.grad, atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, atol=1e-9, rtol=1e-9)

    def test_backward_matches_pypose_fp32(self, device):
        """Verify backward matches PyPose in fp32 within tolerance."""
        dtype = torch.float32
        X_data = pp.randn_SO3(10, device=device, dtype=dtype)
        delta_data = torch.randn(10, 3, device=device, dtype=dtype) * 0.1
        
        # Our implementation
        delta_ours = delta_data.clone().requires_grad_(True)
        X_ours = X_data.clone().requires_grad_(True)
        X_warp = to_warp_backend(X_ours)
        
        result_ours = SO3_AddExp.apply(delta_ours, X_warp)
        result_ours.tensor().sum().backward()
        
        # PyPose reference
        delta_ref = delta_data.clone().requires_grad_(True)
        X_ref = X_data.clone().requires_grad_(True)
        delta_exp = pp.so3(delta_ref).Exp()
        result_ref = delta_exp @ X_ref
        result_ref.tensor().sum().backward()
        
        torch.testing.assert_close(delta_ours.grad, delta_ref.grad, atol=5e-3, rtol=5e-3)
        torch.testing.assert_close(X_ours.grad, X_ref.grad, atol=5e-3, rtol=5e-3)


class TestSO3_AddExp_Bwd_BatchDimensions:
    """Test SO3_AddExp backward with various batch dimensions."""

    def test_1d_batch(self, device, dtype_bwd):
        """Test backward with 1D batch dimension."""
        X_data = pp.randn_SO3(5, device=device, dtype=dtype_bwd)
        delta_data = torch.randn(5, 3, device=device, dtype=dtype_bwd) * 0.1
        
        # Our implementation
        delta_ours = delta_data.clone().requires_grad_(True)
        X_ours = X_data.clone().requires_grad_(True)
        X_warp = to_warp_backend(X_ours)
        
        result_ours = SO3_AddExp.apply(delta_ours, X_warp)
        result_ours.tensor().sum().backward()
        
        # PyPose reference
        delta_ref = delta_data.clone().requires_grad_(True)
        X_ref = X_data.clone().requires_grad_(True)
        delta_exp = pp.so3(delta_ref).Exp()
        result_ref = delta_exp @ X_ref
        result_ref.tensor().sum().backward()
        
        torch.testing.assert_close(delta_ours.grad, delta_ref.grad, **get_bwd_tolerances(dtype_bwd))
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd))

    def test_2d_batch(self, device, dtype_bwd):
        """Test backward with 2D batch dimensions."""
        X_data = pp.randn_SO3(3, 4, device=device, dtype=dtype_bwd)
        delta_data = torch.randn(3, 4, 3, device=device, dtype=dtype_bwd) * 0.1
        
        delta_ours = delta_data.clone().requires_grad_(True)
        X_ours = X_data.clone().requires_grad_(True)
        X_warp = to_warp_backend(X_ours)
        
        result_ours = SO3_AddExp.apply(delta_ours, X_warp)
        result_ours.tensor().sum().backward()
        
        delta_ref = delta_data.clone().requires_grad_(True)
        X_ref = X_data.clone().requires_grad_(True)
        delta_exp = pp.so3(delta_ref).Exp()
        result_ref = delta_exp @ X_ref
        result_ref.tensor().sum().backward()
        
        torch.testing.assert_close(delta_ours.grad, delta_ref.grad, **get_bwd_tolerances(dtype_bwd))
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd))

    def test_scalar_no_batch(self, device, dtype_bwd):
        """Test backward with no batch dimensions."""
        X_data = pp.randn_SO3(device=device, dtype=dtype_bwd)
        delta_data = torch.randn(3, device=device, dtype=dtype_bwd) * 0.1
        
        delta_ours = delta_data.clone().requires_grad_(True)
        X_ours = X_data.clone().requires_grad_(True)
        X_warp = to_warp_backend(X_ours)
        
        result_ours = SO3_AddExp.apply(delta_ours, X_warp)
        result_ours.tensor().sum().backward()
        
        delta_ref = delta_data.clone().requires_grad_(True)
        X_ref = X_data.clone().requires_grad_(True)
        delta_exp = pp.so3(delta_ref).Exp()
        result_ref = delta_exp @ X_ref
        result_ref.tensor().sum().backward()
        
        torch.testing.assert_close(delta_ours.grad, delta_ref.grad, **get_bwd_tolerances(dtype_bwd))
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd))


class TestSO3_AddExp_Bwd_EdgeCases:
    """Test SO3_AddExp backward with edge cases."""

    def test_zero_delta_gradient(self, device):
        """Test backward with zero delta."""
        dtype = torch.float64
        X_data = pp.randn_SO3(5, device=device, dtype=dtype)
        delta_data = torch.zeros(5, 3, device=device, dtype=dtype)
        
        delta = delta_data.clone().requires_grad_(True)
        X = X_data.clone().requires_grad_(True)
        X_warp = to_warp_backend(X)
        
        result = SO3_AddExp.apply(delta, X_warp)
        result.tensor().sum().backward()
        
        # Gradients should be finite
        assert not torch.isnan(delta.grad).any()
        assert not torch.isinf(delta.grad).any()
        assert not torch.isnan(X.grad).any()
        assert not torch.isinf(X.grad).any()

    def test_small_delta_gradient(self, device):
        """Test backward with small delta (Taylor expansion regime)."""
        dtype = torch.float64
        X_data = pp.randn_SO3(5, device=device, dtype=dtype)
        delta_data = torch.randn(5, 3, device=device, dtype=dtype) * 1e-8
        
        delta = delta_data.clone().requires_grad_(True)
        X = X_data.clone().requires_grad_(True)
        X_warp = to_warp_backend(X)
        
        result = SO3_AddExp.apply(delta, X_warp)
        result.tensor().sum().backward()
        
        # Gradients should be finite
        assert not torch.isnan(delta.grad).any()
        assert not torch.isinf(delta.grad).any()
        assert not torch.isnan(X.grad).any()
        assert not torch.isinf(X.grad).any()

    def test_grad_dtype_preserved(self, device, dtype_bwd):
        """Test that gradient dtype matches input dtype."""
        X_data = pp.randn_SO3(5, device=device, dtype=dtype_bwd)
        delta_data = torch.randn(5, 3, device=device, dtype=dtype_bwd) * 0.1
        
        delta = delta_data.clone().requires_grad_(True)
        X = X_data.clone().requires_grad_(True)
        X_warp = to_warp_backend(X)
        
        result = SO3_AddExp.apply(delta, X_warp)
        result.tensor().sum().backward()
        
        assert delta.grad.dtype == dtype_bwd
        assert X.grad.dtype == dtype_bwd

    def test_grad_device_preserved(self, device, dtype_bwd):
        """Test that gradient device matches input device."""
        X_data = pp.randn_SO3(5, device=device, dtype=dtype_bwd)
        delta_data = torch.randn(5, 3, device=device, dtype=dtype_bwd) * 0.1
        
        delta = delta_data.clone().requires_grad_(True)
        X = X_data.clone().requires_grad_(True)
        X_warp = to_warp_backend(X)
        
        result = SO3_AddExp.apply(delta, X_warp)
        result.tensor().sum().backward()
        
        assert str(delta.grad.device).startswith(device)
        assert str(X.grad.device).startswith(device)


# =============================================================================
# Integration Tests for add_ method
# =============================================================================


class TestSO3_Add_Integration:
    """Test the add_ method integration with SO3_AddExp."""

    def test_add_matches_pypose(self, device, dtype):
        """Test that warp add_ matches PyPose add_."""
        X_data = pp.randn_SO3(10, device=device, dtype=dtype)
        delta = torch.randn(10, 3, device=device, dtype=dtype) * 0.1
        
        # PyPose reference
        X_ref = X_data.clone()
        X_ref.add_(delta)
        
        # Warp backend
        X_warp = to_warp_backend(X_data.clone())
        X_warp.add_(delta)
        
        torch.testing.assert_close(X_warp.tensor(), X_ref.tensor(), **get_fwd_tolerances(dtype))

    def test_add_inplace_mutation(self, device, dtype):
        """Test that add_ modifies tensor in place."""
        X = to_warp_backend(pp.randn_SO3(10, device=device, dtype=dtype))
        original_data_ptr = X.tensor().data_ptr()
        
        delta = torch.randn(10, 3, device=device, dtype=dtype) * 0.1
        X.add_(delta)
        
        # Data pointer should remain the same (in-place modification)
        assert X.tensor().data_ptr() == original_data_ptr

    def test_add_larger_other_tensor(self, device, dtype):
        """Test add_ with other tensor having more than 3 dimensions."""
        X = to_warp_backend(pp.randn_SO3(10, device=device, dtype=dtype))
        
        # other has 6 dimensions, but add_ should only use first 3
        other = torch.randn(10, 6, device=device, dtype=dtype) * 0.1
        
        X_before = X.tensor().clone()
        X.add_(other)
        
        # Should have changed
        assert not torch.allclose(X.tensor(), X_before)
        
        # Result should match using only first 3 components
        X_ref = to_warp_backend(pp.LieTensor(X_before, ltype=X.ltype))
        delta = other[..., :3]
        X_ref_manual = SO3_AddExp.apply(delta, X_ref)
        
        torch.testing.assert_close(X.tensor(), X_ref_manual.tensor(), **get_fwd_tolerances(dtype))

