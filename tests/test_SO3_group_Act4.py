"""
Unit tests for SO3_Act4 forward and backward passes.

SO3_Act4 applies rotation to 4D homogeneous points:
  out = [R @ p[:3], p[3]]
where R is the rotation matrix from quaternion X.
"""

import pytest
import torch
import pypose as pp

from conftest import get_fwd_tolerances, get_bwd_tolerances, Operator

# Import warp SO3_Act4
from pypose_warp.ltype.SO3_group import SO3_Act4, SO3_Act4_fwd, SO3_Act4_bwd


# ==============================================================================
# Test SO3_Act4 Forward
# ==============================================================================

class TestSO3Act4Fwd:
    """Tests for SO3_Act4 forward pass."""

    def test_basic_1d(self, device, dtype):
        """Test basic 1D batch forward."""
        tols = get_fwd_tolerances(dtype, Operator.SO3_Act4)

        X = pp.randn_SO3(5, device=device, dtype=dtype)
        p = torch.randn(5, 4, device=device, dtype=dtype)

        out = SO3_Act4.apply(X, p)

        # Reference: rotate first 3 components
        p3 = p[..., :3]
        R = pp.SO3(X).matrix()
        rotated = torch.einsum('...ij,...j->...i', R, p3)
        expected = torch.cat([rotated, p[..., 3:]], dim=-1)

        assert out.shape == expected.shape
        torch.testing.assert_close(out, expected, **tols)

    def test_basic_2d(self, device, dtype):
        """Test 2D batch forward."""
        tols = get_fwd_tolerances(dtype, Operator.SO3_Act4)

        X = pp.randn_SO3(3, 4, device=device, dtype=dtype)
        p = torch.randn(3, 4, 4, device=device, dtype=dtype)

        out = SO3_Act4.apply(X, p)

        p3 = p[..., :3]
        R = pp.SO3(X).matrix()
        rotated = torch.einsum('...ij,...j->...i', R, p3)
        expected = torch.cat([rotated, p[..., 3:]], dim=-1)

        assert out.shape == expected.shape
        torch.testing.assert_close(out, expected, **tols)

    def test_basic_3d(self, device, dtype):
        """Test 3D batch forward."""
        tols = get_fwd_tolerances(dtype, Operator.SO3_Act4)

        X = pp.randn_SO3(2, 3, 4, device=device, dtype=dtype)
        p = torch.randn(2, 3, 4, 4, device=device, dtype=dtype)

        out = SO3_Act4.apply(X, p)

        p3 = p[..., :3]
        R = pp.SO3(X).matrix()
        rotated = torch.einsum('...ij,...j->...i', R, p3)
        expected = torch.cat([rotated, p[..., 3:]], dim=-1)

        assert out.shape == expected.shape
        torch.testing.assert_close(out, expected, **tols)

    def test_broadcast_1d(self, device, dtype):
        """Test broadcasting in 1D."""
        # X (5,) broadcasts with p (1, 4)
        X = pp.randn_SO3(5, device=device, dtype=dtype)
        p = torch.randn(1, 4, device=device, dtype=dtype)

        out = SO3_Act4.apply(X, p)
        assert out.shape == torch.Size([5, 4])

    def test_broadcast_2d(self, device, dtype):
        """Test broadcasting in 2D."""
        # X (5, 1, 4) broadcasts with p (1, 3, 4)
        X = pp.randn_SO3(5, 1, device=device, dtype=dtype)
        p = torch.randn(1, 3, 4, device=device, dtype=dtype)

        out = SO3_Act4.apply(X, p)
        assert out.shape == torch.Size([5, 3, 4])

    def test_fourth_component_unchanged(self, device, dtype):
        """Test that the 4th component is passed through unchanged."""
        tols = get_fwd_tolerances(dtype, Operator.SO3_Act4)

        X = pp.randn_SO3(5, device=device, dtype=dtype)
        p = torch.randn(5, 4, device=device, dtype=dtype)

        out = SO3_Act4.apply(X, p)

        # 4th component should be unchanged
        torch.testing.assert_close(out[..., 3], p[..., 3], **tols)


# ==============================================================================
# Test SO3_Act4 Backward
# ==============================================================================

class TestSO3Act4Bwd:
    """Tests for SO3_Act4 backward pass."""

    def test_grad_shapes(self, device, dtype_bwd):
        """Test that backward produces correct gradient shapes."""
        X = pp.randn_SO3(5, device=device, dtype=dtype_bwd, requires_grad=True)
        p = torch.randn(5, 4, device=device, dtype=dtype_bwd, requires_grad=True)

        out = SO3_Act4.apply(X, p)
        loss = out.sum()
        loss.backward()

        assert X.grad is not None
        assert p.grad is not None
        assert X.grad.shape == X.shape
        assert p.grad.shape == p.shape

    def test_grad_p_fourth_component(self, device, dtype_bwd):
        """Test that gradient for p[..., 3] is correctly passed through."""
        tols = get_bwd_tolerances(dtype_bwd, Operator.SO3_Act4)

        X = pp.randn_SO3(5, device=device, dtype=dtype_bwd, requires_grad=True)
        p = torch.randn(5, 4, device=device, dtype=dtype_bwd, requires_grad=True)

        # Loss = sum(out)
        out = SO3_Act4.apply(X, p)
        loss = out.sum()
        loss.backward()

        # Gradient of sum w.r.t. p[..., 3] should be 1
        torch.testing.assert_close(
            p.grad[..., 3], 
            torch.ones(5, device=device, dtype=dtype_bwd),
            **tols
        )

    def test_grad_finite_difference(self, device):
        """Test gradients against finite differences (fp64 only for precision)."""
        dtype = torch.float64
        eps = 1e-6

        X = pp.randn_SO3(3, device=device, dtype=dtype, requires_grad=True)
        p = torch.randn(3, 4, device=device, dtype=dtype, requires_grad=True)

        out = SO3_Act4.apply(X, p)
        loss = out.sum()
        loss.backward()

        grad_p_analytical = p.grad.clone()

        # Finite difference for p
        grad_p_fd = torch.zeros_like(p)
        for i in range(3):
            for j in range(4):
                p_plus = p.detach().clone()
                p_minus = p.detach().clone()
                p_plus[i, j] += eps
                p_minus[i, j] -= eps
                
                loss_plus = SO3_Act4.apply(X.detach(), p_plus).sum()
                loss_minus = SO3_Act4.apply(X.detach(), p_minus).sum()
                grad_p_fd[i, j] = (loss_plus - loss_minus) / (2 * eps)

        torch.testing.assert_close(grad_p_analytical, grad_p_fd, atol=1e-6, rtol=1e-5)


# ==============================================================================
# Test Backward Broadcasting
# ==============================================================================

class TestSO3Act4BwdBroadcasting:
    """Tests for gradient reduction in broadcasting scenarios."""

    def test_broadcast_X_single(self, device, dtype_bwd):
        """Test gradient reduction when X has size 1."""
        X = pp.randn_SO3(1, device=device, dtype=dtype_bwd, requires_grad=True)
        p = torch.randn(5, 4, device=device, dtype=dtype_bwd, requires_grad=True)

        out = SO3_Act4.apply(X, p)
        loss = out.sum()
        loss.backward()

        # X.grad should have shape (1, 4)
        assert X.grad.shape == torch.Size([1, 4])
        assert p.grad.shape == torch.Size([5, 4])

    def test_broadcast_p_single(self, device, dtype_bwd):
        """Test gradient reduction when p has size 1."""
        X = pp.randn_SO3(5, device=device, dtype=dtype_bwd, requires_grad=True)
        p = torch.randn(1, 4, device=device, dtype=dtype_bwd, requires_grad=True)

        out = SO3_Act4.apply(X, p)
        loss = out.sum()
        loss.backward()

        # p.grad should have shape (1, 4)
        assert X.grad.shape == torch.Size([5, 4])
        assert p.grad.shape == torch.Size([1, 4])

    def test_broadcast_2d_cross(self, device, dtype_bwd):
        """Test gradient reduction with 2D cross-broadcasting."""
        X = pp.randn_SO3(5, 1, device=device, dtype=dtype_bwd, requires_grad=True)
        p = torch.randn(1, 3, 4, device=device, dtype=dtype_bwd, requires_grad=True)

        out = SO3_Act4.apply(X, p)
        loss = out.sum()
        loss.backward()

        assert X.grad.shape == torch.Size([5, 1, 4])
        assert p.grad.shape == torch.Size([1, 3, 4])


# ==============================================================================
# Test via warp_SO3Type
# ==============================================================================

class TestWarpSO3TypeAct4:
    """Test SO3_Act4 via the warp_SO3Type interface."""

    def test_act_dispatches_to_act4(self, device, dtype):
        """Test that Act correctly dispatches to Act4 for 4D points."""
        tols = get_fwd_tolerances(dtype, Operator.SO3_Act4)

        from pypose_warp.ltype.SO3_group import warpSO3_type

        X = pp.randn_SO3(5, device=device, dtype=dtype)
        p = torch.randn(5, 4, device=device, dtype=dtype)

        # Call through the warp_SO3Type interface
        out = warpSO3_type.Act(X, p)

        # Reference
        p3 = p[..., :3]
        R = pp.SO3(X).matrix()
        rotated = torch.einsum('...ij,...j->...i', R, p3)
        expected = torch.cat([rotated, p[..., 3:]], dim=-1)

        torch.testing.assert_close(out, expected, **tols)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
