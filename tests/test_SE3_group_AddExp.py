"""Unit tests for SE3_AddExp forward and backward."""
import pytest
import torch
import pypose as pp
from pypose_warp import to_warp_backend, to_pypose_backend
from pypose_warp.ltype.SE3_group import SE3_AddExp, SE3_AddExp_fwd
from conftest import get_fwd_tolerances, get_bwd_tolerances, Operator, skip_if_nan_inputs


# =============================================================================
# Forward Pass Tests
# =============================================================================


class TestSE3AddExpBatchDimensions:
    """Test SE3_AddExp_fwd with various batch dimensions."""

    def test_1d_batch(self, device, dtype):
        """Test with 1D batch dimension."""
        delta = pp.randn_se3(5, device=device, dtype=dtype)
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        skip_if_nan_inputs(delta, X)
        
        delta_warp = to_warp_backend(delta)
        X_warp = to_warp_backend(X)
        
        result = SE3_AddExp_fwd(delta_warp, X_warp)
        expected = delta.Exp() * X
        
        # Skip if expected has NaN (PyPose FP16 numerical instability)
        if torch.isnan(expected.tensor()).any():
            pytest.skip("PyPose expected output contains NaN")
        
        assert result.shape == expected.shape == (5, 7)
        assert to_pypose_backend(result).ltype == expected.ltype == pp.SE3_type
        assert result.dtype == dtype
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_AddExp))

    def test_2d_batch(self, device, dtype):
        """Test with 2D batch dimensions."""
        delta = pp.randn_se3(3, 4, device=device, dtype=dtype)
        X = pp.randn_SE3(3, 4, device=device, dtype=dtype)
        skip_if_nan_inputs(delta, X)
        
        delta_warp = to_warp_backend(delta)
        X_warp = to_warp_backend(X)
        
        result = SE3_AddExp_fwd(delta_warp, X_warp)
        expected = delta.Exp() * X
        
        if torch.isnan(expected.tensor()).any():
            pytest.skip("PyPose expected output contains NaN")
        
        assert result.shape == expected.shape == (3, 4, 7)
        assert to_pypose_backend(result).ltype == pp.SE3_type
        assert result.dtype == dtype
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_AddExp))

    def test_3d_batch(self, device, dtype):
        """Test with 3D batch dimensions."""
        delta = pp.randn_se3(2, 3, 4, device=device, dtype=dtype)
        X = pp.randn_SE3(2, 3, 4, device=device, dtype=dtype)
        skip_if_nan_inputs(delta, X)
        
        delta_warp = to_warp_backend(delta)
        X_warp = to_warp_backend(X)
        
        result = SE3_AddExp_fwd(delta_warp, X_warp)
        expected = delta.Exp() * X
        
        if torch.isnan(expected.tensor()).any():
            pytest.skip("PyPose expected output contains NaN")
        
        assert result.shape == expected.shape == (2, 3, 4, 7)
        assert to_pypose_backend(result).ltype == pp.SE3_type
        assert result.dtype == dtype
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_AddExp))

    def test_4d_batch(self, device, dtype):
        """Test with 4D batch dimensions."""
        delta = pp.randn_se3(2, 2, 3, 4, device=device, dtype=dtype)
        X = pp.randn_SE3(2, 2, 3, 4, device=device, dtype=dtype)
        skip_if_nan_inputs(delta, X)
        
        delta_warp = to_warp_backend(delta)
        X_warp = to_warp_backend(X)
        
        result = SE3_AddExp_fwd(delta_warp, X_warp)
        expected = delta.Exp() * X
        
        if torch.isnan(expected.tensor()).any():
            pytest.skip("PyPose expected output contains NaN")
        
        assert result.shape == expected.shape == (2, 2, 3, 4, 7)
        assert to_pypose_backend(result).ltype == pp.SE3_type
        assert result.dtype == dtype
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_AddExp))

    def test_scalar_no_batch(self, device, dtype):
        """Test with no batch dimensions (single element)."""
        delta = pp.randn_se3(device=device, dtype=dtype)
        X = pp.randn_SE3(device=device, dtype=dtype)
        skip_if_nan_inputs(delta, X)
        
        delta_warp = to_warp_backend(delta)
        X_warp = to_warp_backend(X)
        
        result = SE3_AddExp_fwd(delta_warp, X_warp)
        expected = delta.Exp() * X
        
        if torch.isnan(expected.tensor()).any():
            pytest.skip("PyPose expected output contains NaN")
        
        assert result.shape == expected.shape == (7,)
        assert to_pypose_backend(result).ltype == pp.SE3_type
        assert result.dtype == dtype
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_AddExp))


class TestSE3AddExpPrecision:
    """Test SE3_AddExp_fwd precision handling."""

    def test_fp32_precision(self, device):
        """Test float32 precision and accuracy."""
        dtype = torch.float32
        delta = pp.randn_se3(10, device=device, dtype=dtype)
        X = pp.randn_SE3(10, device=device, dtype=dtype)
        
        delta_warp = to_warp_backend(delta)
        X_warp = to_warp_backend(X)
        
        result = SE3_AddExp_fwd(delta_warp, X_warp)
        expected = delta.Exp() * X
        
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_AddExp))

    def test_fp64_precision(self, device):
        """Test float64 precision and accuracy."""
        dtype = torch.float64
        delta = pp.randn_se3(10, device=device, dtype=dtype)
        X = pp.randn_SE3(10, device=device, dtype=dtype)
        
        delta_warp = to_warp_backend(delta)
        X_warp = to_warp_backend(X)
        
        result = SE3_AddExp_fwd(delta_warp, X_warp)
        expected = delta.Exp() * X
        
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_AddExp))

    def test_output_dtype_preserved(self, device, dtype):
        """Test that output dtype matches input dtype."""
        delta = pp.randn_se3(5, device=device, dtype=dtype)
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        skip_if_nan_inputs(delta, X)
        
        delta_warp = to_warp_backend(delta)
        X_warp = to_warp_backend(X)
        
        result = SE3_AddExp_fwd(delta_warp, X_warp)
        
        assert result.dtype == dtype


class TestSE3AddExpEdgeCases:
    """Test SE3_AddExp_fwd edge cases."""

    def test_zero_delta(self, device, dtype):
        """Test that zero delta leaves X unchanged."""
        delta = torch.zeros(5, 6, device=device, dtype=dtype)
        delta_lie = pp.LieTensor(delta, ltype=pp.se3_type)
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        skip_if_nan_inputs(X)
        
        delta_warp = to_warp_backend(delta_lie)
        X_warp = to_warp_backend(X)
        
        result = SE3_AddExp_fwd(delta_warp, X_warp)
        expected = delta_lie.Exp() * X
        
        # Exp(0) * X = identity * X = X
        torch.testing.assert_close(result.tensor(), expected.tensor(), atol=1e-5, rtol=1e-5)

    def test_identity_X(self, device):
        """Test with identity X."""
        dtype = torch.float64
        delta = pp.randn_se3(5, device=device, dtype=dtype)
        X = pp.identity_SE3(5, device=device, dtype=dtype)
        
        delta_warp = to_warp_backend(delta)
        X_warp = to_warp_backend(X)
        
        result = SE3_AddExp_fwd(delta_warp, X_warp)
        expected = delta.Exp() * X
        
        # Exp(delta) * identity = Exp(delta)
        torch.testing.assert_close(result.tensor(), expected.tensor(), atol=1e-9, rtol=1e-9)

    def test_single_element_batch(self, device, dtype):
        """Test with batch size of 1."""
        delta = pp.randn_se3(1, device=device, dtype=dtype)
        X = pp.randn_SE3(1, device=device, dtype=dtype)
        skip_if_nan_inputs(delta, X)
        
        delta_warp = to_warp_backend(delta)
        X_warp = to_warp_backend(X)
        
        result = SE3_AddExp_fwd(delta_warp, X_warp)
        expected = delta.Exp() * X
        
        if torch.isnan(expected.tensor()).any():
            pytest.skip("PyPose expected output contains NaN")
        
        assert result.shape == (1, 7)
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_AddExp))

    def test_large_batch(self, device, dtype):
        """Test with large batch size."""
        delta = pp.randn_se3(1000, device=device, dtype=dtype)
        X = pp.randn_SE3(1000, device=device, dtype=dtype)
        skip_if_nan_inputs(delta, X)
        
        delta_warp = to_warp_backend(delta)
        X_warp = to_warp_backend(X)
        
        result = SE3_AddExp_fwd(delta_warp, X_warp)
        expected = delta.Exp() * X
        
        if torch.isnan(expected.tensor()).any():
            pytest.skip("PyPose expected output contains NaN")
        
        assert result.shape == (1000, 7)
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_AddExp))

    def test_output_device(self, device, dtype):
        """Test that output is on the same device as input."""
        delta = pp.randn_se3(5, device=device, dtype=dtype)
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        skip_if_nan_inputs(delta, X)
        
        delta_warp = to_warp_backend(delta)
        X_warp = to_warp_backend(X)
        
        result = SE3_AddExp_fwd(delta_warp, X_warp)
        
        assert result.device == delta.device

    def test_output_ltype(self, device, dtype):
        """Test that output is SE3 LieTensor."""
        delta = pp.randn_se3(5, device=device, dtype=dtype)
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        skip_if_nan_inputs(delta, X)
        
        delta_warp = to_warp_backend(delta)
        X_warp = to_warp_backend(X)
        
        result = SE3_AddExp_fwd(delta_warp, X_warp)
        
        assert isinstance(result, pp.LieTensor)
        assert to_pypose_backend(result).ltype == pp.SE3_type


class TestSE3AddExpErrors:
    """Test SE3_AddExp_fwd error handling."""

    def test_5d_batch_raises(self, device, dtype):
        """Test that 5D batch dimensions raise NotImplementedError."""
        delta = pp.randn_se3(2, 2, 2, 2, 2, device=device, dtype=dtype)
        X = pp.randn_SE3(2, 2, 2, 2, 2, device=device, dtype=dtype)
        
        delta_warp = to_warp_backend(delta)
        X_warp = to_warp_backend(X)
        
        with pytest.raises(NotImplementedError, match="Batch dimensions > 4"):
            SE3_AddExp_fwd(delta_warp, X_warp)


class TestSE3AddExpAutogradFunction:
    """Test SE3_AddExp autograd function wrapper."""

    def test_apply_matches_fwd(self, device, dtype):
        """Test that SE3_AddExp.apply matches SE3_AddExp_fwd."""
        delta = pp.randn_se3(5, device=device, dtype=dtype)
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        skip_if_nan_inputs(delta, X)
        
        delta_warp = to_warp_backend(delta)
        X_warp = to_warp_backend(X)
        
        result_apply = SE3_AddExp.apply(delta_warp, X_warp)
        result_fwd = SE3_AddExp_fwd(delta_warp, X_warp)
        
        torch.testing.assert_close(result_apply.tensor(), result_fwd.tensor())

    def test_apply_1d_batch(self, device, dtype):
        """Test SE3_AddExp.apply with 1D batch."""
        delta = pp.randn_se3(5, device=device, dtype=dtype)
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        skip_if_nan_inputs(delta, X)
        
        delta_warp = to_warp_backend(delta)
        X_warp = to_warp_backend(X)
        
        result = SE3_AddExp.apply(delta_warp, X_warp)
        expected = delta.Exp() * X
        
        if torch.isnan(expected.tensor()).any():
            pytest.skip("PyPose expected output contains NaN")
        
        assert result.shape == expected.shape == (5, 7)
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_AddExp))


class TestSE3AddExpWarpBackend:
    """Test SE3_AddExp through the warp backend interface."""

    def test_warp_backend_add_(self, device, dtype):
        """Test add_ through warp backend LieType."""
        delta = pp.randn_se3(5, device=device, dtype=dtype)
        X = pp.randn_SE3(5, device=device, dtype=dtype)
        skip_if_nan_inputs(delta, X)
        
        # Native PyPose
        X_native = X.clone()
        X_native.add_(delta)
        
        # Warp backend
        X_warp = to_warp_backend(X.clone())
        delta_warp = to_warp_backend(delta)
        X_warp.add_(delta_warp)
        
        if torch.isnan(X_native.tensor()).any():
            pytest.skip("PyPose expected output contains NaN")
        
        torch.testing.assert_close(X_warp.tensor(), X_native.tensor(), **get_fwd_tolerances(dtype, Operator.SE3_AddExp))


# =============================================================================
# Backward Pass Tests
# =============================================================================


class TestSE3AddExpBwdBatchDimensions:
    """Test SE3_AddExp backward with various batch dimensions."""

    def test_1d_batch(self, device, dtype_bwd):
        """Test backward with 1D batch dimension."""
        delta_data = pp.randn_se3(5, device=device, dtype=dtype_bwd)
        X_data = pp.randn_SE3(5, device=device, dtype=dtype_bwd)
        
        delta_ours = to_warp_backend(delta_data).requires_grad_(True)
        X_ours = to_warp_backend(X_data).requires_grad_(True)
        result_ours = SE3_AddExp.apply(delta_ours, X_ours)
        result_ours.sum().backward()
        
        delta_ref = delta_data.clone().requires_grad_(True)
        X_ref = X_data.clone().requires_grad_(True)
        result_ref = delta_ref.Exp() * X_ref
        result_ref.sum().backward()
        
        assert delta_ours.grad.shape == delta_ref.grad.shape == (5, 6)
        assert X_ours.grad.shape == X_ref.grad.shape == (5, 7)
        torch.testing.assert_close(delta_ours.grad, delta_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AddExp))
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AddExp))

    def test_2d_batch(self, device, dtype_bwd):
        """Test backward with 2D batch dimensions."""
        delta_data = pp.randn_se3(3, 4, device=device, dtype=dtype_bwd)
        X_data = pp.randn_SE3(3, 4, device=device, dtype=dtype_bwd)
        
        delta_ours = to_warp_backend(delta_data).requires_grad_(True)
        X_ours = to_warp_backend(X_data).requires_grad_(True)
        result_ours = SE3_AddExp.apply(delta_ours, X_ours)
        result_ours.sum().backward()
        
        delta_ref = delta_data.clone().requires_grad_(True)
        X_ref = X_data.clone().requires_grad_(True)
        result_ref = delta_ref.Exp() * X_ref
        result_ref.sum().backward()
        
        assert delta_ours.grad.shape == delta_ref.grad.shape == (3, 4, 6)
        assert X_ours.grad.shape == X_ref.grad.shape == (3, 4, 7)
        torch.testing.assert_close(delta_ours.grad, delta_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AddExp))
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AddExp))

    def test_3d_batch(self, device, dtype_bwd):
        """Test backward with 3D batch dimensions."""
        delta_data = pp.randn_se3(2, 3, 4, device=device, dtype=dtype_bwd)
        X_data = pp.randn_SE3(2, 3, 4, device=device, dtype=dtype_bwd)
        
        delta_ours = to_warp_backend(delta_data).requires_grad_(True)
        X_ours = to_warp_backend(X_data).requires_grad_(True)
        result_ours = SE3_AddExp.apply(delta_ours, X_ours)
        result_ours.sum().backward()
        
        delta_ref = delta_data.clone().requires_grad_(True)
        X_ref = X_data.clone().requires_grad_(True)
        result_ref = delta_ref.Exp() * X_ref
        result_ref.sum().backward()
        
        assert delta_ours.grad.shape == delta_ref.grad.shape == (2, 3, 4, 6)
        assert X_ours.grad.shape == X_ref.grad.shape == (2, 3, 4, 7)
        torch.testing.assert_close(delta_ours.grad, delta_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AddExp))
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AddExp))

    def test_4d_batch(self, device, dtype_bwd):
        """Test backward with 4D batch dimensions."""
        delta_data = pp.randn_se3(2, 2, 3, 4, device=device, dtype=dtype_bwd)
        X_data = pp.randn_SE3(2, 2, 3, 4, device=device, dtype=dtype_bwd)
        
        delta_ours = to_warp_backend(delta_data).requires_grad_(True)
        X_ours = to_warp_backend(X_data).requires_grad_(True)
        result_ours = SE3_AddExp.apply(delta_ours, X_ours)
        result_ours.sum().backward()
        
        delta_ref = delta_data.clone().requires_grad_(True)
        X_ref = X_data.clone().requires_grad_(True)
        result_ref = delta_ref.Exp() * X_ref
        result_ref.sum().backward()
        
        assert delta_ours.grad.shape == delta_ref.grad.shape == (2, 2, 3, 4, 6)
        assert X_ours.grad.shape == X_ref.grad.shape == (2, 2, 3, 4, 7)
        torch.testing.assert_close(delta_ours.grad, delta_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AddExp))
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AddExp))

    def test_scalar_no_batch(self, device, dtype_bwd):
        """Test backward with no batch dimensions."""
        delta_data = pp.randn_se3(device=device, dtype=dtype_bwd)
        X_data = pp.randn_SE3(device=device, dtype=dtype_bwd)
        
        delta_ours = to_warp_backend(delta_data).requires_grad_(True)
        X_ours = to_warp_backend(X_data).requires_grad_(True)
        result_ours = SE3_AddExp.apply(delta_ours, X_ours)
        result_ours.sum().backward()
        
        delta_ref = delta_data.clone().requires_grad_(True)
        X_ref = X_data.clone().requires_grad_(True)
        result_ref = delta_ref.Exp() * X_ref
        result_ref.sum().backward()
        
        assert delta_ours.grad.shape == delta_ref.grad.shape == (6,)
        assert X_ours.grad.shape == X_ref.grad.shape == (7,)
        torch.testing.assert_close(delta_ours.grad, delta_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AddExp))
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AddExp))


class TestSE3AddExpBwdPrecision:
    """Test SE3_AddExp backward precision handling."""

    def test_precision(self, device, dtype_bwd):
        """Test backward precision."""
        delta_data = pp.randn_se3(10, device=device, dtype=dtype_bwd)
        X_data = pp.randn_SE3(10, device=device, dtype=dtype_bwd)
        
        delta_ours = to_warp_backend(delta_data).requires_grad_(True)
        X_ours = to_warp_backend(X_data).requires_grad_(True)
        result_ours = SE3_AddExp.apply(delta_ours, X_ours)
        result_ours.sum().backward()
        
        delta_ref = delta_data.clone().requires_grad_(True)
        X_ref = X_data.clone().requires_grad_(True)
        result_ref = delta_ref.Exp() * X_ref
        result_ref.sum().backward()
        
        torch.testing.assert_close(delta_ours.grad, delta_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AddExp))
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AddExp))

    def test_grad_dtype_preserved(self, device, dtype_bwd):
        """Test that gradient dtype matches input dtype."""
        delta = pp.randn_se3(5, device=device, dtype=dtype_bwd)
        X = pp.randn_SE3(5, device=device, dtype=dtype_bwd)
        
        delta_warp = to_warp_backend(delta).requires_grad_(True)
        X_warp = to_warp_backend(X).requires_grad_(True)
        
        result = SE3_AddExp.apply(delta_warp, X_warp)
        result.sum().backward()
        
        assert delta_warp.grad.dtype == dtype_bwd
        assert X_warp.grad.dtype == dtype_bwd


class TestSE3AddExpBwdEdgeCases:
    """Test SE3_AddExp backward edge cases."""

    def test_large_batch_gradient(self, device, dtype_bwd):
        """Test gradient with large batch."""
        delta_data = pp.randn_se3(1000, device=device, dtype=dtype_bwd)
        X_data = pp.randn_SE3(1000, device=device, dtype=dtype_bwd)
        
        delta_ours = to_warp_backend(delta_data).requires_grad_(True)
        X_ours = to_warp_backend(X_data).requires_grad_(True)
        result_ours = SE3_AddExp.apply(delta_ours, X_ours)
        result_ours.sum().backward()
        
        delta_ref = delta_data.clone().requires_grad_(True)
        X_ref = X_data.clone().requires_grad_(True)
        result_ref = delta_ref.Exp() * X_ref
        result_ref.sum().backward()
        
        torch.testing.assert_close(delta_ours.grad, delta_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AddExp))
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AddExp))

    def test_grad_device(self, device, dtype_bwd):
        """Test that gradients are on the correct device."""
        delta = pp.randn_se3(5, device=device, dtype=dtype_bwd)
        X = pp.randn_SE3(5, device=device, dtype=dtype_bwd)
        
        delta_warp = to_warp_backend(delta).requires_grad_(True)
        X_warp = to_warp_backend(X).requires_grad_(True)
        
        result = SE3_AddExp.apply(delta_warp, X_warp)
        result.sum().backward()
        
        assert delta_warp.grad.device == delta_warp.device
        assert X_warp.grad.device == X_warp.device

    def test_zero_delta_gradient(self, device, dtype_bwd):
        """Test backward with zero delta."""
        delta_data = torch.zeros(5, 6, device=device, dtype=dtype_bwd)
        delta_lie = pp.LieTensor(delta_data, ltype=pp.se3_type)
        X_data = pp.randn_SE3(5, device=device, dtype=dtype_bwd)
        
        delta_ours = to_warp_backend(delta_lie).requires_grad_(True)
        X_ours = to_warp_backend(X_data).requires_grad_(True)
        result_ours = SE3_AddExp.apply(delta_ours, X_ours)
        result_ours.sum().backward()
        
        delta_ref = delta_lie.clone().requires_grad_(True)
        X_ref = X_data.clone().requires_grad_(True)
        result_ref = delta_ref.Exp() * X_ref
        result_ref.sum().backward()
        
        # Gradients should be finite
        assert not torch.isnan(delta_ours.grad).any()
        assert not torch.isinf(delta_ours.grad).any()
        assert not torch.isnan(X_ours.grad).any()
        assert not torch.isinf(X_ours.grad).any()
        torch.testing.assert_close(delta_ours.grad, delta_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AddExp))
        torch.testing.assert_close(X_ours.grad, X_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.SE3_AddExp))

