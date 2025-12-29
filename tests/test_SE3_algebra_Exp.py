"""Unit tests for se3_Exp forward and backward."""
import pytest
import torch
import pypose as pp
from pypose_warp import to_warp_backend, to_pypose_backend
from pypose_warp.ltype.SE3_algebra import se3_Exp, se3_Exp_fwd
from conftest import get_fwd_tolerances, get_bwd_tolerances, Operator, skip_if_nan_inputs


# =============================================================================
# Forward Pass Tests
# =============================================================================


class TestSe3ExpBatchDimensions:
    """Test se3_Exp_fwd with various batch dimensions."""

    def test_1d_batch(self, device, dtype):
        """Test with 1D batch dimension."""
        x = pp.randn_se3(5, device=device, dtype=dtype)
        skip_if_nan_inputs(x)
        x_warp = to_warp_backend(x)
        
        result = se3_Exp_fwd(x_warp)
        expected = x.Exp()
        
        assert result.shape == expected.shape == (5, 7)
        assert to_pypose_backend(result).ltype == expected.ltype == pp.SE3_type
        assert result.dtype == dtype
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.se3_Exp))

    def test_2d_batch(self, device, dtype):
        """Test with 2D batch dimensions."""
        x = pp.randn_se3(3, 4, device=device, dtype=dtype)
        skip_if_nan_inputs(x)
        x_warp = to_warp_backend(x)
        
        result = se3_Exp_fwd(x_warp)
        expected = x.Exp()
        
        assert result.shape == expected.shape == (3, 4, 7)
        assert to_pypose_backend(result).ltype == pp.SE3_type
        assert result.dtype == dtype
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.se3_Exp))

    def test_3d_batch(self, device, dtype):
        """Test with 3D batch dimensions."""
        x = pp.randn_se3(2, 3, 4, device=device, dtype=dtype)
        skip_if_nan_inputs(x)
        x_warp = to_warp_backend(x)
        
        result = se3_Exp_fwd(x_warp)
        expected = x.Exp()
        
        assert result.shape == expected.shape == (2, 3, 4, 7)
        assert to_pypose_backend(result).ltype == pp.SE3_type
        assert result.dtype == dtype
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.se3_Exp))

    def test_4d_batch(self, device, dtype):
        """Test with 4D batch dimensions."""
        x = pp.randn_se3(2, 2, 3, 4, device=device, dtype=dtype)
        skip_if_nan_inputs(x)
        x_warp = to_warp_backend(x)
        
        result = se3_Exp_fwd(x_warp)
        expected = x.Exp()
        
        # Skip if expected has NaN (PyPose FP16 numerical instability)
        if torch.isnan(expected.tensor()).any():
            pytest.skip("PyPose expected output contains NaN (FP16 numerical instability)")
        
        assert result.shape == expected.shape == (2, 2, 3, 4, 7)
        assert to_pypose_backend(result).ltype == pp.SE3_type
        assert result.dtype == dtype
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.se3_Exp))

    def test_scalar_no_batch(self, device, dtype):
        """Test with no batch dimensions (single element)."""
        x = pp.randn_se3(device=device, dtype=dtype)
        skip_if_nan_inputs(x)
        x_warp = to_warp_backend(x)
        
        result = se3_Exp_fwd(x_warp)
        expected = x.Exp()
        
        assert result.shape == expected.shape == (7,)
        assert to_pypose_backend(result).ltype == pp.SE3_type
        assert result.dtype == dtype
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.se3_Exp))


class TestSe3ExpPrecision:
    """Test se3_Exp_fwd precision handling."""

    def test_fp32_precision(self, device):
        """Test float32 precision and accuracy."""
        dtype = torch.float32
        x = pp.randn_se3(10, device=device, dtype=dtype)
        x_warp = to_warp_backend(x)
        
        result = se3_Exp_fwd(x_warp)
        expected = x.Exp()
        
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.se3_Exp))

    def test_fp64_precision(self, device):
        """Test float64 precision and accuracy."""
        dtype = torch.float64
        x = pp.randn_se3(10, device=device, dtype=dtype)
        x_warp = to_warp_backend(x)
        
        result = se3_Exp_fwd(x_warp)
        expected = x.Exp()
        
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.se3_Exp))

    def test_output_dtype_preserved(self, device, dtype):
        """Test that output dtype matches input dtype."""
        x = pp.randn_se3(5, device=device, dtype=dtype)
        skip_if_nan_inputs(x)
        x_warp = to_warp_backend(x)
        
        result = se3_Exp_fwd(x_warp)
        
        assert result.dtype == dtype


class TestSe3ExpEdgeCases:
    """Test se3_Exp_fwd edge cases."""

    def test_zero_input(self, device, dtype):
        """Test that zero input maps to identity SE3."""
        x = torch.zeros(5, 6, device=device, dtype=dtype)
        x_lie = pp.LieTensor(x, ltype=pp.se3_type)
        x_warp = to_warp_backend(x_lie)
        
        result = se3_Exp_fwd(x_warp)
        expected = x_lie.Exp()
        
        # Exp of zero should be identity SE3: [0,0,0, 0,0,0,1]
        identity = torch.zeros(5, 7, device=device, dtype=dtype)
        identity[:, 6] = 1.0  # qw = 1
        
        torch.testing.assert_close(result.tensor(), identity, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.se3_Exp))

    def test_pure_translation(self, device):
        """Test with pure translation (zero rotation)."""
        dtype = torch.float64
        # se3: [tau_x, tau_y, tau_z, 0, 0, 0]
        x = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype)
        x_lie = pp.LieTensor(x, ltype=pp.se3_type)
        x_warp = to_warp_backend(x_lie)
        
        result = se3_Exp_fwd(x_warp)
        expected = x_lie.Exp()
        
        # For pure translation with zero rotation, t = Jl(0) @ tau = I @ tau = tau
        expected_se3 = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]], device=device, dtype=dtype)
        torch.testing.assert_close(result.tensor(), expected_se3, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(result.tensor(), expected.tensor(), atol=1e-10, rtol=1e-10)

    def test_pure_rotation(self, device):
        """Test with pure rotation (zero translation)."""
        dtype = torch.float64
        import math
        # se3: [0, 0, 0, phi_x, phi_y, phi_z] - 90 deg rotation around z
        phi = math.pi / 2
        x = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, phi]], device=device, dtype=dtype)
        x_lie = pp.LieTensor(x, ltype=pp.se3_type)
        x_warp = to_warp_backend(x_lie)
        
        result = se3_Exp_fwd(x_warp)
        expected = x_lie.Exp()
        
        torch.testing.assert_close(result.tensor(), expected.tensor(), atol=1e-10, rtol=1e-10)

    def test_single_element_batch(self, device, dtype):
        """Test with batch size of 1."""
        x = pp.randn_se3(1, device=device, dtype=dtype)
        skip_if_nan_inputs(x)
        x_warp = to_warp_backend(x)
        
        result = se3_Exp_fwd(x_warp)
        expected = x.Exp()
        
        assert result.shape == (1, 7)
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.se3_Exp))

    def test_large_batch(self, device, dtype):
        """Test with large batch size."""
        x = pp.randn_se3(1000, device=device, dtype=dtype)
        skip_if_nan_inputs(x)
        x_warp = to_warp_backend(x)
        
        result = se3_Exp_fwd(x_warp)
        expected = x.Exp()
        
        # Skip if expected has NaN (PyPose FP16 numerical instability)
        if torch.isnan(expected.tensor()).any():
            pytest.skip("PyPose expected output contains NaN (FP16 numerical instability)")
        
        assert result.shape == (1000, 7)
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.se3_Exp))

    def test_output_device(self, device, dtype):
        """Test that output is on the same device as input."""
        x = pp.randn_se3(5, device=device, dtype=dtype)
        skip_if_nan_inputs(x)
        x_warp = to_warp_backend(x)
        
        result = se3_Exp_fwd(x_warp)
        
        assert result.device == x.device

    def test_output_ltype(self, device, dtype):
        """Test that output is SE3 LieTensor."""
        x = pp.randn_se3(5, device=device, dtype=dtype)
        skip_if_nan_inputs(x)
        x_warp = to_warp_backend(x)
        
        result = se3_Exp_fwd(x_warp)
        
        assert isinstance(result, pp.LieTensor)
        assert to_pypose_backend(result).ltype == pp.SE3_type


class TestSe3ExpErrors:
    """Test se3_Exp_fwd error handling."""

    def test_5d_batch_raises(self, device, dtype):
        """Test that 5D batch dimensions raise NotImplementedError."""
        x = pp.randn_se3(2, 2, 2, 2, 2, device=device, dtype=dtype)
        x_warp = to_warp_backend(x)
        
        with pytest.raises(NotImplementedError, match="Batch dimensions > 4"):
            se3_Exp_fwd(x_warp)


class TestSe3ExpAutogradFunction:
    """Test se3_Exp autograd function wrapper."""

    def test_apply_matches_fwd(self, device, dtype):
        """Test that se3_Exp.apply matches se3_Exp_fwd."""
        x = pp.randn_se3(5, device=device, dtype=dtype)
        skip_if_nan_inputs(x)
        x_warp = to_warp_backend(x)
        
        result_apply = se3_Exp.apply(x_warp)
        result_fwd = se3_Exp_fwd(x_warp)
        
        torch.testing.assert_close(result_apply.tensor(), result_fwd.tensor())

    def test_apply_1d_batch(self, device, dtype):
        """Test se3_Exp.apply with 1D batch."""
        x = pp.randn_se3(5, device=device, dtype=dtype)
        skip_if_nan_inputs(x)
        x_warp = to_warp_backend(x)
        
        result = se3_Exp.apply(x_warp)
        expected = x.Exp()
        
        assert result.shape == expected.shape == (5, 7)
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.se3_Exp))


class TestSe3ExpWarpBackend:
    """Test se3_Exp through the warp backend interface."""

    def test_warp_backend_exp(self, device, dtype):
        """Test Exp through warp backend LieType."""
        x = pp.randn_se3(5, device=device, dtype=dtype)
        skip_if_nan_inputs(x)
        x_warp = to_warp_backend(x)
        
        result = x_warp.Exp()
        expected = x.Exp()
        
        assert result.shape == expected.shape == (5, 7)
        assert to_pypose_backend(result).ltype == pp.SE3_type
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.se3_Exp))

    def test_warp_backend_exp_2d(self, device, dtype):
        """Test Exp through warp backend with 2D batch."""
        x = pp.randn_se3(3, 4, device=device, dtype=dtype)
        skip_if_nan_inputs(x)
        x_warp = to_warp_backend(x)
        
        result = x_warp.Exp()
        expected = x.Exp()
        
        assert result.shape == expected.shape == (3, 4, 7)
        torch.testing.assert_close(result.tensor(), expected.tensor(), **get_fwd_tolerances(dtype, Operator.se3_Exp))


# =============================================================================
# Backward Pass Tests
# =============================================================================


class TestSe3ExpBwdBatchDimensions:
    """Test se3_Exp backward with various batch dimensions."""

    def test_1d_batch(self, device, dtype_bwd):
        """Test backward with 1D batch dimension."""
        x_data = pp.randn_se3(5, device=device, dtype=dtype_bwd)
        
        x_ours = to_warp_backend(x_data).requires_grad_(True)
        result_ours = se3_Exp.apply(x_ours)
        result_ours.sum().backward()
        
        x_ref = x_data.clone().requires_grad_(True)
        result_ref = x_ref.Exp()
        result_ref.sum().backward()
        
        assert x_ours.grad.shape == x_ref.grad.shape == (5, 6)
        torch.testing.assert_close(x_ours.grad, x_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.se3_Exp))

    def test_2d_batch(self, device, dtype_bwd):
        """Test backward with 2D batch dimensions."""
        x_data = pp.randn_se3(3, 4, device=device, dtype=dtype_bwd)
        
        x_ours = to_warp_backend(x_data).requires_grad_(True)
        result_ours = se3_Exp.apply(x_ours)
        result_ours.sum().backward()
        
        x_ref = x_data.clone().requires_grad_(True)
        result_ref = x_ref.Exp()
        result_ref.sum().backward()
        
        assert x_ours.grad.shape == x_ref.grad.shape == (3, 4, 6)
        torch.testing.assert_close(x_ours.grad, x_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.se3_Exp))

    def test_3d_batch(self, device, dtype_bwd):
        """Test backward with 3D batch dimensions."""
        x_data = pp.randn_se3(2, 3, 4, device=device, dtype=dtype_bwd)
        
        x_ours = to_warp_backend(x_data).requires_grad_(True)
        result_ours = se3_Exp.apply(x_ours)
        result_ours.sum().backward()
        
        x_ref = x_data.clone().requires_grad_(True)
        result_ref = x_ref.Exp()
        result_ref.sum().backward()
        
        assert x_ours.grad.shape == x_ref.grad.shape == (2, 3, 4, 6)
        torch.testing.assert_close(x_ours.grad, x_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.se3_Exp))

    def test_4d_batch(self, device, dtype_bwd):
        """Test backward with 4D batch dimensions."""
        x_data = pp.randn_se3(2, 2, 3, 4, device=device, dtype=dtype_bwd)
        
        x_ours = to_warp_backend(x_data).requires_grad_(True)
        result_ours = se3_Exp.apply(x_ours)
        result_ours.sum().backward()
        
        x_ref = x_data.clone().requires_grad_(True)
        result_ref = x_ref.Exp()
        result_ref.sum().backward()
        
        assert x_ours.grad.shape == x_ref.grad.shape == (2, 2, 3, 4, 6)
        torch.testing.assert_close(x_ours.grad, x_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.se3_Exp))

    def test_scalar_no_batch(self, device, dtype_bwd):
        """Test backward with no batch dimensions."""
        x_data = pp.randn_se3(device=device, dtype=dtype_bwd)
        
        x_ours = to_warp_backend(x_data).requires_grad_(True)
        result_ours = se3_Exp.apply(x_ours)
        result_ours.sum().backward()
        
        x_ref = x_data.clone().requires_grad_(True)
        result_ref = x_ref.Exp()
        result_ref.sum().backward()
        
        assert x_ours.grad.shape == x_ref.grad.shape == (6,)
        torch.testing.assert_close(x_ours.grad, x_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.se3_Exp))


class TestSe3ExpBwdPrecision:
    """Test se3_Exp backward precision handling."""

    def test_precision(self, device, dtype_bwd):
        """Test backward precision."""
        x_data = pp.randn_se3(10, device=device, dtype=dtype_bwd)
        
        x_ours = to_warp_backend(x_data).requires_grad_(True)
        result_ours = se3_Exp.apply(x_ours)
        result_ours.sum().backward()
        
        x_ref = x_data.clone().requires_grad_(True)
        result_ref = x_ref.Exp()
        result_ref.sum().backward()
        
        torch.testing.assert_close(x_ours.grad, x_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.se3_Exp))

    def test_grad_dtype_preserved(self, device, dtype_bwd):
        """Test that gradient dtype matches input dtype."""
        x = pp.randn_se3(5, device=device, dtype=dtype_bwd)
        x_warp = to_warp_backend(x).requires_grad_(True)
        
        result = se3_Exp.apply(x_warp)
        result.sum().backward()
        
        assert x_warp.grad.dtype == dtype_bwd


class TestSe3ExpBwdEdgeCases:
    """Test se3_Exp backward edge cases."""

    def test_large_batch_gradient(self, device, dtype_bwd):
        """Test gradient with large batch."""
        x_data = pp.randn_se3(1000, device=device, dtype=dtype_bwd)
        
        x_ours = to_warp_backend(x_data).requires_grad_(True)
        result_ours = se3_Exp.apply(x_ours)
        result_ours.sum().backward()
        
        x_ref = x_data.clone().requires_grad_(True)
        result_ref = x_ref.Exp()
        result_ref.sum().backward()
        
        torch.testing.assert_close(x_ours.grad, x_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.se3_Exp))

    def test_grad_device(self, device, dtype_bwd):
        """Test that gradients are on the correct device."""
        x = pp.randn_se3(5, device=device, dtype=dtype_bwd)
        x_warp = to_warp_backend(x).requires_grad_(True)
        
        result = se3_Exp.apply(x_warp)
        result.sum().backward()
        
        assert x_warp.grad.device == x_warp.device

    def test_zero_input_gradient(self, device, dtype_bwd):
        """Test backward with zero input."""
        x_data = torch.zeros(5, 6, device=device, dtype=dtype_bwd)
        x_lie = pp.LieTensor(x_data, ltype=pp.se3_type)
        
        x_ours = to_warp_backend(x_lie).requires_grad_(True)
        result_ours = se3_Exp.apply(x_ours)
        result_ours.sum().backward()
        
        x_ref = x_lie.clone().requires_grad_(True)
        result_ref = x_ref.Exp()
        result_ref.sum().backward()
        
        # Gradients should be finite
        assert not torch.isnan(x_ours.grad).any()
        assert not torch.isinf(x_ours.grad).any()
        torch.testing.assert_close(x_ours.grad, x_ref.grad, **get_bwd_tolerances(dtype_bwd, Operator.se3_Exp))

