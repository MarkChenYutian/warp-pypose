import torch
import warp as wp


def wp_quat_type(dtype: torch.dtype):
    match dtype:
        case torch.float64: return wp.quatd
        case torch.float32: return wp.quatf
        case torch.float16: return wp.quath
        case _: raise NotImplementedError()


def wp_vec3_type(dtype: torch.dtype):
    match dtype:
        case torch.float64: return wp.vec3d
        case torch.float32: return wp.vec3f
        case torch.float16: return wp.vec3h
        case _: raise NotImplementedError()


def wp_vec4_type(dtype: torch.dtype):
    match dtype:
        case torch.float64: return wp.vec4d
        case torch.float32: return wp.vec4f
        case torch.float16: return wp.vec4h
        case _: raise NotImplementedError()


def wp_mat33_type(dtype: torch.dtype):
    match dtype:
        case torch.float64: return wp.mat33d
        case torch.float32: return wp.mat33f
        case torch.float16: return wp.mat33h
        case _: raise NotImplementedError()


def wp_transform_type(dtype: torch.dtype):
    """Warp transform type for SE3 (3D translation + quaternion rotation)."""
    match dtype:
        case torch.float64: return wp.transformd
        case torch.float32: return wp.transformf
        case torch.float16: return wp.transformh
        case _: raise NotImplementedError()

