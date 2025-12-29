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


def wp_vec6_type(dtype: torch.dtype):
    """Warp vector type for 6D vectors (se3 twists, etc.)."""
    match dtype:
        case torch.float64: return wp.types.vector(length=6, dtype=wp.float64)
        case torch.float32: return wp.types.vector(length=6, dtype=wp.float32)
        case torch.float16: return wp.types.vector(length=6, dtype=wp.float16)
        case _: raise NotImplementedError()


def wp_mat33_type(dtype: torch.dtype):
    match dtype:
        case torch.float64: return wp.mat33d
        case torch.float32: return wp.mat33f
        case torch.float16: return wp.mat33h
        case _: raise NotImplementedError()


def wp_mat44_type(dtype: torch.dtype):
    match dtype:
        case torch.float64: return wp.mat44d
        case torch.float32: return wp.mat44f
        case torch.float16: return wp.mat44h
        case _: raise NotImplementedError()


def wp_transform_type(dtype: torch.dtype):
    """Warp transform type for SE3 (3D translation + quaternion rotation)."""
    match dtype:
        case torch.float64: return wp.transformd
        case torch.float32: return wp.transformf
        case torch.float16: return wp.transformh
        case _: raise NotImplementedError()

