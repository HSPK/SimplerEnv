"""
Rotation conversion utilities. Only support numpy arrays because using of scipy.
Reference:
https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
"""

from functools import cache

import numpy as np
from scipy.spatial.transform import Rotation as R


@cache
def get_rotation_matrix(new_axes: tuple[str, str, str]):
    axis_idx = {"x": 0, "y": 1, "z": 2}
    # 构造旋转矩阵 (3, 3)
    return np.stack(
        [
            np.eye(3)[axis_idx[ax.lstrip("-")]] * (-1 if ax.startswith("-") else 1)
            for ax in new_axes
        ]
    )


def transform_to_xyz(new_axes: tuple[str, str, str], pos: np.ndarray) -> np.ndarray:
    """
    支持任意形状的 pos（最后一维为3），将新坐标系下的向量变换到 xyz 坐标系。
    """
    rot = get_rotation_matrix(new_axes).astype(pos.dtype)
    # 广播矩阵乘法，适配任意形状
    return np.matmul(pos, rot.T)


def rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-2).swapaxes(-2, -1)


def matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)
    """
    batch_dim = matrix.shape[:-2]
    # use first two columns instead of first two rows(as in the original code)
    return np.swapaxes(matrix[..., :, :2], -2, -1).copy().reshape(batch_dim + (6,))


def euler_to_matrix(euler: np.ndarray) -> np.ndarray:
    """
    Converts Euler angles (roll, pitch, yaw) to a 3x3 rotation matrix.
    The (roll, pitch, yaw) corresponds to `Rotation.as_euler("xyz")` convention.
    """
    batch_dim = euler.shape[:-1]
    x = euler.reshape(-1, 3)
    x = R.from_euler("xyz", x, degrees=False)
    x = x.as_matrix()
    x = x.reshape(batch_dim + (3, 3))
    return x


def matrix_to_euler(matrix: np.ndarray) -> np.ndarray:
    """
    Converts a 3x3 rotation matrix to Euler angles (roll, pitch, yaw).
    The (roll, pitch, yaw) corresponds to `Rotation.as_euler("xyz")` convention.
    """
    batch_dim = matrix.shape[:-2]
    x = matrix.reshape(-1, 3, 3)
    x = R.from_matrix(x)
    x = x.as_euler("xyz", degrees=False)
    x = x.reshape(batch_dim + (3,))
    return x


def axis_angle_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    batch_dim = rotvec.shape[:-1]
    x = rotvec.reshape(-1, 3)
    x = R.from_rotvec(x)
    x = x.as_matrix()
    x = x.reshape(batch_dim + (3, 3))
    return x


def axis_angle_to_quaternion(rotvec: np.ndarray) -> np.ndarray:
    batch_dim = rotvec.shape[:-1]
    x = rotvec.reshape(-1, 3)
    x = R.from_rotvec(x)
    x = x.as_quat()
    x = x.reshape(batch_dim + (4,))
    return x


def quaternion_to_matrix(quat: np.ndarray) -> np.ndarray:
    batch_dim = quat.shape[:-1]
    x = quat.reshape(-1, 4)
    x = R.from_quat(x)
    x = x.as_matrix()
    x = x.reshape(batch_dim + (3, 3))
    return x


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    batch_dim = matrix.shape[:-2]
    x = matrix.reshape(-1, 3, 3)
    x = R.from_matrix(x)
    x = x.as_quat()
    x = x.reshape(batch_dim + (4,))
    return make_unique_quaternion(x)


def euler_to_axis_angle(euler: np.ndarray) -> np.ndarray:
    """
    Converts Euler angles (roll, pitch, yaw) to axis-angle representation.
    The (roll, pitch, yaw) corresponds to `Rotation.as_euler("xyz")` convention.
    """
    batch_dim = euler.shape[:-1]
    x = euler.reshape(-1, 3)
    x = R.from_euler("xyz", x, degrees=False)
    x = x.as_rotvec()
    x = x.reshape(batch_dim + (3,))
    return x


def axangle_to_euler(rotvec: np.ndarray) -> np.ndarray:
    """
    Converts axis-angle representation to Euler angles (roll, pitch, yaw).
    The (roll, pitch, yaw) corresponds to `Rotation.as_euler("xyz")` convention.
    """
    batch_dim = rotvec.shape[:-1]
    x = rotvec.reshape(-1, 3)
    x = R.from_rotvec(x)
    x = x.as_euler("xyz", degrees=False)
    x = x.reshape(batch_dim + (3,))
    return x


def rotation_6d_to_axis_angle(d6: np.ndarray) -> np.ndarray:
    return matrix_to_axis_angle(rotation_6d_to_matrix(d6))


def axangle_to_quaternion(rotvec: np.ndarray) -> np.ndarray:
    """
    Converts axis-angle representation to quaternion.
    The axis-angle is expected in the format [x, y, z].
    """
    batch_dim = rotvec.shape[:-1]
    x = rotvec.reshape(-1, 3)
    x = R.from_rotvec(x)
    x = x.as_quat()
    x = x.reshape(batch_dim + (4,))
    return make_unique_quaternion(x)


def axangle_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    """
    Converts axis-angle representation to a rotation matrix.
    The axis-angle is expected in the format [x, y, z].
    """
    batch_dim = rotvec.shape[:-1]
    x = rotvec.reshape(-1, 3)
    x = R.from_rotvec(x)
    x = x.as_matrix()
    x = x.reshape(batch_dim + (3, 3))
    return x


def axangle_to_rotation_6d(rotvec: np.ndarray) -> np.ndarray:
    """
    Converts axis-angle representation to 6D rotation representation.
    The axis-angle is expected in the format [x, y, z].
    """
    return matrix_to_rotation_6d(axangle_to_matrix(rotvec))


def quaternion_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """
    Converts quaternion to axis-angle representation.
    The quaternion is expected in the format [x, y, z, w].
    """
    batch_dim = quat.shape[:-1]
    x = quat.reshape(-1, 4)
    x = R.from_quat(x)
    x = x.as_rotvec()
    x = x.reshape(batch_dim + (3,))
    return x


def axis_angle_to_euler(rotvec: np.ndarray) -> np.ndarray:
    """
    Converts axis-angle representation to Euler angles (roll, pitch, yaw).
    The (roll, pitch, yaw) corresponds to `Rotation.as_euler("xyz")` convention.
    """
    batch_dim = rotvec.shape[:-1]
    x = rotvec.reshape(-1, 3)
    x = R.from_rotvec(x)
    x = x.as_euler("xyz", degrees=False)
    x = x.reshape(batch_dim + (3,))
    return x


def matrix_to_axis_angle(matrix: np.ndarray) -> np.ndarray:
    batch_dim = matrix.shape[:-2]
    x = matrix.reshape(-1, 3, 3)
    x = R.from_matrix(x)
    x = x.as_rotvec()
    x = x.reshape(batch_dim + (3,))
    return x


def make_unique_quaternion(quat: np.ndarray, scaler_first: bool = False) -> np.ndarray:
    """
    Make quaternion unique by ensuring the first element is non-negative.
    """
    if scaler_first:
        mask = quat[..., 0:1] < 0
    else:
        mask = quat[..., -1:] < 0
    return np.where(mask, -quat, quat)


def euler_to_quaternion(euler: np.ndarray) -> np.ndarray:
    """
    Converts Euler angles (roll, pitch, yaw) to a quaternion.
    The (roll, pitch, yaw) corresponds to `Rotation.as_euler("xyz")` convention.
    """
    batch_dim = euler.shape[:-1]
    x = euler.reshape(-1, 3)
    x = R.from_euler("xyz", x, degrees=False)
    x = x.as_quat()
    x = x.reshape(batch_dim + (4,))
    return make_unique_quaternion(x)


def quaternion_to_euler(quat: np.ndarray) -> np.ndarray:
    batch_dim = quat.shape[:-1]
    x = quat.reshape(-1, 4)
    x = R.from_quat(x)
    x = x.as_euler("xyz", degrees=False)
    x = x.reshape(batch_dim + (3,))
    return x


def quat_inv(quat: np.ndarray) -> np.ndarray:
    return np.concatenate([-quat[..., :3], quat[..., 3:]], axis=-1)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    四元数乘法：q1 * q2，输入格式为 [x, y, z, w]
    支持任意 shape 的 batch 维度
    """
    x1, y1, z1, w1 = np.split(q1, 4, axis=-1)
    x2, y2, z2, w2 = np.split(q2, 4, axis=-1)

    return np.concatenate(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
            w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        axis=-1,
    )


def solve_xq1_q2(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    解方程 x * q1 = q2, x = q2 * q1-1
    适用于单位四元数
    """
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    q2 = q2 / np.linalg.norm(q2, axis=-1, keepdims=True)
    return make_unique_quaternion(quat_mul(q2, quat_inv(q1)))


def euler_to_rotation_6d(euler: np.ndarray) -> np.ndarray:
    return matrix_to_rotation_6d(euler_to_matrix(euler))


def quaternion_to_rotation_6d(quat: np.ndarray) -> np.ndarray:
    return matrix_to_rotation_6d(quaternion_to_matrix(quat))
