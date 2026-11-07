# Copyright (c) 2026 西安交通大学
# SPDX-License-Identifier: GPL-3.0-or-later

"""Rigid coordinate transformations for CSG geometry."""

from dataclasses import dataclass

import numpy as np


def rotation_x(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ]
    )


def rotation_y(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ]
    )


def rotation_z(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def rotation_about_axis(axis, angle):
    if axis == 0:
        return np.eye(3)
    if axis == 1:
        return rotation_x(angle)
    if axis == 2:
        return rotation_y(angle)
    if axis == 3:
        return rotation_z(angle)
    raise ValueError(f"invalid rotation axis: {axis}")


def _orthonormalize(matrix):
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


@dataclass(frozen=True)
class Transform:
    """Affine rigid transform: global = rotation @ local + translation."""

    rotation: np.ndarray
    translation: np.ndarray
    number: int | None = None
    mode: int | None = None

    def __post_init__(self):
        rotation = np.asarray(self.rotation, dtype=float)
        translation = np.asarray(self.translation, dtype=float)

        if rotation.shape != (3, 3):
            raise ValueError("rotation must be a 3x3 matrix")
        if translation.shape != (3,):
            raise ValueError("translation must be a length-3 vector")

        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    def local_to_global(self, points):
        points = np.asarray(points, dtype=float)
        return points @ self.rotation.T + self.translation

    def global_to_local(self, points):
        points = np.asarray(points, dtype=float)
        return (points - self.translation) @ self.rotation

    def local_to_global_xyz(self, x, y, z):
        x, y, z = np.broadcast_arrays(x, y, z)
        points = np.stack([x, y, z], axis=-1)
        result = self.local_to_global(points)
        return result[..., 0], result[..., 1], result[..., 2]

    def global_to_local_xyz(self, x, y, z):
        x, y, z = np.broadcast_arrays(x, y, z)
        points = np.stack([x, y, z], axis=-1)
        result = self.global_to_local(points)
        return result[..., 0], result[..., 1], result[..., 2]

    def inverse(self):
        rotation = self.rotation.T
        translation = -rotation @ self.translation
        return Transform(
            rotation=rotation,
            translation=translation,
            number=self.number,
            mode=self.mode,
        )

    def compose(self, other):
        """Apply other first, then self."""
        rotation = self.rotation @ other.rotation
        translation = self.rotation @ other.translation + self.translation
        return Transform(rotation=rotation, translation=translation)

    @classmethod
    def identity(cls):
        return cls(np.eye(3), np.zeros(3))

    @classmethod
    def from_phits(cls, number, values, mode, degrees=False):
        values = np.asarray(values, dtype=float)
        if values.shape != (12,):
            raise ValueError("PHITS transform needs 12 values before M")

        translation = values[:3]
        rotation = None
        offset = translation

        if abs(mode) == 1:
            b = values[3:12]
            rotation = np.array(
                [
                    [b[0], b[3], b[6]],
                    [b[1], b[4], b[7]],
                    [b[2], b[5], b[8]],
                ]
            )
            rotation = _orthonormalize(rotation)
            if mode < 0:
                offset = -rotation @ translation

        elif abs(mode) == 2:
            theta_z, theta_y, theta_x = values[3:6]
            if degrees:
                theta_z, theta_y, theta_x = np.deg2rad([theta_z, theta_y, theta_x])
            rotation = rotation_x(theta_x) @ rotation_y(theta_y) @ rotation_z(theta_z)
            if mode < 0:
                offset = -rotation @ translation

        elif abs(mode) == 3:
            center = values[3:6]
            axis_1, theta_1, axis_2, theta_2, axis_3, theta_3 = values[6:12]
            if degrees:
                theta_1, theta_2, theta_3 = np.deg2rad([theta_1, theta_2, theta_3])
            rotation = (
                rotation_about_axis(int(axis_3), theta_3)
                @ rotation_about_axis(int(axis_2), theta_2)
                @ rotation_about_axis(int(axis_1), theta_1)
            )
            offset = center - rotation @ center + translation
            if mode < 0:
                offset = center - rotation @ center - rotation @ translation

        else:
            raise ValueError(f"invalid PHITS transform mode: {mode}")

        return cls(
            rotation=rotation,
            translation=offset,
            number=number,
            mode=mode,
        )

    @classmethod
    def from_mcnp(cls, number, values):
        values = np.asarray(values, dtype=float)
        if values.shape != (12,):
            raise ValueError("MCNP transform needs 12 values")

        translation = values[:3]
        rotation = values[3:12].reshape(3, 3)
        rotation = _orthonormalize(rotation)
        return cls(
            rotation=rotation,
            translation=translation,
            number=number,
            mode=1,
        )
