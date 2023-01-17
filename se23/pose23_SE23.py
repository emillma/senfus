# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

import symforce.internal.symbolic as sf
from symforce import typing as T
from symforce.ops.interfaces.lie_group import LieGroup

from symforce.geo.matrix import Matrix, Matrix55, Matrix31, Vector3
from symforce.geo import Rot3
from .pose23 import Pose23


class Pose23_SE23(Pose23):
    """
    TODO
    """

    @classmethod
    def from_tangent(
        cls, vec: T.Sequence[T.Scalar], epsilon: T.Scalar = sf.epsilon()
    ) -> Pose23_SE23:
        R_tangent = (vec[0], vec[1], vec[2])
        t_tangent_vector = Vector3(vec[3], vec[4], vec[5])
        v_tancent_vector = Vector3(vec[6], vec[7], vec[8])

        R = Rot3.from_tangent(R_tangent, epsilon=epsilon)
        R_hat = Rot3.hat(R_tangent)
        R_hat_sq = R_hat * R_hat
        R_tangent_vector = Vector3(R_tangent)
        theta = sf.sqrt(R_tangent_vector.squared_norm() + epsilon**2)

        V = (
            Matrix.eye(3)
            + (1 - sf.cos(theta)) / (theta**2) * R_hat
            + (theta - sf.sin(theta)) / (theta**3) * R_hat_sq
        )
        return cls(R, V * t_tangent_vector, V * v_tancent_vector)

    def to_tangent(self, epsilon: T.Scalar = sf.epsilon()) -> T.List[T.Scalar]:
        R_tangent = self.R.to_tangent(epsilon=epsilon)
        R_tangent_vector = Vector3(R_tangent)

        theta = sf.sqrt(R_tangent_vector.squared_norm() + epsilon)
        R_hat = Rot3.hat(R_tangent)

        half_theta = 0.5 * theta

        V_inv = (
            Matrix.eye(3)
            - 0.5 * R_hat
            + (1 - theta * sf.cos(half_theta) / (2 * sf.sin(half_theta)))
            / (theta * theta)
            * (R_hat * R_hat)
        )
        t_tangent = V_inv * self.t
        v_tangent = V_inv * self.v
        return R_tangent_vector.col_join(v_tangent).col_join(t_tangent).to_flat_list()

    def storage_D_tangent(self) -> Matrix:
        """
        Note: generated from symforce/notebooks/storage_D_tangent.ipynb
        """
        storage_D_tangent_R = self.R.storage_D_tangent()
        storage_D_tangent_v = self.R.to_rotation_matrix()
        storage_D_tangent_t = self.R.to_rotation_matrix()
        return Matrix.block_matrix(
            [
                [storage_D_tangent_R, Matrix.zeros(4, 6)],
                [Matrix.zeros(3, 3), storage_D_tangent_v, Matrix.zeros(3, 3)],
                [Matrix.zeros(3, 6), storage_D_tangent_t],
            ]
        )

    def tangent_D_storage(self) -> Matrix:
        """
        Note: generated from symforce/notebooks/tangent_D_storage.ipynb
        """
        tangent_D_storage_R = self.R.tangent_D_storage()
        tangent_D_storage_v = self.R.to_rotation_matrix().T
        tangent_D_storage_t = self.R.to_rotation_matrix().T
        return Matrix.block_matrix(
            [
                [tangent_D_storage_R, Matrix.zeros(3, 6)],
                [Matrix.zeros(3, 4), tangent_D_storage_v, Matrix.zeros(3, 3)],
                [Matrix.zeros(3, 7), tangent_D_storage_t],
            ]
        )

    def retract(
        self, vec: T.Sequence[T.Scalar], epsilon: T.Scalar = sf.epsilon()
    ) -> Pose23_SE23:
        return LieGroup.retract(self, vec, epsilon)

    def local_coordinates(
        self, b: Pose23_SE23, epsilon: T.Scalar = sf.epsilon()
    ) -> T.List[T.Scalar]:
        return LieGroup.local_coordinates(self, b, epsilon)

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------
