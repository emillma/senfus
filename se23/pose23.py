# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------
"""Some info
"""
from __future__ import annotations

import symforce.internal.symbolic as sf
from symforce import ops
from symforce import typing as T
from symforce.ops.interfaces import LieGroup

from symforce.geo.matrix import Matrix, Matrix33, Matrix55, Matrix99, Vector3, Matrix31
from symforce.geo import Rot3


class Pose23(LieGroup):
    """
    TODO
    """

    Pose23T = T.TypeVar("Pose23T", bound="Pose23")

    def __init__(self, R: Rot3 = None, v: Vector3 = None, t: Vector3 = None) -> None:
        """
        Construct from elements in SO3 and R3.

        Args:
            R: Frame orientation
            t: Translation 3-vector in the global frame
        """
        self.R = R or Rot3()
        self.v = v or Vector3()
        self.t = t or Vector3()

        assert isinstance(self.R, Rot3)
        assert isinstance(self.v, Vector3)
        assert isinstance(self.t, Vector3)
        assert self.v.shape == (3, 1), self.t.shape
        assert self.t.shape == (3, 1), self.t.shape

    def rotation(self) -> Rot3:
        """
        Accessor for the rotation component

        Does not make a copy.  Also accessible as `self.R`
        """
        return self.R

    def position(self) -> Vector3:
        """
        Accessor for the position component

        Does not make a copy.  Also accessible as `self.t`
        """
        return self.t

    def velocity(self) -> Vector3:
        """
        Accessor for the position component

        Does not make a copy.  Also accessible as `self.v`
        """
        return self.v

    # -------------------------------------------------------------------------
    # Storage concept - see symforce.ops.storage_ops
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return "<{} R={}, v=({},{},{}), t=({}, {}, {})>".format(
            self.__class__.__name__,
            repr(self.R),
            repr(self.v[0]),
            repr(self.v[1]),
            repr(self.v[2]),
            repr(self.t[0]),
            repr(self.t[1]),
            repr(self.t[2]),
        )

    @classmethod
    def storage_dim(cls) -> int:
        return Rot3.storage_dim() + Vector3.storage_dim() * 2

    def to_storage(self) -> T.List[T.Scalar]:
        return self.R.to_storage() + self.v.to_storage() + self.t.to_storage()

    @classmethod
    def from_storage(cls: T.Type[Pose23T], elements: T.Sequence[T.Scalar]) -> Pose23T:
        assert len(elements) == cls.storage_dim()
        R_dim = Rot3.storage_dim()
        v3_dim = Vector3.storage_dim()
        return cls(
            R=Rot3.from_storage(elements[0 : Rot3.storage_dim()]),
            v=Vector3(*elements[R_dim : R_dim + v3_dim]),
            t=Vector3(*elements[R_dim + v3_dim : R_dim + 2 * v3_dim]),
        )

    @classmethod
    def symbolic(cls: T.Type[Pose23T], name: str, **kwargs: T.Any) -> Pose23T:
        return cls(
            R=Rot3.symbolic(f"{name}.R"),
            v=Vector3.symbolic(f"{name}.v"),
            t=Vector3.symbolic(f"{name}.t"),
        )

    # -------------------------------------------------------------------------
    # Group concept - see symforce.ops.group_ops
    # -------------------------------------------------------------------------

    @classmethod
    def identity(cls: T.Type[Pose23T]) -> Pose23T:
        return cls(R=Rot3.identity(), v=Vector3.zero(), t=Vector3.zero())

    def compose(self: Pose23T, other: Pose23T) -> Pose23T:
        assert isinstance(other, self.__class__)
        R = self.R * other.R
        v = self.R * other.v + self.v
        t = self.R * other.t + self.t
        return self.__class__(R=R, v=v, t=t)

    def inverse(self: Pose23T) -> Pose23T:
        so3_inv = self.R.inverse()
        return self.__class__(R=so3_inv, v=-(so3_inv * self.v), t=-(so3_inv * self.t))

    # -------------------------------------------------------------------------
    # Lie group implementation
    # -------------------------------------------------------------------------

    @classmethod
    def tangent_dim(cls) -> int:
        return 9

    @classmethod
    def from_tangent(
        cls: T.Type[Pose23T],
        vec: T.Sequence[T.Scalar],
        epsilon: T.Scalar = sf.epsilon(),
    ) -> Pose23T:
        R_tangent = (vec[0], vec[1], vec[2])
        v_tangent_vector = Vector3(vec[3], vec[4], vec[5])
        t_tangent_vector = Vector3(vec[6], vec[7], vec[8])

        R = Rot3.from_tangent(R_tangent, epsilon=epsilon)
        return cls(R, v_tangent_vector, t_tangent_vector)

    def to_tangent(self: Pose23T, epsilon: T.Scalar = sf.epsilon()) -> T.List[T.Scalar]:
        R_tangent = Vector3(self.R.to_tangent(epsilon=epsilon))
        return R_tangent.col_join(self.v).col_join(self.t).to_flat_list()

    def storage_D_tangent(self: Pose23T) -> Matrix:
        """
        Note: generated from symforce/notebooks/storage_D_tangent.ipynb
        """
        storage_D_tangent_R = self.R.storage_D_tangent()
        storage_D_tangent_v = Matrix33.eye()
        storage_D_tangent_t = Matrix33.eye()
        return Matrix.block_matrix(
            [
                [storage_D_tangent_R, Matrix.zeros(4, 6)],
                [Matrix.zeros(3, 3), storage_D_tangent_v, Matrix.zeros(3, 3)],
                [Matrix.zeros(3, 6), storage_D_tangent_t],
            ]
        )

    def tangent_D_storage(self: Pose23T) -> Matrix:
        """
        Note: generated from symforce/notebooks/tangent_D_storage.ipynb
        """
        tangent_D_storage_R = self.R.tangent_D_storage()
        tangent_D_storage_v = Matrix33.eye()
        tangent_D_storage_t = Matrix33.eye()
        return Matrix.block_matrix(
            [
                [tangent_D_storage_R, Matrix.zeros(3, 6)],
                [Matrix.zeros(3, 4), tangent_D_storage_v, Matrix.zeros(3, 3)],
                [Matrix.zeros(3, 7), tangent_D_storage_t],
            ]
        )

    # NOTE(hayk, aaron): Override retract + local_coordinates, because we're treating
    # the Lie group as the product manifold of SO3 x R3 while leaving compose as normal
    # Pose3 composition.

    def retract(
        self: Pose23T, vec: T.Sequence[T.Scalar], epsilon: T.Scalar = sf.epsilon()
    ) -> Pose23T:
        return self.__class__(
            R=self.R.retract(vec[:3], epsilon=epsilon),
            v=ops.LieGroupOps.retract(self.v, vec[3:6], epsilon=epsilon),
            t=ops.LieGroupOps.retract(self.t, vec[6:9], epsilon=epsilon),
        )

    def local_coordinates(
        self: Pose23T, b: Pose23T, epsilon: T.Scalar = sf.epsilon()
    ) -> T.List[T.Scalar]:
        def local_coordinates(self, b):
            return ops.LieGroupOps.local_coordinates(self, b, epsilon=epsilon)

        return (
            local_coordinates(self.R, b.R)
            + local_coordinates(self.v, b.v)
            + local_coordinates(self.t, b.t)
        )

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    @T.overload
    def __mul__(self: Pose23T, right: Pose23T) -> Pose23T:  # pragma: no cover
        pass

    @T.overload
    def __mul__(self: Pose23T, right: Vector3) -> Vector3:  # pragma: no cover
        pass

    def __mul__(
        self: Pose23T, right: T.Union[Pose23T, Vector3]
    ) -> T.Union[Pose23T, Vector3]:
        """
        Left-multiply with a compatible quantity.
        """
        if isinstance(right, Vector3):
            return self.R * right + self.t
        elif isinstance(right, Pose23):
            return self.compose(right)
        else:
            raise NotImplementedError(f'Unsupported type: "{right}"')

    def to_homogenous_matrix(self) -> Matrix55:
        """
        5x5 matrix representing this pose transform.
        """
        R = self.R.to_rotation_matrix()
        return Matrix.block_matrix(
            [[R, self.v, self.t], [Matrix.zeros(2, 3), Matrix.eye(2)]]
        )

    @classmethod
    def from_homogenous_matrix(cls, mat: Matrix55) -> Pose23T:
        """
        5x5 matrix representing this pose transform.
        """
        R = Rot3.from_rotation_matrix(mat[:3, :3])
        v = mat[:3, 3]
        t = mat[:3, 4]
        return cls(R, v, t)

    def adjoint(self) -> Matrix99:
        """
        Adjoint matrix of this pose.
        """
        R = self.R.to_rotation_matrix()
        v_x = Rot3.hat(self.v)
        p_x = Rot3.hat(self.t)
        return Matrix.block_matrix(
            [
                [R, Matrix.zeros(3, 6)],
                [v_x * R, R, Matrix.zeros(3, 3)],
                [p_x * R, Matrix.zeros(3, 3), R],
            ]
        )

    @classmethod
    def hat(cls, vec: T.Sequence[T.Scalar]) -> Matrix55:
        """Get the lie algebra element corresponding to the given tangent
        vector."""
        R_tangent = [vec[0], vec[1], vec[2]]
        v_tangent = [vec[3], vec[4], vec[5]]
        t_tangent = [vec[6], vec[7], vec[8]]
        top_left = Rot3.hat(R_tangent)
        top_right = Matrix31(v_tangent).row_join(Matrix31(t_tangent))
        bottom = Matrix.zeros(2, 5)
        return T.cast(Matrix55, top_left.row_join(top_right).col_join(bottom))
