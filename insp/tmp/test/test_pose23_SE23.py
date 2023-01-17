# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import numpy as np

import symforce.symbolic as sf
from symforce import typing as T
from symforce.ops import LieGroupOps
from symforce.test_util import TestCase
from symforce.test_util.lie_group_ops_test_mixin import LieGroupOpsTestMixin
import sympy

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from se23.pose23_SE23 import Pose23_SE23


class GeoPose3SE3Test(LieGroupOpsTestMixin, TestCase):
    """
    Test the Pose3_SE3 geometric class.
    Note the mixin that tests all storage, group and lie group ops.
    """

    @classmethod
    def element(cls) -> Pose23_SE23:
        return Pose23_SE23.from_tangent([1.3, 0.2, 1.1,
                                         0.3, -3.8, 16.7,
                                         -0.2, 5.3, 1.2])

    def test_lie_exponential(self) -> None:
        """
        Tests:
            Pose3_SE3.hat
            Pose3_SE3.to_tangent
            Pose3_SE3.to_homogenous_matrix
        """
        element = self.element()
        dim = LieGroupOps.tangent_dim(element)
        pertubation = list(np.random.normal(scale=0.1, size=(dim,)))

        # Compute the hat matrix
        hat = element.hat(pertubation)

        # Take the matrix exponential (only supported with sympy)

        hat_exp = sf.M(sympy.expand(sympy.exp(sympy.S(hat.mat))))

        # As a comparison, take the exponential map and convert to a matrix
        expmap = Pose23_SE23.from_tangent(pertubation, epsilon=self.EPSILON)
        matrix_expected = expmap.to_homogenous_matrix()

        # They should match!
        self.assertStorageNear(hat_exp, matrix_expected, places=5)

    def pose3_operations(self, a: Pose23_SE23, b: Pose23_SE23) -> None:
        """
        Tests Pose3_SE3 operations
        """
        self.assertEqual(a * b, a.compose(b))
        self.assertEqual(a * b.t, a.R * b.t + a.t)

    def test_pose3_operations_numeric(self) -> None:
        """
        Tests (numeric):
            Pose3_SE3.__mul__
        """
        R_a = sf.Rot3.random()
        t_a = sf.V3(np.random.rand(3))
        a = Pose23_SE23(R_a, t_a)

        R_b = sf.Rot3.random()
        t_b = sf.V3(np.random.rand(3))
        b = Pose23_SE23(R_b, t_b)

        self.pose3_operations(a, b)

    def test_pose3_operations_symbolic(self) -> None:
        """
        Tests (symbolic):
            Pose3_SE3.__mul__
        """
        a = Pose23_SE23.symbolic("a")
        b = Pose23_SE23.symbolic("b")
        self.pose3_operations(a, b)


if __name__ == "__main__":
    TestCase.main()
