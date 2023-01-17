from symforce.geo import Rot3
from symforce.geo.matrix import Matrix, Matrix33, Vector3
from symforce import symbolic as sf
from symforce import typing as T


class Rot3(Rot3):
    """Extended symforce Rot3 class with Jacobian methods"""

    def ljac(self, epsilon: T.Scalar = sf.epsilon()) -> Matrix33:
        """Left jacobian"""
        R_tangent = self.to_tangent(epsilon=epsilon)
        R_hat = Rot3.hat(R_tangent)
        R_hat_sq = R_hat * R_hat
        R_tangent_vector = Vector3(R_tangent)
        theta = sf.sqrt(R_tangent_vector.squared_norm() + epsilon**2)

        V = (
            Matrix.eye(3)
            + (1 - sf.cos(theta)) / (theta**2) * R_hat
            + (theta - sf.sin(theta)) / (theta**3) * R_hat_sq
        )
        return V

    def ljac_inv(self, epsilon: T.Scalar = sf.epsilon()) -> Matrix33:
        """Left jacobian inverse"""
        R_tangent = self.to_tangent(epsilon=epsilon)
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
        return V_inv

    def jac(self, epsilon: T.Scalar = sf.epsilon()) -> Matrix33:
        """Right jacobian, see MicroLie(147)"""
        return self.jac(epsilon=epsilon).T

    def jac_inv(self, epsilon: T.Scalar = sf.epsilon()) -> Matrix33:
        """Right jacobian inverse, see MicroLie(147)"""
        return self.jac_inv(epsilon=epsilon).T
