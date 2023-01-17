from symforce.geo import Matrix, Matrix33, Vector3, Rot3, Matrix99, Matrix66
from se23.pose23 import Pose23

from se23.pose23_SE23 import Pose23_SE23
import symforce.symbolic as sf
from symforce import typing as T

from states import ImuNoise, ZImuEst, ImuPreint, State


def SO3_ljac_inv(phi: Vector3, epsilon: T.Scalar = sf.epsilon()) -> Matrix33:
    """Left jacobian inverse"""
    theta = sf.sqrt(phi.squared_norm() + epsilon)
    R_hat = Rot3.hat(phi)
    half_theta = 0.5 * theta

    V_inv = (
        Matrix.eye(3)
        - 0.5 * R_hat
        + (
            (1 - theta * sf.cos(half_theta) / (2 * sf.sin(half_theta)))
            / (theta * theta)
        )
        * (R_hat * R_hat)
    )
    return V_inv


def Phi(pose: Pose23_SE23, dt: sf.Scalar) -> Pose23_SE23:
    return Pose23_SE23(pose.R, pose.v, pose.t + dt * pose.v)


def Gamma(g: Vector3, dt: sf.Scalar) -> Pose23_SE23:
    return Pose23_SE23(Rot3.identity(), g * dt, g * dt**2 / 2)


def preintegrate(
    imu_noise: ImuNoise,
    preint_prev: ImuPreint,
    z_imu_est: ZImuEst,
    dt: sf.Scalar,
) -> Matrix99:
    """[se23(39)]"""
    delta_R = Rot3.from_tangent((z_imu_est.gyro) * dt)

    a_0 = z_imu_est.accl
    a_1 = delta_R * (z_imu_est.accl)
    delta_v = a_0 * dt + (a_1 - a_0) * dt**2 / 2
    delta_t = a_0 * dt**2 / 2 + (a_1 - a_0) * dt**3 / 6

    upsilon_new = Phi(preint_prev.upsilon, dt).compose(
        Pose23_SE23(R=delta_R, v=delta_v, t=delta_t)
    )

    J_inv = SO3_ljac_inv(z_imu_est.gyro * dt)
    rot_from_omega_dt = Rot3.from_tangent(-z_imu_est.gyro * dt)
    R = rot_from_omega_dt.to_rotation_matrix()
    G = -Matrix.block_matrix(
        [
            [J_inv * dt, Matrix.zeros(3, 3)],
            [Matrix.zeros(3, 3), R * dt],
            [Matrix.zeros(3, 3), R * dt**2 / 2],
        ]
    )
    Q_i = G * (imu_noise.cov * dt) * G.T

    F = Matrix.eye(9, 9)
    F[6:9, 3:6] = dt * Matrix.eye(3, 3)
    A = preint_prev.upsilon.inverse().adjoint() * F

    cov_new = A * preint_prev.cov * A.T + Q_i
    return ImuPreint(upsilon_new, cov_new)
