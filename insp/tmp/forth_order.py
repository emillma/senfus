from symforce.geo import Matrix, Matrix33, Vector3, Rot3, Matrix99, Vector9
from se23.pose23 import Pose23

from se23.pose23_SE23 import Pose23_SE23
import symforce.symbolic as sf
from symforce import typing as T


def trace(A: Matrix):
    return sum(A[i, i] for i in range(min(A.shape)))


def op1(A: Matrix):
    return -trace(A) * Matrix.eye(3) + A


def op2(A: Matrix, B: Matrix):
    return op1(A) @ op1(B) + op1(B @ A)


def four_order(Sigma: Matrix, Q: Matrix):
    Sigma_pp = Sigma[:3, :3]
    Sigma_vp = Sigma[3:6, :3]
    Sigma_vv = Sigma[3:6, 3:6]
    Sigma_vr = Sigma[3:6, 6:9]
    Sigma_rp = Sigma[6:9, :3]
    Sigma_rr = Sigma[6:9, 6:9]

    Qpp = Q[:3, :3]
    Qvp = Q[3:6, :3]
    Qvv = Q[3:6, 3:6]
    Qvr = Q[3:6, 6:9]
    Qrp = Q[6:9, :3]
    Qrr = Q[6:9, 6:9]

    A1 = Matrix.zeros(9, 9)
    A1[:3, :3] = op1(Sigma_pp)
    A1[3:6, :3] = op1(Sigma_vp + Sigma_vp.T)
    A1[3:6, 3:6] = op1(Sigma_pp)
    A1[6:9, :3] = op1(Sigma_rp + Sigma_rp.T)
    A1[6:9, 6:9] = op1(Sigma_pp)

    A2 = Matrix.zeros(9, 9)
    A2[:3, :3] = op1(Qpp)
    A2[3:6, :3] = op1(Qvp + Qvp.T)
    A2[3:6, 3:6] = op1(Qpp)
    A2[6:9, :3] = op1(Qrp + Qrp.T)
    A2[6:9, 6:9] = op1(Qpp)

    Bpp = op2(Sigma_pp, Qpp)
    Bvv = (
        op2(Sigma_pp, Qvv)
        + op2(Sigma_vp.T, Qvp)
        + op2(Sigma_vp, Qvp.T)
        + op2(Sigma_vv, Qpp)
    )
    Brr = (
        op2(Sigma_pp, Qrr)
        + op2(Sigma_rp.T, Qrp)
        + op2(Sigma_rp, Qrp.T)
        + op2(Sigma_rr, Qpp)
    )
    Bvp = op2(Sigma_pp, Qvp.T) + op2(Sigma_vp.T, Qpp)
    Brp = op2(Sigma_pp, Qrp.T) + op2(Sigma_rp.T, Qpp)
    Bvr = (
        op2(Sigma_pp, Qvr)
        + op2(Sigma_vp.T, Qrp)
        + op2(Sigma_rp, Qvp.T)
        + op2(Sigma_vr, Qpp)
    )

    B = Matrix.zeros(9, 9)
    B[:3, :3] = Bpp
    B[:3, 3:6] = Bvp.T
    B[:3, 6:9] = Brp.T
    B[3:6, :3] = B[:3, 3:6].T
    B[3:6, 3:6] = Bvv
    B[3:6, 6:9] = Bvr
    B[6:9, :3] = B[:3, 6:9].T
    B[6:9, 3:6] = B[3:6, 6:9].T
    B[6:9, 6:9] = Brr

    return (A1 @ Q + Q.T @ A1.T + A2 @ Sigma + Sigma.T @ A2.T) / 12 + B / 4
