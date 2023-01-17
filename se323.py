import symforce

symforce.set_symbolic_api("sympy")

import symforce.symbolic as sf
from symforce.geo import Matrix, Matrix33, Vector3, Pose3, Rot3
from symforce.geo.unsupported import Pose3_SE3
import sympy as sp
from sympy.printing import latex
from sympy.matrices.matrices import MatrixCalculus

omega = Vector3.symbolic("omega")
t = sf.Symbol("t")

rotvec = omega * t
rotmat = Rot3.from_tangent(rotvec).to_rotation_matrix()
# sp.dotprint
print(sp.cse(rotmat))
# print(rotmat.simplify())
# rmat = Rot3.from_tangent(rotvec).to_rotation_matrix().mat

# rmat_hello = MatrixCalculus.integrate(omega, rmat)
# print(latex(rmat_hello))
# rmat_t = rmat[0, 0]
# print(rmat_t)
