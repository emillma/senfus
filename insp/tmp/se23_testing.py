"""module for testing stuff"""

import numpy as np
import os
import sympy as sp
import symforce

symforce.set_symbolic_api("symengine")
symforce.set_epsilon_to_symbol()

import symforce.symbolic as sf

from symforce.geo import Matrix, Matrix33, Vector3, Rot3
from se23.pose23 import Pose23
from se23.pose23_SE23 import Pose23_SE23
from symforce import codegen
from functools import partial

# engine = 'symengine'
# engine = 'sympy'


def integrate(
    pose: Pose23_SE23,
    accel: Vector3,
    omega: Vector3,
    accel_b: Vector3,
    omega_b: Vector3,
    g: Vector3,
    t: sf.Scalar,
    epsilon: sf.Scalar = 0,
) -> Pose23_SE23:
    delta_R = Rot3.from_tangent((omega - omega_b) * t, epsilon=epsilon)
    a_0 = accel - accel_b
    a_1 = delta_R * (accel - accel_b)
    delta_v = a_0 * t + (a_1 - a_0) * t**2 / 2
    delta_t = a_0 * t**2 / 2 + (a_1 - a_0) * t**3 / 6

    Upsilon = Matrix.block_matrix(
        [
            [delta_R.to_rotation_matrix(), delta_v, delta_t],
            [Matrix.zeros(2, 3), Matrix.eye(2, 2)],
        ]
    )

    Gamma = Matrix.block_matrix(
        [
            [Matrix.eye(3, 3), t * g, g * t**2 / 2],
            [Matrix.zeros(2, 3), Matrix.eye(2, 2)],
        ]
    )

    Theta = Matrix.block_matrix(
        [
            [pose.R.to_rotation_matrix(), pose.v, pose.t + t * pose.v],
            [Matrix.zeros(2, 3), Matrix.eye(2, 2)],
        ]
    )

    pose_new = Pose23_SE23.from_homogenous_matrix(Gamma * Theta * Upsilon)

    return pose_new


pose = Pose23_SE23.symbolic("T_0")
accel = Vector3.symbolic("a")
accel_b = Vector3.symbolic("a_b")
omega = Vector3.symbolic("w")
omega_b = Vector3.symbolic("w_b")

g = Vector3.symbolic("g")
t = sf.Symbol("t", positive=True)
print(sf.cse(integrate(pose, accel, omega, accel_b, omega_b, g, t).to_storage()))

az_el_codegen = codegen.Codegen.function(
    func=partial(integrate, t=0.1),
    config=codegen.PythonConfig(),
)
az_el_codegen_data = az_el_codegen.generate_function()

for f in az_el_codegen_data.generated_files:
    print("  |- {}".format(os.path.relpath(f, az_el_codegen_data.output_dir)))

print(az_el_codegen_data.generated_files[0].read_text())
# delta_v_tmp  = sp.Matrix(delta_R.to_rotation_matrix() * (accel - accel_b))
# delta_v = sp.integrate(delta_v_tmp[0,0], (t, 0, t))
# print(delta_R)
