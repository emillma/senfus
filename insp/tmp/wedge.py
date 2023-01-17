def wedge(v: Vector9) -> Matrix99:
    phi_x = Rot3.hat(v[0:3])
    v_x = Rot3.hat(v[3:6])
    rho_x = Rot3.hat(v[6:9])
    zero33 = Matrix.zeros(3, 3)
    return Matrix.block_matrix(
        [[phi_x, zero33, zero33], [v_x, phi_x, zero33], [rho_x, v_x, phi_x]]
    )
