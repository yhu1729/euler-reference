import numpy
import numba
import matplotlib.pyplot as pyplot

GHOST = 3


@numba.njit
def roll_1D(in_Q, shift, periodic=True):
    Q = numpy.empty_like(in_Q)

    if periodic:
        n = Q.shape[1]
        shift %= n
        Q[:, :shift] = in_Q[:, n - shift :]
        Q[:, shift:] = in_Q[:, : n - shift]
    else:
        if shift > 0:
            Q[:, shift:] = in_Q[:, :-shift]
            for index in range(shift):
                Q[:, index] = in_Q[:, 0]
        elif shift < 0:
            Q[:, :shift] = in_Q[:, -shift:]
            for index in range(-shift):
                Q[:, -1 - index] = in_Q[:, -1]
        else:
            Q = in_Q.copy()

    return Q


@numba.njit
def find_energy(density, momentum_x, momentum_y, pressure, gamma=1.4):
    return pressure / (gamma - 1) + 0.5 * (momentum_x**2 + momentum_y**2) / density


@numba.njit
def find_pressure(density, momentum_x, momentum_y, energy, gamma=1.4):
    return (gamma - 1) * (energy - 0.5 * (momentum_x**2 + momentum_y**2) / density)


@numba.njit
def find_flux_physical(Q, velocity_x, velocity_y, pressure, direction, gamma=1.4):
    flux = numpy.zeros_like(Q)
    if direction == "x":
        flux[0] = Q[0] * velocity_x
        flux[1] = Q[0] * velocity_x**2 + pressure
        flux[2] = Q[0] * velocity_x * velocity_y
        flux[3] = (Q[3] + pressure) * velocity_x
    elif direction == "y":
        flux[0] = Q[0] * velocity_y
        flux[1] = Q[0] * velocity_x * velocity_y
        flux[2] = Q[0] * velocity_y**2 + pressure
        flux[3] = (Q[3] + pressure) * velocity_y
    else:
        raise RuntimeError(f"Invalid parameter: direction={direction}")

    return flux


@numba.njit
def find_flux_Roe(
    Q_L,
    Q_R,
    direction,
    gamma=1.4,
    use_dissipation=True,
    use_JST_dissipation=False,
    JST_kappa_2=(1 / 4),
    JST_kappa_4=(1 / 256),
):
    rho_L = Q_L[0]
    u_L = Q_L[1] / rho_L
    v_L = Q_L[2] / rho_L
    p_L = find_pressure(Q_L[0], Q_L[1], Q_L[2], Q_L[3], gamma=gamma)
    H_L = (Q_L[3] + p_L) / rho_L

    rho_R = Q_R[0]
    u_R = Q_R[1] / rho_R
    v_R = Q_R[2] / rho_R
    p_R = find_pressure(Q_R[0], Q_R[1], Q_R[2], Q_R[3], gamma=gamma)
    H_R = (Q_R[3] + p_R) / rho_R

    ratio = numpy.sqrt(rho_R / rho_L)
    rho_Roe = ratio * rho_L
    u_Roe = (u_L + ratio * u_R) / (1.0 + ratio)
    v_Roe = (v_L + ratio * v_R) / (1.0 + ratio)
    H_Roe = (H_L + ratio * H_R) / (1.0 + ratio)
    speed2_Roe = u_Roe**2 + v_Roe**2
    c_Roe = numpy.sqrt(numpy.maximum((gamma - 1.0) * (H_Roe - 0.5 * speed2_Roe), 1e-12))

    drho = rho_R - rho_L
    du = u_R - u_L
    dv = v_R - v_L
    dp = p_R - p_L

    if direction == "x":
        u_Roe_normal = u_Roe
        u_Roe_tangential = v_Roe
        du_normal = du
        du_tangential = dv
    else:
        u_Roe_normal = v_Roe
        u_Roe_tangential = u_Roe
        du_normal = dv
        du_tangential = du

    # low Mach number correction
    du_normal *= numpy.minimum(1.0, numpy.sqrt(speed2_Roe) / c_Roe)

    alpha_1 = 0.5 * (dp - rho_Roe * c_Roe * du_normal) / c_Roe**2
    alpha_2 = drho - dp / c_Roe**2
    alpha_3 = rho_Roe * du_tangential
    alpha_4 = 0.5 * (dp + rho_Roe * c_Roe * du_normal) / c_Roe**2

    lambda_1 = numpy.abs(u_Roe_normal - c_Roe)
    lambda_2 = numpy.abs(u_Roe_normal)
    lambda_3 = numpy.abs(u_Roe_normal)
    lambda_4 = numpy.abs(u_Roe_normal + c_Roe)

    # entropy correction
    epsilon_entropy = numpy.maximum(0.1 * c_Roe, 1e-12)
    lambda_1 = numpy.where(
        lambda_1 < epsilon_entropy,
        (lambda_1**2 + epsilon_entropy**2) / (2 * epsilon_entropy),
        lambda_1,
    )
    lambda_4 = numpy.where(
        lambda_4 < epsilon_entropy,
        (lambda_4**2 + epsilon_entropy**2) / (2 * epsilon_entropy),
        lambda_4,
    )

    # dissipation
    dissipation = numpy.zeros_like(Q_L)
    if use_dissipation:
        dW_1 = alpha_1 * lambda_1
        dW_2 = alpha_2 * lambda_2
        dW_3 = alpha_3 * lambda_3
        dW_4 = alpha_4 * lambda_4

        dissipation_density = dW_1 + dW_2 + dW_4
        dissipation_normal = (
            (u_Roe_normal - c_Roe) * dW_1
            + u_Roe_normal * dW_2
            + (u_Roe_normal + c_Roe) * dW_4
        )
        dissipation_tangential = u_Roe_tangential * (dW_1 + dW_2 + dW_4) + dW_3
        dissipation_energy = (
            (H_Roe - u_Roe_normal * c_Roe) * dW_1
            + 0.5 * speed2_Roe * dW_2
            + u_Roe_tangential * dW_3
            + (H_Roe + u_Roe_normal * c_Roe) * dW_4
        )

        dissipation[0] = dissipation_density
        dissipation[1] = (
            dissipation_normal if direction == "x" else dissipation_tangential
        )
        dissipation[2] = (
            dissipation_tangential if direction == "x" else dissipation_normal
        )
        dissipation[3] = dissipation_energy

    # JST dissipation
    dissipation_JST = numpy.zeros_like(Q_L)
    if use_JST_dissipation:
        p_Q_L = find_pressure(Q_L[0], Q_L[1], Q_L[2], Q_L[3], gamma=gamma)
        p_Q_R = find_pressure(Q_R[0], Q_R[1], Q_R[2], Q_R[3], gamma=gamma)

        Q_L_m1 = roll_1D(Q_L, 1, periodic=False)
        Q_L_p1 = roll_1D(Q_L, -1, periodic=False)
        p_Q_L_m1 = find_pressure(
            Q_L_m1[0], Q_L_m1[1], Q_L_m1[2], Q_L_m1[3], gamma=gamma
        )
        p_Q_L_p1 = find_pressure(
            Q_L_p1[0], Q_L_p1[1], Q_L_p1[2], Q_L_p1[3], gamma=gamma
        )
        nu_L = numpy.abs(p_Q_L_p1 - 2 * p_Q_L + p_Q_L_m1) / (
            numpy.abs(p_Q_L_p1 + 2 * p_Q_L + p_Q_L_m1) + 1e-16
        )

        Q_R_m1 = roll_1D(Q_R, 1, periodic=False)
        Q_R_p1 = roll_1D(Q_R, -1, periodic=False)
        p_Q_R_m1 = find_pressure(
            Q_R_m1[0], Q_R_m1[1], Q_R_m1[2], Q_R_m1[3], gamma=gamma
        )
        p_Q_R_p1 = find_pressure(
            Q_R_p1[0], Q_R_p1[1], Q_R_p1[2], Q_R_p1[3], gamma=gamma
        )
        nu_R = numpy.abs(p_Q_R_p1 - 2 * p_Q_R + p_Q_R_m1) / (
            numpy.abs(p_Q_R_p1 + 2 * p_Q_R + p_Q_R_m1) + 1e-16
        )

        nu = numpy.maximum(nu_L, nu_R)
        epsilon_2 = JST_kappa_2 * nu
        epsilon_4 = numpy.maximum(0, JST_kappa_4 - epsilon_2)

        dQ = Q_R - Q_L
        d2Q_L = Q_L_p1 - 2 * Q_L + Q_L_m1
        d2Q_R = Q_R_p1 - 2 * Q_R + Q_R_m1
        dissipation_JST = (numpy.abs(u_Roe_normal) + c_Roe) * (
            epsilon_2 * dQ - epsilon_4 * (d2Q_R - d2Q_L)
        )

    F_L = find_flux_physical(Q_L, u_L, v_L, p_L, direction, gamma=gamma)
    F_R = find_flux_physical(Q_R, u_R, v_R, p_R, direction, gamma=gamma)

    return 0.5 * (F_L + F_R - dissipation) - dissipation_JST


@numba.njit
def apply_limiter(Q, Q_WENO, gamma=1.4, weno_epsilon=1e-6):
    out_Q = numpy.copy(Q_WENO)

    rho = Q[0]
    rho_WENO = out_Q[0]

    theta_1 = numpy.ones_like(rho)
    valid = (rho_WENO < weno_epsilon) & (rho > weno_epsilon)
    theta_1[valid] = (rho[valid] - weno_epsilon) / (rho[valid] - rho_WENO[valid])

    out_Q = Q + theta_1 * (out_Q - Q)

    p = find_pressure(Q[0], Q[1], Q[2], Q[3], gamma=gamma)
    p_WENO = find_pressure(out_Q[0], out_Q[1], out_Q[2], out_Q[3], gamma=gamma)

    theta_2 = numpy.ones_like(p)
    valid = (p_WENO < weno_epsilon) & (p > weno_epsilon)
    theta_2[valid] = (p[valid] - weno_epsilon) / (p[valid] - p_WENO[valid])

    out_Q = Q + theta_2 * (out_Q - Q)

    return out_Q


@numba.njit
def apply_boundary_condition(Q, style_x, style_y):
    N_x = Q.shape[1] - 2 * GHOST
    N_y = Q.shape[2] - 2 * GHOST

    if style_x == "periodic":
        Q[:, :GHOST, :] = Q[:, N_x : N_x + GHOST, :]
        Q[:, N_x + GHOST :, :] = Q[:, GHOST : 2 * GHOST, :]
    elif style_x == "outflow":
        for index in range(GHOST):
            Q[:, index, :] = Q[:, GHOST, :]
            Q[:, N_x + GHOST + index, :] = Q[:, N_x + GHOST - 1, :]
    elif style_x == "inflow":
        pass
    elif style_x == "reflective":
        for index in range(GHOST):
            Q[:, GHOST - 1 - index, :] = Q[:, GHOST + index, :]
            Q[:, N_x + GHOST + index, :] = Q[:, N_x + GHOST - 1 - index, :]

            Q[1, GHOST - 1 - index, :] *= -1.0
            Q[1, N_x + GHOST + index, :] *= -1.0
    else:
        raise RuntimeError(f"Invalid parameter: style_x={style_x}")

    if style_y == "periodic":
        Q[:, :, :GHOST] = Q[:, :, N_y : N_y + GHOST]
        Q[:, :, N_y + GHOST :] = Q[:, :, GHOST : 2 * GHOST]
    elif style_y == "inflow":
        pass
    elif style_y == "outflow":
        for index in range(GHOST):
            Q[:, :, index] = Q[:, :, GHOST]
            Q[:, :, N_y + GHOST + index] = Q[:, :, N_y + GHOST - 1]
    elif style_y == "reflective":
        for index in range(GHOST):
            Q[:, :, GHOST - 1 - index] = Q[:, :, GHOST + index]
            Q[:, :, N_y + GHOST + index] = Q[:, :, N_y + GHOST - 1 - index]

            Q[2, :, GHOST - 1 - index] *= -1.0
            Q[2, :, N_y + GHOST + index] *= -1.0
    else:
        raise RuntimeError(f"Invalid parameter: style_y={style_y}")

    return Q


@numba.njit
def apply_boundary_condition_inflow(Q, Q_infinity, axis, direction):
    if axis == "x":
        if direction < 0:
            for index in range(4):
                Q[index, :GHOST, :] = Q_infinity[index]
        elif direction > 0:
            for index in range(4):
                Q[index, -GHOST:, :] = Q_infinity[index]
        else:
            raise RuntimeError(f"Invalid parameter: direction={direction}")
    elif axis == "y":
        if direction < 0:
            for index in range(4):
                Q[index, :, :GHOST] = Q_infinity[index]
        elif direction > 0:
            for index in range(4):
                Q[index, :, -GHOST:] = Q_infinity[index]
        else:
            raise RuntimeError(f"Invalid parameter: direction={direction}")
    else:
        raise RuntimeError(f"Invalid parameter: axis={axis}")

    return Q


@numba.njit
def setup_wall(mask):
    solid_mask = ~mask
    solid_x, solid_y = numpy.where(solid_mask)
    fluid_x, fluid_y = numpy.where(mask)

    index_fluid_x = numpy.zeros_like(solid_x)
    index_fluid_y = numpy.zeros_like(solid_y)

    for i in range(len(solid_x)):
        r2 = (solid_x[i] - fluid_x) ** 2 + (solid_y[i] - fluid_y) ** 2
        index = numpy.argmin(r2)
        index_fluid_x[i] = fluid_x[index]
        index_fluid_y[i] = fluid_y[index]

    index_normal_x = (index_fluid_x - solid_x).astype(numpy.float64)
    index_normal_y = (index_fluid_y - solid_y).astype(numpy.float64)

    r = numpy.sqrt(index_normal_x**2 + index_normal_y**2)
    index_normal_x /= r
    index_normal_y /= r

    return (
        solid_x,
        solid_y,
        index_fluid_x,
        index_fluid_y,
        index_normal_x,
        index_normal_y,
    )


@numba.njit
def apply_wall(Q, wall, gamma=1.4):
    solid_x, solid_y, fluid_x, fluid_y, normal_x, normal_y = wall

    for i in range(len(solid_x)):
        rho_fluid = Q[0, fluid_x[i], fluid_y[i]]
        rho_u_fluid = Q[1, fluid_x[i], fluid_y[i]]
        rho_v_fluid = Q[2, fluid_x[i], fluid_y[i]]
        u_fluid = rho_u_fluid / rho_fluid
        v_fluid = rho_v_fluid / rho_fluid
        E_fluid = Q[3, fluid_x[i], fluid_y[i]]
        p_fluid = find_pressure(
            Q[0, fluid_x[i], fluid_y[i]],
            Q[1, fluid_x[i], fluid_y[i]],
            Q[2, fluid_x[i], fluid_y[i]],
            Q[3, fluid_x[i], fluid_y[i]],
            gamma=gamma,
        )

        s = u_fluid * normal_x[i] + v_fluid * normal_y[i]
        u_mirror = u_fluid - 2.0 * s * normal_x[i]
        v_mirror = v_fluid - 2.0 * s * normal_y[i]

        Q[0, solid_x[i], solid_y[i]] = rho_fluid
        Q[1, solid_x[i], solid_y[i]] = rho_fluid * u_mirror
        Q[2, solid_x[i], solid_y[i]] = rho_fluid * v_mirror
        Q[3, solid_x[i], solid_y[i]] = find_energy(
            Q[0, solid_x[i], solid_y[i]],
            Q[1, solid_x[i], solid_y[i]],
            Q[2, solid_x[i], solid_y[i]],
            p_fluid,
            gamma=gamma,
        )

    return Q


@numba.njit
def reconstruct(Q, weno_epsilon=1e-6):
    q_m2 = Q[:-4]
    q_m1 = Q[1:-3]
    q_c = Q[2:-2]
    q_p1 = Q[3:-1]
    q_p2 = Q[4:]

    beta_1 = (
        13.0 / 12.0 * (q_c - 2.0 * q_p1 + q_p2) ** 2
        + 1.0 / 4.0 * (3.0 * q_c - 4.0 * q_p1 + q_p2) ** 2
    )
    beta_2 = (
        13.0 / 12.0 * (q_m1 - 2.0 * q_c + q_p1) ** 2 + 1.0 / 4.0 * (q_m1 - q_p1) ** 2
    )
    beta_3 = (
        13.0 / 12.0 * (q_m2 - 2.0 * q_m1 + q_c) ** 2
        + 1.0 / 4.0 * (q_m2 - 4.0 * q_m1 + 3.0 * q_c) ** 2
    )

    alpha_1 = 0.3 / (weno_epsilon + beta_1) ** 2
    alpha_2 = 0.6 / (weno_epsilon + beta_2) ** 2
    alpha_3 = 0.1 / (weno_epsilon + beta_3) ** 2
    sum_alpha = alpha_1 + alpha_2 + alpha_3

    weight_1 = alpha_1 / sum_alpha
    weight_2 = alpha_2 / sum_alpha
    weight_3 = alpha_3 / sum_alpha

    return (
        weight_1 * (1.0 / 3.0 * q_c + 5.0 / 6.0 * q_p1 - 1.0 / 6.0 * q_p2)
        + weight_2 * (-1.0 / 6.0 * q_m1 + 5.0 / 6.0 * q_c + 1.0 / 3.0 * q_p1)
        + weight_3 * (1.0 / 3.0 * q_m2 - 7.0 / 6.0 * q_m1 + 11.0 / 6.0 * q_c)
    )


@numba.njit
def reconstruct_Z(Q, weno_epsilon=1e-40, power=2):
    q_m2 = Q[:-4]
    q_m1 = Q[1:-3]
    q_c = Q[2:-2]
    q_p1 = Q[3:-1]
    q_p2 = Q[4:]

    beta_1 = (
        13.0 / 12.0 * (q_c - 2.0 * q_p1 + q_p2) ** 2
        + 1.0 / 4.0 * (3.0 * q_c - 4.0 * q_p1 + q_p2) ** 2
    )
    beta_2 = (
        13.0 / 12.0 * (q_m1 - 2.0 * q_c + q_p1) ** 2 + 1.0 / 4.0 * (q_m1 - q_p1) ** 2
    )
    beta_3 = (
        13.0 / 12.0 * (q_m2 - 2.0 * q_m1 + q_c) ** 2
        + 1.0 / 4.0 * (q_m2 - 4.0 * q_m1 + 3.0 * q_c) ** 2
    )

    tau_5 = numpy.abs(beta_1 - beta_3)

    alpha_1 = 0.3 * (1.0 + (tau_5 / (beta_1 + weno_epsilon)) ** power)
    alpha_2 = 0.6 * (1.0 + (tau_5 / (beta_2 + weno_epsilon)) ** power)
    alpha_3 = 0.1 * (1.0 + (tau_5 / (beta_3 + weno_epsilon)) ** power)
    sum_alpha = alpha_1 + alpha_2 + alpha_3

    weight_1 = alpha_1 / sum_alpha
    weight_2 = alpha_2 / sum_alpha
    weight_3 = alpha_3 / sum_alpha

    return (
        weight_1 * (1.0 / 3.0 * q_c + 5.0 / 6.0 * q_p1 - 1.0 / 6.0 * q_p2)
        + weight_2 * (-1.0 / 6.0 * q_m1 + 5.0 / 6.0 * q_c + 1.0 / 3.0 * q_p1)
        + weight_3 * (1.0 / 3.0 * q_m2 - 7.0 / 6.0 * q_m1 + 11.0 / 6.0 * q_c)
    )


@numba.njit
def find_flux_1D(
    Q,
    dx,
    direction,
    gamma=1.4,
    weno_epsilon=1e-6,
    use_dissipation=True,
    use_JST_dissipation=False,
    JST_kappa_2=(1 / 4),
    JST_kappa_4=(1 / 256),
):
    L = Q.shape[1] - 2 * GHOST + 1

    W = numpy.zeros_like(Q)
    W[0, :] = Q[0, :]
    W[1, :] = Q[1, :] / Q[0, :]
    W[2, :] = Q[2, :] / Q[0, :]
    W[3, :] = find_pressure(Q[0, :], Q[1, :], Q[2, :], Q[3, :], gamma=gamma)

    W_L = numpy.zeros((4, L))
    W_R = numpy.zeros((4, L))

    for n in range(4):
        W_L[n, :] = reconstruct(W[n, :-1], weno_epsilon=weno_epsilon)
        W_R[n, :] = reconstruct(W[n, 1:][::-1], weno_epsilon=weno_epsilon)[::-1]

    Q_L = numpy.zeros((4, L))
    Q_L[0, :] = W_L[0, :]
    Q_L[1, :] = W_L[0, :] * W_L[1, :]
    Q_L[2, :] = W_L[0, :] * W_L[2, :]
    Q_L[3, :] = find_energy(Q_L[0, :], Q_L[1, :], Q_L[2, :], W_L[3, :], gamma=gamma)

    Q_R = numpy.zeros((4, L))
    Q_R[0, :] = W_R[0, :]
    Q_R[1, :] = W_R[0, :] * W_R[1, :]
    Q_R[2, :] = W_R[0, :] * W_R[2, :]
    Q_R[3, :] = find_energy(Q_R[0, :], Q_R[1, :], Q_R[2, :], W_R[3, :], gamma=gamma)

    Q_L_average = Q[:, GHOST - 1 : GHOST - 1 + L]
    Q_R_average = Q[:, GHOST : GHOST + L]

    Q_L_limited = apply_limiter(
        Q_L_average, Q_L, gamma=gamma, weno_epsilon=weno_epsilon
    )
    Q_R_limited = apply_limiter(
        Q_R_average, Q_R, gamma=gamma, weno_epsilon=weno_epsilon
    )

    flux = find_flux_Roe(
        Q_L_limited,
        Q_R_limited,
        direction,
        gamma=gamma,
        use_dissipation=use_dissipation,
        use_JST_dissipation=use_JST_dissipation,
        JST_kappa_2=JST_kappa_2,
        JST_kappa_4=JST_kappa_4,
    )

    return -(flux[:, 1:] - flux[:, :-1]) / dx


@numba.njit
def find_RHS(
    Q,
    dx,
    dy,
    gamma=1.4,
    boundary_x="periodic",
    boundary_y="periodic",
    update_boundary=None,
    wall=None,
    weno_epsilon=1e-6,
    use_dissipation=True,
    use_JST_dissipation=False,
    JST_kappa_2=(1 / 4),
    JST_kappa_4=(1 / 256),
):
    Q = numpy.copy(Q)
    Q = apply_boundary_condition(Q, boundary_x, boundary_y)
    if update_boundary is not None:
        Q = update_boundary(Q, gamma=gamma)

    if wall is not None:
        Q = apply_wall(Q, wall, gamma=gamma)

    N_x = Q.shape[1] - 2 * GHOST
    N_y = Q.shape[2] - 2 * GHOST

    RHS = numpy.zeros((4, N_x, N_y))
    for index in range(GHOST, N_y + GHOST):
        RHS[:, :, index - GHOST] += find_flux_1D(
            Q[:, :, index],
            dx,
            "x",
            gamma=gamma,
            weno_epsilon=weno_epsilon,
            use_dissipation=use_dissipation,
            use_JST_dissipation=use_JST_dissipation,
            JST_kappa_2=JST_kappa_2,
            JST_kappa_4=JST_kappa_4,
        )
    for index in range(GHOST, N_x + GHOST):
        RHS[:, index - GHOST, :] += find_flux_1D(
            Q[:, index, :],
            dy,
            "y",
            gamma=gamma,
            weno_epsilon=weno_epsilon,
            use_dissipation=use_dissipation,
            use_JST_dissipation=use_JST_dissipation,
            JST_kappa_2=JST_kappa_2,
            JST_kappa_4=JST_kappa_4,
        )

    return RHS
