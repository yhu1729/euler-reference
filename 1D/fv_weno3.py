import math
import numba
import numpy


@numba.njit(fastmath=True, cache=True)
def find_energy_1D(gamma, density, momentum, pressure):
    return pressure / (gamma - 1) + momentum**2 / 2 / density


@numba.njit(fastmath=True, cache=True)
def find_pressure_1D(gamma, density, momentum, energy):
    return (gamma - 1) * (energy - momentum**2 / 2 / density)


@numba.njit(fastmath=True, cache=True)
def pack_1D(w, index_j, density, momentum, energy):
    for index in range(6):
        index_ghost = index_j - 3 + index
        w[index, 0] = density[index_ghost]
        w[index, 1] = momentum[index_ghost]
        w[index, 2] = 0
        w[index, 3] = 0
        w[index, 4] = energy[index_ghost]


@numba.njit(fastmath=True, cache=True)
def pack_1D_boundary_periodic(w, N, index_j, density, momentum, energy):
    for i in range(6):
        index_periodic = index_j + i - 3
        if index_periodic < 0:
            index_periodic += N
        elif index_periodic >= N:
            index_periodic -= N
        w[i, 0] = density[index_periodic]
        w[i, 1] = momentum[index_periodic]
        w[i, 2] = 0
        w[i, 3] = 0
        w[i, 4] = energy[index_periodic]


@numba.njit(fastmath=True, cache=True)
def pack_1D_boundary_outflow(w, N, index_j, density, momentum, energy):
    for i in range(6):
        index_outflow = index_j + i - 3
        index_outflow = max(0, min(index_outflow, N - 1))
        w[i, 0] = density[index_outflow]
        w[i, 1] = momentum[index_outflow]
        w[i, 2] = 0
        w[i, 3] = 0
        w[i, 4] = energy[index_outflow]


@numba.njit(fastmath=True, cache=True)
def pack_1D_boundary_reflective(w, N, index_j, density, momentum, energy):
    # left
    for index in range(3):
        index_ghost = index_j + index - 3
        index_mirror = 2 - (index_j + index)
        w[index, 0] = (
            density[index_mirror] if (index_ghost < 0) else density[index_ghost]
        )
        w[index, 1] = (
            -momentum[index_mirror] if (index_ghost < 0) else momentum[index_ghost]
        )
        w[index, 2] = 0
        w[index, 3] = 0
        w[index, 4] = energy[index_mirror] if (index_ghost < 0) else energy[index_ghost]

    # right
    for index in range(3):
        index_ghost = index_j + index
        index_mirror = 2 * N - index_ghost - 1
        w[index + 3, 0] = (
            density[index_mirror] if (index_ghost >= N) else density[index_ghost]
        )
        w[index + 3, 1] = (
            -momentum[index_mirror] if (index_ghost >= N) else momentum[index_ghost]
        )
        w[index + 3, 2] = 0
        w[index + 3, 3] = 0
        w[index + 3, 4] = (
            energy[index_mirror] if (index_ghost >= N) else energy[index_ghost]
        )


@numba.njit(fastmath=True, cache=True)
def find_flux_1D(gamma, w, flux_face):
    beta_coefficient = 13 / 12
    epsilon = 1e-6
    pressure = numpy.zeros(6)
    for index in range(6):
        pressure[index] = find_pressure_1D(gamma, w[index, 0], w[index, 1], w[index, 4])
    density_sqrt_L = numpy.sqrt(w[2, 0])
    density_sqrt_R = numpy.sqrt(w[3, 0])
    density_sqrt_bar = (density_sqrt_L + density_sqrt_R) / 2
    velocity = (w[2, 1] / density_sqrt_L + w[3, 1] / density_sqrt_R) / (
        2 * density_sqrt_bar
    )
    H = (
        (pressure[2] + w[2, 4]) / density_sqrt_L
        + (pressure[3] + w[3, 4]) / density_sqrt_R
    ) / (2 * density_sqrt_bar)

    velocity_squared = velocity**2
    gamma_m1 = gamma - 1
    sound_speed_squared = gamma_m1 * (H - velocity_squared / 2)
    sound_speed_squared_inverse = 1 / sound_speed_squared

    RV = numpy.zeros((5, 5))
    RV[0, 0] = 1
    RV[0, 3] = 1
    RV[0, 4] = 1

    RV[1, 0] = velocity - sound_speed_squared
    RV[1, 3] = velocity
    RV[1, 4] = velocity + sound_speed_squared

    RV[2, 1] = 1

    RV[3, 2] = 1

    RV[4, 0] = H - velocity * sound_speed_squared
    RV[4, 3] = velocity_squared / 2
    RV[4, 4] = H + velocity * sound_speed_squared

    LV = numpy.zeros((5, 5))
    LV[0, 0] = (
        sound_speed_squared_inverse / 2 * (velocity + gamma_m1 / 2 * velocity_squared)
    )
    LV[0, 1] = -sound_speed_squared_inverse / 2 * (gamma_m1 * velocity + 1)
    LV[0, 4] = gamma_m1 / 2 * sound_speed_squared_inverse

    LV[1, 2] = 1

    LV[2, 3] = 1

    LV[3, 0] = -gamma_m1 * sound_speed_squared_inverse * (velocity_squared - H)
    LV[3, 1] = velocity * gamma_m1 * sound_speed_squared_inverse
    LV[3, 4] = -gamma_m1 * sound_speed_squared_inverse

    LV[4, 0] = (
        -sound_speed_squared_inverse / 2 * (velocity - gamma_m1 / 2 * velocity_squared)
    )
    LV[4, 1] = -sound_speed_squared_inverse / 2 * (gamma_m1 * velocity - 1)
    LV[4, 4] = gamma_m1 / 2 * sound_speed_squared_inverse

    alpha = 0
    flux = numpy.zeros((6, 5))
    for index in range(6):
        velocity = w[index, 1] / w[index, 0]
        flux[index, 0] = w[index, 1]
        flux[index, 1] = velocity * w[index, 1] + pressure[index]
        flux[index, 2] = velocity * w[index, 2]
        flux[index, 3] = velocity * w[index, 3]
        flux[index, 4] = velocity * (w[index, 4] + pressure[index])
        sound_speed = math.sqrt(gamma * pressure[index] / w[index, 0])
        alpha = max(alpha, abs(velocity) + sound_speed)

    flux_surface = numpy.zeros((5, 5))
    for index in range(5):
        for n in range(5):
            flux_surface[index, n] = (flux[index, n] + alpha * w[index, n]) / 2
    flux_projected = numpy.zeros((5, 5))
    for row in range(5):
        for col in range(5):
            flux_projected[row, col] = (
                LV[col, 0] * flux_surface[row, 0]
                + LV[col, 1] * flux_surface[row, 1]
                + LV[col, 2] * flux_surface[row, 2]
                + LV[col, 3] * flux_surface[row, 3]
                + LV[col, 4] * flux_surface[row, 4]
            )

    flux_at_face = numpy.zeros(5)
    for n in range(5):
        beta = [
            None,
            (
                beta_coefficient
                * (
                    flux_projected[2, n]
                    - 2 * flux_projected[3, n]
                    + flux_projected[4, n]
                )
                ** 2
            )
            + (
                (
                    3 * flux_projected[2, n]
                    - 4 * flux_projected[3, n]
                    + flux_projected[4, n]
                )
                ** 2
            )
            / 4,
            beta_coefficient
            * (flux_projected[1, n] - 2 * flux_projected[2, n] + flux_projected[3, n])
            ** 2
            + ((flux_projected[1, n] - flux_projected[3, n]) ** 2) / 4,
            beta_coefficient
            * (flux_projected[0, n] - 2 * flux_projected[1, n] + flux_projected[2, n])
            ** 2
            + (
                (
                    flux_projected[0, n]
                    - 4 * flux_projected[1, n]
                    + 3 * flux_projected[2, n]
                )
                ** 2
            )
            / 4,
        ]
        weight = [
            None,
            0.3 / ((epsilon + beta[1]) ** 2),
            0.6 / ((epsilon + beta[2]) ** 2),
            0.1 / ((epsilon + beta[3]) ** 2),
        ]
        f = [
            None,
            1 / 3 * flux_projected[2, n]
            + 5 / 6 * flux_projected[3, n]
            - 1 / 6 * flux_projected[4, n],
            -1 / 6 * flux_projected[1, n]
            + 5 / 6 * flux_projected[2, n]
            + 1 / 3 * flux_projected[3, n],
            1 / 3 * flux_projected[0, n]
            - 7 / 6 * flux_projected[1, n]
            + 11 / 6 * flux_projected[2, n],
        ]
        flux_at_face[n] = (f[1] * weight[1] + f[2] * weight[2] + f[3] * weight[3]) / (
            weight[1] + weight[2] + weight[3]
        )

    for index in range(5):
        for n in range(5):
            flux_surface[index, n] = (
                flux[index + 1, n] - alpha * w[(index + 1), n]
            ) / 2
    for row in range(5):
        for col in range(5):
            flux_projected[row, col] = (
                LV[col, 0] * flux_surface[row, 0]
                + LV[col, 1] * flux_surface[row, 1]
                + LV[col, 2] * flux_surface[row, 2]
                + LV[col, 3] * flux_surface[row, 3]
                + LV[col, 4] * flux_surface[row, 4]
            )

    for n in range(5):
        beta = [
            None,
            (
                beta_coefficient
                * (
                    flux_projected[2, n]
                    - 2 * flux_projected[3, n]
                    + flux_projected[4, n]
                )
                ** 2
            )
            + (
                (
                    3 * flux_projected[2, n]
                    - 4 * flux_projected[3, n]
                    + flux_projected[4, n]
                )
                ** 2
            )
            / 4,
            beta_coefficient
            * (flux_projected[1, n] - 2 * flux_projected[2, n] + flux_projected[3, n])
            ** 2
            + ((flux_projected[1, n] - flux_projected[3, n]) ** 2) / 4,
            beta_coefficient
            * (flux_projected[0, n] - 2 * flux_projected[1, n] + flux_projected[2, n])
            ** 2
            + (
                (
                    flux_projected[0, n]
                    - 4 * flux_projected[1, n]
                    + 3 * flux_projected[2, n]
                )
                ** 2
            )
            / 4,
        ]
        weight = [
            None,
            0.1 / ((epsilon + beta[1]) ** 2),
            0.6 / ((epsilon + beta[2]) ** 2),
            0.3 / ((epsilon + beta[3]) ** 2),
        ]
        f = [
            None,
            11 / 6 * flux_projected[2, n]
            - 7 / 6 * flux_projected[3, n]
            + 1 / 3 * flux_projected[4, n],
            1 / 3 * flux_projected[1, n]
            + 5 / 6 * flux_projected[2, n]
            - 1 / 6 * flux_projected[3, n],
            -1 / 6 * flux_projected[0, n]
            + 5 / 6 * flux_projected[1, n]
            + 1 / 3 * flux_projected[2, n],
        ]
        flux_at_face[n] += (f[1] * weight[1] + f[2] * weight[2] + f[3] * weight[3]) / (
            weight[1] + weight[2] + weight[3]
        )

    for n in range(5):
        flux_face[n] = (
            RV[n, 0] * flux_at_face[0]
            + RV[n, 1] * flux_at_face[1]
            + RV[n, 2] * flux_at_face[2]
            + RV[n, 3] * flux_at_face[3]
            + RV[n, 4] * flux_at_face[4]
        )
