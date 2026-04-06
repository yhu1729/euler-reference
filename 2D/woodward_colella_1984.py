# https://doi.org/10.1016/0021-9991(84)90142-6
# https://doi.org/10.1006/jcph.1996.0130

import argparse
import numba
import numpy
import pathlib
from sundials4py.core import N_VGetArrayPointer

from weno import (
    GHOST,
    find_energy,
    find_pressure,
    apply_boundary_condition_inflow,
    setup_wall,
)
from stepper import parse_input, Stepper


def find_wall(N_x, N_y, config):
    mask = numpy.ones((N_x + 2 * GHOST, N_y + 2 * GHOST), dtype=bool)
    for i in range(N_x // 5 + GHOST, N_x + 2 * GHOST):
        for j in range(N_y // 5 + GHOST):
            mask[i, j] = False
    return mask


def setup_initial_condition(y, X, Y, config):
    Q = N_VGetArrayPointer(y)
    Q = numpy.reshape(Q, (4, config.N_x + 2 * GHOST, config.N_y + 2 * GHOST))

    rho = config.gamma * numpy.ones_like(X)
    u = 3 * numpy.ones_like(X)
    v = numpy.zeros_like(X)
    p = numpy.ones_like(X)

    Q[0] = rho
    Q[1] = rho * u
    Q[2] = rho * v

    mask = find_wall(config.N_x, config.N_y, config)

    Q[1, ~mask] = 0.0
    Q[2, ~mask] = 0.0

    Q[3] = find_energy(Q[0], Q[1], Q[2], p, gamma=config.gamma)


@numba.njit
def update_boundary(Q, gamma=1.4):
    E = find_energy(gamma, 3 * gamma, 0, 1, gamma=gamma)
    Q_infinity = numpy.array([gamma, 3 * gamma, 0, E])

    Q = apply_boundary_condition_inflow(Q, Q_infinity, "x", -1)

    return Q


def main():
    name = pathlib.Path(__file__).stem
    default = {
        "L_x": 3.0,
        "L_y": 1.0,
        "N_x": 240,
        "N_y": 80,
        "boundary_x": "outflow",
        "boundary_y": "reflective",
        "use_limiter": True,
        "use_JST_dissipation": True,
        "t_final": 4.0,
        "plot_ratio": 0.4,
    }
    config = parse_input(name, default=default)

    wall_mask = find_wall(config.N_x, config.N_y, config)
    wall = setup_wall(wall_mask)
    stepper = Stepper(
        name,
        config,
        setup_initial_condition,
        update_boundary=update_boundary,
        wall=wall,
        plot_wall=wall_mask,
    )
    stepper.start()


if __name__ == "__main__":
    main()
