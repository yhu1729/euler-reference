# https://doi.org/10.1002/fld.1650110509
# https://doi.org/10.1137/S1064827502402120

import argparse
import numpy
import pathlib
from sundials4py.core import N_VGetArrayPointer

from weno import GHOST, find_energy
from stepper import parse_input, Stepper


def setup_initial_condition(y, X, Y, config):
    Q = N_VGetArrayPointer(y)
    Q = numpy.reshape(Q, (4, config.N_x + 2 * GHOST, config.N_y + 2 * GHOST))

    rho = numpy.ones_like(X)

    x_center = config.L_x / 2 + config.S_x
    y_center = config.L_y / 2 + config.S_y
    r = numpy.sqrt((X - x_center) ** 2 + (Y - y_center) ** 2)
    r_safe = numpy.where(r == 0, 1, r)
    velocity = (r < 0.2) * (5 * r) + (r >= 0.2) * (r < 0.4) * (2 - 5 * r)
    u = -velocity * ((Y - y_center) / r_safe)
    v = velocity * ((X - x_center) / r_safe)
    p = (
        (r < 0.2) * (5 + 25 / 2 * r**2)
        + ((r >= 0.2) & (r < 0.4))
        * (9 + 25 / 2 * r**2 - 20 * r + 4 * numpy.log(5 * r_safe))
        + (r >= 0.4) * (3 + 4 * numpy.log(2))
    )

    Q[0] = rho
    Q[1] = rho * u
    Q[2] = rho * v
    Q[3] = find_energy(Q[0], Q[1], Q[2], p, gamma=config.gamma)


def main():
    name = pathlib.Path(__file__).stem
    default = {
        "N_x": 80,
        "N_y": 80,
        "boundary_x": "outflow",
        "boundary_y": "outflow",
        "t_final": 3.0,
    }
    config = parse_input(name, default=default)

    stepper = Stepper(name, config, setup_initial_condition)
    stepper.start()


if __name__ == "__main__":
    main()
