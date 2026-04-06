import argparse
import numba
import numpy
import pathlib
from sundials4py.core import N_VGetArrayPointer

from weno import GHOST, find_energy
from stepper import parse_input, Stepper


def setup_initial_condition(y, X, Y, config):
    Q = N_VGetArrayPointer(y)
    Q = numpy.reshape(Q, (4, config.N_x + 2 * GHOST, config.N_y + 2 * GHOST))

    rho = (X < 0) * 1 + (X >= 0) * 0.125
    u = numpy.zeros_like(X)
    v = numpy.zeros_like(X)
    p = (X < 0) * 1 + (X >= 0) * 0.1

    Q[0] = rho
    Q[1] = rho * u
    Q[2] = rho * v
    Q[3] = find_energy(Q[0], Q[1], Q[2], p, gamma=config.gamma)


def main():
    name = pathlib.Path(__file__).stem
    default = {
        "L_x": 10.0,
        "L_y": 2.0,
        "S_x": -5.0,
        "S_y": -1.0,
        "N_x": 100,
        "N_y": 10,
        "boundary_x": "outflow",
        "t_final": 2.0,
        "plot_ratio": 0.4,
    }
    config = parse_input(name, default=default)

    stepper = Stepper(name, config, setup_initial_condition)
    stepper.start()


if __name__ == "__main__":
    main()
