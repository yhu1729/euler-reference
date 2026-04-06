# https://doi.org/10.1016/0021-9991(89)90222-2

import argparse
from matplotlib import pyplot
import numba
import numpy
import pathlib
from sundials4py.core import (
    N_VClone,
    N_VGetArrayPointer,
    N_VGetSubvector_ManyVector,
    N_VNew_ManyVector,
    N_VNew_Serial,
    SUNContext_Create,
    SUN_COMM_NULL,
)
from sundials4py.arkode import (
    ARK_NORMAL,
    ARKodeClearStopTime,
    ARKodeEvolve,
    ARKodeSStolerances,
    ARKodeSetMaxNumSteps,
    ARKodeSetOrder,
    ARKodeSetPostprocessStepFn,
    ARKodeSetStopTime,
    ERKStepCreate,
    LSRKStepCreateSSP,
    LSRKStepSetSSPMethodByName,
)

from fv_weno3 import (
    find_flux_1D,
    find_energy_1D,
    find_pressure_1D,
    pack_1D,
    pack_1D_boundary_outflow,
)
from snapshot import save_snapshot


def get_subvector(x, n):
    return N_VGetArrayPointer(N_VGetSubvector_ManyVector(x, n))


@numba.njit(fastmath=True)
def find_flux(flux, w, N, gamma, density, momentum, energy):
    for index in range(3, N - 2):
        pack_1D(w, index, density, momentum, energy)
        find_flux_1D(gamma, w, flux[index, :])


@numba.njit(fastmath=True)
def update_flux(density_dot, momentum_dot, energy_dot, flux, N, dx):
    for index in range(N):
        density_dot[index] = -(flux[index + 1, 0] - flux[index, 0]) / dx
        momentum_dot[index] = -(flux[index + 1, 1] - flux[index, 1]) / dx
        energy_dot[index] = -(flux[index + 1, 4] - flux[index, 4]) / dx


class Problem:
    def __init__(self, N):
        self.gamma = 1.4
        self.t_stop = 1.8

        self.N = N
        self.dx = 15 / N
        self.x = numpy.linspace(-10, 5, N, endpoint=False) + self.dx / 2

        self.w = numpy.zeros((6, 5))
        self.flux = numpy.zeros((N + 1, 5))

        self.n_step = 0

    def set_initial_condition(self, y):
        density = get_subvector(y, 0)
        momentum = get_subvector(y, 1)
        energy = get_subvector(y, 2)

        density[:] = (self.x < -4) * (27 / 7) + (self.x >= -4) * (
            1 + 0.2 * numpy.sin(5 * self.x)
        )
        velocity = (self.x < -4) * (4 * numpy.sqrt(35) / 9) + (self.x >= -4) * 0
        momentum[:] = density * velocity
        pressure = (self.x < -4) * (31 / 3) + (self.x >= -4) * 1
        energy[:] = find_energy_1D(self.gamma, density, momentum, pressure)

    def pack(self, index_j, density, momentum, energy):
        pack_1D(self.w, index_j, density, momentum, energy)

    def pack_boundary(self, index_j, density, momentum, energy):
        pack_1D_boundary_outflow(self.w, self.N, index_j, density, momentum, energy)

    def find_flux(self, flux):
        find_flux_1D(self.gamma, self.w, flux)

    def after_step(self, t, y, user_data):
        self.n_step += 1
        print(f"{self.n_step:10}: {t:.6e}, {(t/self.t_stop*100):.3f}%")

        return 0

    def f(self, t, y, y_dot, user_data):
        density = get_subvector(y, 0)
        momentum = get_subvector(y, 1)
        energy = get_subvector(y, 2)

        self.flux[:, :] = 0

        find_flux(self.flux, self.w, self.N, self.gamma, density, momentum, energy)
        for index in range(0, 3):
            self.pack_boundary(index, density, momentum, energy)
            self.find_flux(self.flux[index, :])
        for index in range(self.N - 2, self.N + 1):
            self.pack_boundary(index, density, momentum, energy)
            self.find_flux(self.flux[index, :])

        density_dot = get_subvector(y_dot, 0)
        momentum_dot = get_subvector(y_dot, 1)
        energy_dot = get_subvector(y_dot, 2)

        update_flux(density_dot, momentum_dot, energy_dot, self.flux, self.N, self.dx)

        return 0


def main():
    name = pathlib.Path(__file__).stem
    parser = argparse.ArgumentParser(prog=name, description="", epilog="")
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        dest="method",
        choices=["ssp", "erk"],
        default="ssp",
        help="time integration method",
    )
    parser.add_argument(
        "-nx", type=int, dest="nx", default=1000, help="mesh resolution in x-direction"
    )
    arg = parser.parse_args()

    N = arg.nx

    problem = Problem(N)
    _, context = SUNContext_Create(SUN_COMM_NULL)

    buffer = []
    for _ in range(3):
        buffer.append(N_VNew_Serial(N, context))
    y = N_VNew_ManyVector(3, buffer, context)

    problem.set_initial_condition(y)

    density = get_subvector(y, 0)
    momentum = get_subvector(y, 1)
    energy = get_subvector(y, 2)
    velocity = momentum / density
    pressure = find_pressure_1D(problem.gamma, density, momentum, energy)

    pyplot.rcParams["text.usetex"] = True
    figure, ax = pyplot.subplots(nrows=5, ncols=5, figsize=(40, 30), dpi=200)
    ax[0, 0].plot(problem.x, density)
    ax[1, 0].plot(problem.x, momentum)
    ax[2, 0].plot(problem.x, energy)
    ax[3, 0].plot(problem.x, velocity)
    ax[4, 0].plot(problem.x, pressure)
    ax[0, 0].set_title(rf"$t$ = {0:.6E}")
    ax[0, 0].set_ylabel("density")
    ax[1, 0].set_ylabel("momentum")
    ax[2, 0].set_ylabel("energy")
    ax[3, 0].set_ylabel("velocity")
    ax[4, 0].set_ylabel("pressure")

    if arg.method == "ssp":
        stepper = LSRKStepCreateSSP(problem.f, 0, y, context)
        LSRKStepSetSSPMethodByName(stepper.get(), "ARKODE_LSRK_SSP_10_4")
    elif arg.method == "erk":
        stepper = ERKStepCreate(problem.f, 0, y, context)
        ARKodeSetOrder(stepper.get(), 4)

    ARKodeSStolerances(stepper.get(), 1e-4, 1e-11)
    ARKodeSetMaxNumSteps(stepper.get(), -1)
    ARKodeSetPostprocessStepFn(stepper.get(), problem.after_step)

    for n in range(1, 4 + 1):
        t_stop = problem.t_stop / 4 * n
        ARKodeSetStopTime(stepper.get(), t_stop)
        ARKodeEvolve(stepper.get(), t_stop, y, ARK_NORMAL)
        ARKodeClearStopTime(stepper.get())

        density = get_subvector(y, 0)
        momentum = get_subvector(y, 1)
        energy = get_subvector(y, 2)
        velocity = momentum / density
        pressure = find_pressure_1D(problem.gamma, density, momentum, energy)

        ax[0, n].plot(problem.x, density)
        ax[1, n].plot(problem.x, momentum)
        ax[2, n].plot(problem.x, energy)
        ax[3, n].plot(problem.x, velocity)
        ax[4, n].plot(problem.x, pressure)
        ax[0, n].set_title(rf"$t$ = {t_stop:.6E}")

    figure.tight_layout()
    pyplot.savefig(f"{name}.svg")

    save_snapshot(name, arg, problem, density, momentum, energy, velocity, pressure)


if __name__ == "__main__":
    main()
