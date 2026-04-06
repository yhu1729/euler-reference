# https://doi.org/10.1137/S1064827595291819

import argparse
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
    find_flux_2D,
    find_energy_2D,
    find_pressure_2D,
    pack_2D,
    pack_2D_boundary_outflow,
)
from plot import (
    plot_density,
    plot_momentum,
    plot_energy,
    plot_pressure,
    plot_velocity,
    plot_entropy,
    plot_vorticity,
)
from snapshot import save_snapshot


def get_subvector(x, n):
    return N_VGetArrayPointer(N_VGetSubvector_ManyVector(x, n))


@numba.njit(fastmath=True, cache=True)
def find_flux(flux, w, N_x, N_y, gamma, density, momentum_x, momentum_y, energy):
    for index_x in range(3, N_x - 2):
        for index_y in range(3, N_y - 2):
            pack_2D(w, index_x, index_y, density, momentum_x, momentum_y, energy)
            find_flux_2D(gamma, w, flux[index_x, index_y, :, :])


@numba.njit(fastmath=True, cache=True)
def update_flux(
    density_dot, momentum_x_dot, momentum_y_dot, energy_dot, flux, N_x, N_y, dx, dy
):
    for index_x in range(N_x):
        for index_y in range(N_y):
            # x-direction
            density_dot[index_x, index_y] = (
                -(flux[index_x + 1, index_y, 0, 0] - flux[index_x, index_y, 0, 0]) / dx
            )
            momentum_x_dot[index_x, index_y] = (
                -(flux[index_x + 1, index_y, 0, 1] - flux[index_x, index_y, 0, 1]) / dx
            )
            momentum_y_dot[index_x, index_y] = (
                -(flux[index_x + 1, index_y, 0, 2] - flux[index_x, index_y, 0, 2]) / dx
            )
            energy_dot[index_x, index_y] = (
                -(flux[index_x + 1, index_y, 0, 4] - flux[index_x, index_y, 0, 4]) / dx
            )

            # y-direction
            density_dot[index_x, index_y] += (
                -(flux[index_x, index_y + 1, 1, 0] - flux[index_x, index_y, 1, 0]) / dy
            )
            momentum_x_dot[index_x, index_y] += (
                -(flux[index_x, index_y + 1, 1, 1] - flux[index_x, index_y, 1, 1]) / dy
            )
            momentum_y_dot[index_x, index_y] += (
                -(flux[index_x, index_y + 1, 1, 2] - flux[index_x, index_y, 1, 2]) / dy
            )
            energy_dot[index_x, index_y] += (
                -(flux[index_x, index_y + 1, 1, 4] - flux[index_x, index_y, 1, 4]) / dy
            )


class Problem:
    def __init__(self, case_id, N_x, N_y):
        self.case_id = case_id
        self.gamma = 1.4

        if self.case_id == 1:
            self.t_stop = 0.2
            self.rho = [[0.5197, 1], [0.1072, 0.2579]]
            self.u = [[-0.7259, 0], [-0.7259, 0]]
            self.v = [[0, 0], [-1.4045, -1.4045]]
            self.p = [[0.4, 1], [0.0439, 0.15]]
        elif self.case_id == 2:
            self.t_stop = 0.2
            self.rho = [[0.5197, 1], [1, 0.5197]]
            self.u = [[-0.7259, 0], [-0.7259, 0]]
            self.v = [[0, 0], [-0.7259, -0.7259]]
            self.p = [[0.4, 1], [1, 0.4]]
        elif self.case_id == 5:
            self.t_stop = 0.23
            self.rho = [[2, 1], [1, 3]]
            self.u = [[-0.75, -0.75], [0.75, 0.75]]
            self.v = [[0.5, -0.5], [0.5, -0.5]]
            self.p = [[1, 1], [1, 1]]
        elif self.case_id == 7:
            self.t_stop = 0.25
            self.rho = [[0.5197, 1], [0.8, 0.5197]]
            self.u = [[-0.6259, 0.1], [0.1, 0.1]]
            self.v = [[0.1, 0.1], [0.1, -0.6259]]
            self.p = [[0.4, 1], [0.4, 0.4]]
        elif self.case_id == 8:
            self.t_stop = 0.25
            self.rho = [[1, 0.5197], [0.8, 1]]
            self.u = [[-0.6259, 0.1], [0.1, 0.1]]
            self.v = [[0.1, 0.1], [0.1, -0.6259]]
            self.p = [[1, 0.4], [1, 1]]
        elif self.case_id == 9:
            self.t_stop = 0.3
            self.rho = [[2, 1], [1.039, 0.5197]]
            self.u = [[0, 0], [0, 0]]
            self.v = [[-0.3, 0.3], [-0.8123, -0.4259]]
            self.p = [[1, 1], [0.4, 0.4]]
        elif self.case_id == 10:
            self.t_stop = 0.15
            self.rho = [[0.5, 1], [0.2281, 0.4562]]
            self.u = [[0, 0], [0, 0]]
            self.v = [[0.6076, 0.4297], [-0.6076, -0.4297]]
            self.p = [[1, 1], [0.3333, 0.3333]]
        elif self.case_id == 11:
            self.t_stop = 0.3
            self.rho = [[0.5313, 1], [0.8, 0.5313]]
            self.u = [[0.8276, 0.1], [0.1, 0.1]]
            self.v = [[0, 0], [0, 0.7276]]
            self.p = [[0.4, 1], [0.4, 0.4]]
        elif self.case_id == 13:
            self.t_stop = 0.3
            self.rho = [[2, 1], [1.0625, 0.5313]]
            self.u = [[0, 0], [0, 0]]
            self.v = [[0.3, -0.3], [0.8145, 0.4276]]
            self.p = [[1, 1], [0.4, 0.4]]
        elif self.case_id == 14:
            self.t_stop = 0.1
            self.rho = [[1, 2], [0.4736, 0.9474]]
            self.u = [[0, 0], [0, 0]]
            self.v = [[-1.2172, -0.5606], [1.2172, 1.1606]]
            self.p = [[8, 8], [2.6667, 2.6667]]
        elif self.case_id == 16:
            self.t_stop = 0.2
            self.rho = [[1.0222, 0.5313], [0.8, 1]]
            self.u = [[-0.7179, 0.1], [0.1, 0.1]]
            self.v = [[0.1, 0.1], [0.1, 0.8276]]
            self.p = [[1, 0.4], [1, 1]]
        elif self.case_id == 18:
            self.t_stop = 0.2
            self.rho = [[2, 1], [1.0625, 0.5197]]
            self.u = [[0, 0], [0, 0]]
            self.v = [[-0.3, 1], [0.2145, 0.2741]]
            self.p = [[1, 1], [0.4, 0.4]]
        elif self.case_id == 19:
            self.t_stop = 0.3
            self.rho = [[2, 1], [1.0625, 0.5197]]
            self.u = [[0, 0], [0, 0]]
            self.v = [[-0.3, 0.3], [0.2145, -0.4259]]
            self.p = [[1, 1], [0.4, 0.4]]
        else:
            raise RuntimeError("Invalid parameter `case_id`")

        self.N_x = N_x
        self.N_y = N_y
        self.dx = 1 / N_x
        self.dy = 1 / N_y
        self.x = numpy.linspace(0, 1, N_x, endpoint=False) + self.dx / 2
        self.y = numpy.linspace(0, 1, N_y, endpoint=False) + self.dy / 2
        self.X, self.Y = numpy.meshgrid(self.x, self.y, indexing="ij")

        self.w = numpy.zeros((2, 6, 5))
        self.flux = numpy.zeros((N_x + 1, N_y + 1, 2, 5))

        self.n_step = 0

    def set_initial_condition(self, y):
        density = get_subvector(y, 0)
        momentum_x = get_subvector(y, 1)
        momentum_y = get_subvector(y, 2)
        energy = get_subvector(y, 3)

        density = numpy.reshape(density, (self.N_x, self.N_y))
        momentum_x = numpy.reshape(momentum_x, (self.N_x, self.N_y))
        momentum_y = numpy.reshape(momentum_y, (self.N_x, self.N_y))
        energy = numpy.reshape(energy, (self.N_x, self.N_y))

        density[:, :] = (
            (self.X < 0.5) * (self.Y >= 0.5) * self.rho[0][0]
            + (self.X >= 0.5) * (self.Y >= 0.5) * self.rho[0][1]
            + (self.X < 0.5) * (self.Y < 0.5) * self.rho[1][0]
            + (self.X >= 0.5) * (self.Y < 0.5) * self.rho[1][1]
        )
        velocity_x = (
            (self.X < 0.5) * (self.Y >= 0.5) * self.u[0][0]
            + (self.X >= 0.5) * (self.Y >= 0.5) * self.u[0][1]
            + (self.X < 0.5) * (self.Y < 0.5) * self.u[1][0]
            + (self.X >= 0.5) * (self.Y < 0.5) * self.u[1][1]
        )
        velocity_y = (
            (self.X < 0.5) * (self.Y >= 0.5) * self.v[0][0]
            + (self.X >= 0.5) * (self.Y >= 0.5) * self.v[0][1]
            + (self.X < 0.5) * (self.Y < 0.5) * self.v[1][0]
            + (self.X >= 0.5) * (self.Y < 0.5) * self.v[1][1]
        )
        momentum_x[:, :] = density * velocity_x
        momentum_y[:, :] = density * velocity_y
        pressure = (
            (self.X < 0.5) * (self.Y >= 0.5) * self.p[0][0]
            + (self.X >= 0.5) * (self.Y >= 0.5) * self.p[0][1]
            + (self.X < 0.5) * (self.Y < 0.5) * self.p[1][0]
            + (self.X >= 0.5) * (self.Y < 0.5) * self.p[1][1]
        )
        energy[:, :] = find_energy_2D(
            self.gamma, density, momentum_x, momentum_y, pressure
        )

    def pack(self, index_j, index_k, density, momentum_x, momentum_y, energy):
        pack_2D(self.w, index_j, index_k, density, momentum_x, momentum_y, energy)

    def pack_boundary(self, index_j, index_k, density, momentum_x, momentum_y, energy):
        pack_2D_boundary_outflow(
            self.w,
            self.N_x,
            self.N_y,
            index_j,
            index_k,
            density,
            momentum_x,
            momentum_y,
            energy,
        )

    def find_flux(self, flux):
        find_flux_2D(self.gamma, self.w, flux)

    def after_step(self, t, y, user_data):
        self.n_step += 1
        print(f"{self.n_step:10}: {t:.6e}, {(t/self.t_stop*100):.3f}%")

        return 0

    def f(self, t, y, y_dot, user_data):
        density = get_subvector(y, 0)
        momentum_x = get_subvector(y, 1)
        momentum_y = get_subvector(y, 2)
        energy = get_subvector(y, 3)

        density = numpy.reshape(density, (self.N_x, self.N_y))
        momentum_x = numpy.reshape(momentum_x, (self.N_x, self.N_y))
        momentum_y = numpy.reshape(momentum_y, (self.N_x, self.N_y))
        energy = numpy.reshape(energy, (self.N_x, self.N_y))

        self.flux[:, :, :, :] = 0

        find_flux(
            self.flux,
            self.w,
            self.N_x,
            self.N_y,
            self.gamma,
            density,
            momentum_x,
            momentum_y,
            energy,
        )
        # x-direction
        for index_x in range(0, 3):
            for index_y in range(self.N_y):
                self.pack_boundary(
                    index_x, index_y, density, momentum_x, momentum_y, energy
                )
                self.find_flux(self.flux[index_x, index_y, :, :])
        for index_x in range(self.N_x - 2, self.N_x + 1):
            for index_y in range(self.N_y):
                self.pack_boundary(
                    index_x, index_y, density, momentum_x, momentum_y, energy
                )
                self.find_flux(self.flux[index_x, index_y, :, :])
        # y-direction
        for index_x in range(self.N_x):
            for index_y in range(0, 3):
                self.pack_boundary(
                    index_x, index_y, density, momentum_x, momentum_y, energy
                )
                self.find_flux(self.flux[index_x, index_y, :, :])
        for index_x in range(self.N_x):
            for index_y in range(self.N_y - 2, self.N_y + 1):
                self.pack_boundary(
                    index_x, index_y, density, momentum_x, momentum_y, energy
                )
                self.find_flux(self.flux[index_x, index_y, :, :])

        density_dot = get_subvector(y_dot, 0)
        momentum_x_dot = get_subvector(y_dot, 1)
        momentum_y_dot = get_subvector(y_dot, 2)
        energy_dot = get_subvector(y_dot, 3)

        density_dot = numpy.reshape(density_dot, (self.N_x, self.N_y))
        momentum_x_dot = numpy.reshape(momentum_x_dot, (self.N_x, self.N_y))
        momentum_y_dot = numpy.reshape(momentum_y_dot, (self.N_x, self.N_y))
        energy_dot = numpy.reshape(energy_dot, (self.N_x, self.N_y))

        update_flux(
            density_dot,
            momentum_x_dot,
            momentum_y_dot,
            energy_dot,
            self.flux,
            self.N_x,
            self.N_y,
            self.dx,
            self.dy,
        )

        return 0


def plot_all(
    name,
    x,
    y,
    density,
    momentum_x,
    momentum_y,
    energy,
    pressure,
    velocity_x,
    velocity_y,
    entropy,
    t,
    t_index,
):
    plot_density(name, x, y, density, t, t_index)
    plot_momentum(name, x, y, momentum_x, momentum_y, t, t_index)
    plot_energy(name, x, y, energy, t, t_index)
    plot_pressure(name, x, y, pressure, t, t_index)
    plot_velocity(name, x, y, velocity_x, velocity_y, t, t_index)
    plot_entropy(name, x, y, entropy, t, t_index)
    plot_vorticity(name, x, y, velocity_x, velocity_y, t, t_index)


def main():
    name = pathlib.Path(__file__).stem
    parser = argparse.ArgumentParser(prog=name, description="", epilog="")
    parser.add_argument(
        "-c",
        "--case",
        type=int,
        dest="case",
        choices=[1, 2, 5, 7, 8, 9, 10, 11, 13, 14, 16, 18, 19],
        default=1,
        help="Riemann problem case ID",
    )
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
        "-nx", type=int, dest="nx", default=400, help="mesh resolution in x-direction"
    )
    parser.add_argument(
        "-ny", type=int, dest="ny", default=400, help="mesh resolution in y-direction"
    )
    arg = parser.parse_args()

    name = f"{name}_{arg.case}"

    problem = Problem(arg.case, arg.nx, arg.ny)
    _, context = SUNContext_Create(SUN_COMM_NULL)

    buffer = []
    for _ in range(4):
        buffer.append(N_VNew_Serial(arg.nx * arg.ny, context))
    y = N_VNew_ManyVector(4, buffer, context)

    problem.set_initial_condition(y)

    density = get_subvector(y, 0)
    momentum_x = get_subvector(y, 1)
    momentum_y = get_subvector(y, 2)
    energy = get_subvector(y, 3)

    density = numpy.reshape(density, (problem.N_x, problem.N_y))
    momentum_x = numpy.reshape(momentum_x, (problem.N_x, problem.N_y))
    momentum_y = numpy.reshape(momentum_y, (problem.N_x, problem.N_y))
    energy = numpy.reshape(energy, (problem.N_x, problem.N_y))

    pressure = find_pressure_2D(problem.gamma, density, momentum_x, momentum_y, energy)
    velocity_x = momentum_x / density
    velocity_y = momentum_y / density
    entropy = numpy.log(pressure / density**problem.gamma)

    plot_all(
        name,
        problem.X,
        problem.Y,
        density,
        momentum_x,
        momentum_y,
        energy,
        pressure,
        velocity_x,
        velocity_y,
        entropy,
        0,
        0,
    )

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
        momentum_x = get_subvector(y, 1)
        momentum_y = get_subvector(y, 2)
        energy = get_subvector(y, 3)

        density = numpy.reshape(density, (problem.N_x, problem.N_y))
        momentum_x = numpy.reshape(momentum_x, (problem.N_x, problem.N_y))
        momentum_y = numpy.reshape(momentum_y, (problem.N_x, problem.N_y))
        energy = numpy.reshape(energy, (problem.N_x, problem.N_y))

        pressure = find_pressure_2D(
            problem.gamma, density, momentum_x, momentum_y, energy
        )
        velocity_x = momentum_x / density
        velocity_y = momentum_y / density
        entropy = numpy.log(pressure / density**problem.gamma)

        plot_all(
            name,
            problem.X,
            problem.Y,
            density,
            momentum_x,
            momentum_y,
            energy,
            pressure,
            velocity_x,
            velocity_y,
            entropy,
            t_stop,
            n,
        )

    save_snapshot(
        name,
        arg,
        problem,
        density,
        momentum_x,
        momentum_y,
        energy,
        pressure,
        entropy,
    )


if __name__ == "__main__":
    main()
