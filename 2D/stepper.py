import argparse
import h5py
import numpy
from sundials4py.core import (
    N_VClone,
    N_VGetArrayPointer,
    N_VNew_Serial,
    SUNContext_Create,
    SUN_COMM_NULL,
)
from sundials4py.arkode import (
    ARK_NORMAL,
    ARKodeClearStopTime,
    ARKodeEvolve,
    ARKodeGetCurrentStep,
    ARKodeSStolerances,
    ARKodeSetFixedStep,
    ARKodeSetMaxNumSteps,
    ARKodeSetOrder,
    ARKodeSetPostprocessStepFn,
    ARKodeSetStopTime,
    ERKStepCreate,
    LSRKStepCreateSSP,
    LSRKStepSetSSPMethodByName,
)

from weno import (
    GHOST,
    find_RHS,
    find_pressure,
)
from plot import (
    plot_density,
    plot_energy,
    plot_entropy,
    plot_momentum,
    plot_pressure,
    plot_velocity,
    plot_vorticity,
)


def s2b(s):
    return str(s).lower() in ["true", "yes", "y", "1"]


def parse_input(name, default={}):
    parser = argparse.ArgumentParser(prog=name, description="", epilog="")

    # physics
    parser.add_argument(
        "--gamma",
        type=float,
        dest="gamma",
        default=default.get("gamma", 1.4),
        help="",
    )

    # domain
    parser.add_argument(
        "--Lx",
        type=float,
        dest="L_x",
        default=default.get("L_x", 1.0),
        help="",
    )
    parser.add_argument(
        "--Ly",
        type=float,
        dest="L_y",
        default=default.get("L_y", 1.0),
        help="",
    )
    parser.add_argument(
        "--Sx",
        type=float,
        dest="S_x",
        default=default.get("S_x", 0.0),
        help="",
    )
    parser.add_argument(
        "--Sy",
        type=float,
        dest="S_y",
        default=default.get("S_y", 0.0),
        help="",
    )
    parser.add_argument(
        "--Nx",
        type=int,
        dest="N_x",
        default=default.get("N_x", 100),
        help="mesh resolution in x-direction",
    )
    parser.add_argument(
        "--Ny",
        type=int,
        dest="N_y",
        default=default.get("N_y", 100),
        help="mesh resolution in y-direction",
    )

    # boundary conditions
    parser.add_argument(
        "--boundary-x",
        type=str,
        dest="boundary_x",
        default=default.get("boundary_x", "periodic"),
        help="",
    )
    parser.add_argument(
        "--boundary-y",
        type=str,
        dest="boundary_y",
        default=default.get("boundary_y", "periodic"),
        help="",
    )

    # solver
    parser.add_argument(
        "--method",
        type=str,
        dest="method",
        choices=["ssp"],
        default=default.get("method", "ssp"),
        help="time integration method",
    )
    parser.add_argument(
        "--dt",
        type=float,
        dest="dt",
        default=default.get("dt", 0.0),
        help="",
    )
    parser.add_argument(
        "--weno-epsilon",
        type=float,
        dest="weno_epsilon",
        default=default.get("weno_epsilon", 1e-6),
        help="",
    )
    parser.add_argument(
        "--use-limiter",
        type=s2b,
        dest="use_limiter",
        default=default.get("use_limiter", False),
        help="",
    )
    parser.add_argument(
        "--use-dissipation",
        type=s2b,
        dest="use_dissipation",
        default=default.get("use_dissipation", True),
        help="",
    )
    parser.add_argument(
        "--use-JST-dissipation",
        type=s2b,
        dest="use_JST_dissipation",
        default=default.get("use_JST_dissipation", False),
        help="",
    )
    parser.add_argument(
        "--JST-kappa-2",
        type=float,
        dest="JST_kappa_2",
        default=default.get("JST_kappa_2", 1 / 4),
        help="",
    )
    parser.add_argument(
        "--JST-kappa-4",
        type=float,
        dest="JST_kappa_4",
        default=default.get("JST_kappa_4", 1 / 256),
        help="",
    )

    # snapshot
    parser.add_argument(
        "--t-initial",
        type=float,
        dest="t_initial",
        default=default.get("t_initial", 0.0),
        help="",
    )
    parser.add_argument(
        "--t-final",
        type=float,
        dest="t_final",
        default=default.get("t_final", 1.0),
        help="",
    )
    parser.add_argument(
        "--snapshot",
        type=int,
        dest="snapshot",
        default=default.get("snapshot", 10),
        help="",
    )
    parser.add_argument(
        "--plot-ratio",
        type=float,
        dest="plot_ratio",
        default=default.get("plot_ratio", 1.0),
        help="",
    )
    parser.add_argument(
        "--plot-contour-level",
        type=float,
        dest="plot_contour_level",
        default=default.get("plot_contour_level", 20),
        help="",
    )

    return parser.parse_args()


class Stepper:
    def __init__(
        self,
        name,
        config,
        setup_initial_condition,
        update_boundary=None,
        wall=None,
        plot_wall=None,
    ):
        self.name = name
        self.config = config
        self.setup_initial_condition = setup_initial_condition
        self.update_boundary = update_boundary
        self.wall = wall
        self.plot_wall = plot_wall

        self.dx = self.config.L_x / self.config.N_x
        self.dy = self.config.L_y / self.config.N_y
        self.x = (
            numpy.linspace(
                0 - GHOST * self.dx,
                self.config.L_x + GHOST * self.dx,
                self.config.N_x + 2 * GHOST,
                endpoint=False,
            )
            + 0.5 * self.dx
            + self.config.S_x
        )
        self.y = (
            numpy.linspace(
                0 - GHOST * self.dy,
                self.config.L_y + GHOST * self.dy,
                self.config.N_y + 2 * GHOST,
                endpoint=False,
            )
            + 0.5 * self.dy
            + self.config.S_y
        )
        self.X, self.Y = numpy.meshgrid(self.x, self.y, indexing="ij")

        self.stepper = None
        self.n_step = 0
        self.t = 0
        self.dt = []

    def save_snapshot(self, y, snapshot):
        Q = N_VGetArrayPointer(y)
        Q = numpy.reshape(
            Q, (4, self.config.N_x + 2 * GHOST, self.config.N_y + 2 * GHOST)
        )

        density = Q[0, GHOST:-GHOST, GHOST:-GHOST]
        momentum_x = Q[1, GHOST:-GHOST, GHOST:-GHOST]
        momentum_y = Q[2, GHOST:-GHOST, GHOST:-GHOST]
        energy = Q[3, GHOST:-GHOST, GHOST:-GHOST]
        pressure = find_pressure(
            density, momentum_x, momentum_y, energy, gamma=self.config.gamma
        )
        velocity_x = momentum_x / density
        velocity_y = momentum_y / density
        entropy = numpy.log(
            numpy.maximum(1e-16, pressure)
            / (numpy.maximum(1e-16, density) ** self.config.gamma)
        )

        with h5py.File(f"{self.name}.{snapshot:03}.h5", "w") as file:
            group = file.create_group("data")

            group_input = group.create_group("input")
            group_input.create_dataset("gamma", data=self.config.gamma)
            group_input.create_dataset("method", data=self.config.method)
            group_input.create_dataset("N_x", data=self.config.N_x)
            group_input.create_dataset("N_y", data=self.config.N_y)

            group_output = group.create_group("output")
            group_output.create_dataset("t", data=self.t)
            group_output.create_dataset("dt", data=self.dt)

            group_mesh = group.create_group("mesh")
            group_mesh.create_dataset("x", data=self.x)
            group_mesh.create_dataset("y", data=self.y)

            group_profile = group.create_group("profile")
            group_profile.create_dataset("density", data=density)
            group_profile.create_dataset("momentum_x", data=momentum_x)
            group_profile.create_dataset("momentum_y", data=momentum_y)
            group_profile.create_dataset("energy", data=energy)
            group_profile.create_dataset("pressure", data=pressure)
            group_profile.create_dataset("velocity_x", data=velocity_x)
            group_profile.create_dataset("velocity_y", data=velocity_y)
            group_profile.create_dataset("entropy", data=entropy)

    def plot(self, y, snapshot):
        Q = N_VGetArrayPointer(y)
        Q = numpy.reshape(
            Q, (4, self.config.N_x + 2 * GHOST, self.config.N_y + 2 * GHOST)
        )

        name = self.name
        X = self.X[GHOST:-GHOST, GHOST:-GHOST]
        Y = self.Y[GHOST:-GHOST, GHOST:-GHOST]

        density = Q[0, GHOST:-GHOST, GHOST:-GHOST]
        momentum_x = Q[1, GHOST:-GHOST, GHOST:-GHOST]
        momentum_y = Q[2, GHOST:-GHOST, GHOST:-GHOST]
        energy = Q[3, GHOST:-GHOST, GHOST:-GHOST]
        pressure = find_pressure(
            density, momentum_x, momentum_y, energy, gamma=self.config.gamma
        )
        velocity_x = momentum_x / density
        velocity_y = momentum_y / density
        entropy = numpy.log(
            numpy.maximum(1e-16, pressure)
            / (numpy.maximum(1e-16, density) ** self.config.gamma)
        )

        ratio = self.config.plot_ratio
        level = self.config.plot_contour_level
        plot_density(
            name,
            X,
            Y,
            density,
            self.t,
            snapshot,
            ratio=ratio,
            level=level,
            wall=self.plot_wall,
        )
        plot_momentum(
            name,
            X,
            Y,
            momentum_x,
            momentum_y,
            self.t,
            snapshot,
            ratio=ratio,
            level=level,
            wall=self.plot_wall,
        )
        plot_energy(
            name,
            X,
            Y,
            energy,
            self.t,
            snapshot,
            ratio=ratio,
            level=level,
            wall=self.plot_wall,
        )
        plot_pressure(
            name,
            X,
            Y,
            pressure,
            self.t,
            snapshot,
            ratio=ratio,
            level=level,
            wall=self.plot_wall,
        )
        plot_velocity(
            name, X, Y, velocity_x, velocity_y, self.t, snapshot, wall=self.plot_wall
        )
        plot_entropy(
            name,
            X,
            Y,
            entropy,
            self.t,
            snapshot,
            ratio=ratio,
            level=level,
            wall=self.plot_wall,
        )
        plot_vorticity(
            name,
            X,
            Y,
            velocity_x,
            velocity_y,
            self.t,
            snapshot,
            ratio=ratio,
            level=level,
            wall=self.plot_wall,
        )

    def RHS(self, t, y, y_dot, user_data):
        N_x = self.config.N_x
        N_y = self.config.N_y

        Q = N_VGetArrayPointer(y)
        Q = numpy.reshape(Q, (4, N_x + 2 * GHOST, N_y + 2 * GHOST))

        Q_dot = N_VGetArrayPointer(y_dot)
        Q_dot = numpy.reshape(Q_dot, (4, N_x + 2 * GHOST, N_y + 2 * GHOST))

        Q_dot[:, :, :] = numpy.zeros_like(Q_dot)
        Q_dot[:, GHOST:-GHOST, GHOST:-GHOST] = find_RHS(
            Q,
            self.dx,
            self.dy,
            gamma=self.config.gamma,
            boundary_x=self.config.boundary_x,
            boundary_y=self.config.boundary_y,
            update_boundary=self.update_boundary,
            wall=self.wall,
            weno_epsilon=self.config.weno_epsilon,
            use_dissipation=self.config.use_dissipation,
            use_JST_dissipation=self.config.use_JST_dissipation,
            JST_kappa_2=self.config.JST_kappa_2,
            JST_kappa_4=self.config.JST_kappa_4,
        )

        return 0

    def after_step(self, t, y, user_data):
        self.t = t
        self.n_step += 1

        _, dt = ARKodeGetCurrentStep(self.stepper.get())
        self.dt.append(dt)

        print(
            f"{self.n_step:10}: {self.t:.6e}, {(self.t/self.config.t_final*100):.3f}%"
        )

        return 0

    def start(self):
        _, context = SUNContext_Create(SUN_COMM_NULL)
        y = N_VNew_Serial(
            4 * (self.config.N_x + 2 * GHOST) * (self.config.N_y + 2 * GHOST), context
        )

        self.setup_initial_condition(y, self.X, self.Y, self.config)

        if self.config.method == "ssp":
            self.stepper = LSRKStepCreateSSP(
                self.RHS, self.config.t_initial, y, context
            )
            LSRKStepSetSSPMethodByName(self.stepper.get(), "ARKODE_LSRK_SSP_S_3")
        else:
            raise RuntimeError(
                f"Invalid command line option: --method={self.config.method}"
            )
        stepper = self.stepper.get()

        ARKodeSStolerances(stepper, 1e-6, 1e-12)
        ARKodeSetMaxNumSteps(stepper, -1)
        if self.config.dt != 0.0:
            ARKodeSetFixedStep(stepper, self.config.dt)

        ARKodeSetPostprocessStepFn(stepper, self.after_step)

        self.plot(y, 0)
        self.save_snapshot(y, 0)
        for snapshot in range(1, self.config.snapshot + 1):
            t_stop = (
                self.config.t_initial
                + (self.config.t_final - self.config.t_initial)
                / self.config.snapshot
                * snapshot
            )
            ARKodeSetStopTime(stepper, t_stop)
            ARKodeEvolve(stepper, t_stop, y, ARK_NORMAL)
            ARKodeClearStopTime(stepper)

            self.plot(y, snapshot)
            self.save_snapshot(y, snapshot)
