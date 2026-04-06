from matplotlib import pyplot
import numpy

from weno import GHOST

pyplot.rcParams["text.usetex"] = True


def plot_density(name, x, y, density, t, snapshot, ratio=1, level=20, wall=None):
    colorbar_location = "bottom" if ratio <= 1 else "right"

    figure, ax = pyplot.subplots(
        nrows=1,
        ncols=2,
        figsize=(10 * 2, 12 * ratio),
        dpi=200,
        constrained_layout=True,
    )

    if wall is not None:
        density = numpy.ma.masked_where(~wall[GHOST:-GHOST, GHOST:-GHOST], density)

    # pcolor
    image_pcolor = ax[0].pcolormesh(x, y, density, shading="gouraud", cmap="plasma")
    ax[0].set_aspect("equal")
    ax[0].set_xlabel(r"$x$")
    ax[0].set_ylabel(r"$y$", rotation=0)
    ax[0].set_title(rf"Density ($t$ = {t:.3E})")
    figure.colorbar(image_pcolor, ax=ax[0], location=colorbar_location)

    # contour
    image_contour = ax[1].contour(x, y, density, levels=level, cmap="plasma")
    ax[1].set_aspect("equal")
    ax[1].set_xlabel(r"$x$")
    ax[1].set_ylabel(r"$y$", rotation=0)
    ax[1].set_title(rf"Density contour ($t$ = {t:.3E})")
    figure.colorbar(image_pcolor, ax=ax[1], location=colorbar_location)

    # wall
    for a in ax.flat:
        plot_wall(a, wall, x, y)

    figure.savefig(f"{name}.{snapshot:03}.density.png")
    pyplot.close()


def plot_momentum(
    name, x, y, momentum_x, momentum_y, t, snapshot, ratio=1, level=20, wall=None
):
    colorbar_location = "bottom" if ratio <= 1 else "right"

    figure, ax = pyplot.subplots(
        nrows=3,
        ncols=2,
        figsize=(10 * 2, 12 * 3 * ratio),
        dpi=200,
        constrained_layout=True,
    )

    if wall is not None:
        momentum_x = numpy.ma.masked_where(
            ~wall[GHOST:-GHOST, GHOST:-GHOST], momentum_x
        )
        momentum_y = numpy.ma.masked_where(
            ~wall[GHOST:-GHOST, GHOST:-GHOST], momentum_y
        )
    momentum_total = numpy.sqrt(momentum_x**2 + momentum_y**2)

    # pcolor: x
    image_x_pcolor = ax[0, 0].pcolormesh(
        x, y, momentum_x, shading="gouraud", cmap="plasma"
    )
    ax[0, 0].set_aspect("equal")
    ax[0, 0].set_xlabel(r"$x$")
    ax[0, 0].set_ylabel(r"$y$", rotation=0)
    ax[0, 0].set_title(rf"Momentum-$x$ ($t$ = {t:.3E})")
    figure.colorbar(image_x_pcolor, ax=ax[0, 0], location=colorbar_location)
    # pcolor: y
    image_y_pcolor = ax[1, 0].pcolormesh(
        x, y, momentum_y, shading="gouraud", cmap="plasma"
    )
    ax[1, 0].set_aspect("equal")
    ax[1, 0].set_xlabel(r"$x$")
    ax[1, 0].set_ylabel(r"$y$", rotation=0)
    ax[1, 0].set_title(rf"Momentum-$y$ ($t$ = {t:.3E})")
    figure.colorbar(image_y_pcolor, ax=ax[1, 0], location=colorbar_location)
    # pcolor: total
    image_total_pcolor = ax[2, 0].pcolormesh(
        x, y, momentum_total, shading="gouraud", cmap="plasma"
    )
    ax[2, 0].set_aspect("equal")
    ax[2, 0].set_xlabel(r"$x$")
    ax[2, 0].set_ylabel(r"$y$", rotation=0)
    ax[2, 0].set_title(rf"Total momentum ($t$ = {t:.3E})")
    figure.colorbar(image_total_pcolor, ax=ax[2, 0], location=colorbar_location)

    # contour: x
    image_x_contour = ax[0, 1].contour(x, y, momentum_x, levels=level, cmap="plasma")
    ax[0, 1].set_aspect("equal")
    ax[0, 1].set_xlabel(r"$x$")
    ax[0, 1].set_ylabel(r"$y$", rotation=0)
    ax[0, 1].set_title(rf"Momentum-$x$ contour ($t$ = {t:.3E})")
    figure.colorbar(image_x_pcolor, ax=ax[0, 1], location=colorbar_location)
    # contour: y
    image_y_contour = ax[1, 1].contour(x, y, momentum_y, levels=level, cmap="plasma")
    ax[1, 1].set_aspect("equal")
    ax[1, 1].set_xlabel(r"$x$")
    ax[1, 1].set_ylabel(r"$y$", rotation=0)
    ax[1, 1].set_title(rf"Momentum-$y$ contour ($t$ = {t:.3E})")
    figure.colorbar(image_y_pcolor, ax=ax[1, 1], location=colorbar_location)
    # contour: total
    image_total_contour = ax[2, 1].contour(
        x, y, momentum_total, levels=level, cmap="plasma"
    )
    ax[2, 1].set_aspect("equal")
    ax[2, 1].set_xlabel(r"$x$")
    ax[2, 1].set_ylabel(r"$y$", rotation=0)
    ax[2, 1].set_title(rf"Total momentum contour ($t$ = {t:.3E})")
    figure.colorbar(image_total_pcolor, ax=ax[2, 1], location=colorbar_location)

    # wall
    for a in ax.flat:
        plot_wall(a, wall, x, y)

    figure.savefig(f"{name}.{snapshot:03}.momentum.png")
    pyplot.close()


def plot_energy(name, x, y, energy, t, snapshot, ratio=1, level=20, wall=None):
    colorbar_location = "bottom" if ratio <= 1 else "right"

    figure, ax = pyplot.subplots(
        nrows=1,
        ncols=2,
        figsize=(10 * 2, 12 * ratio),
        dpi=200,
        constrained_layout=True,
    )

    if wall is not None:
        energy = numpy.ma.masked_where(~wall[GHOST:-GHOST, GHOST:-GHOST], energy)

    # pcolor
    image_pcolor = ax[0].pcolormesh(x, y, energy, shading="gouraud", cmap="plasma")
    ax[0].set_aspect("equal")
    ax[0].set_xlabel(r"$x$")
    ax[0].set_ylabel(r"$y$", rotation=0)
    ax[0].set_title(rf"Energy ($t$ = {t:.3E})")
    figure.colorbar(image_pcolor, ax=ax[0], location=colorbar_location)

    # contour
    image_contour = ax[1].contour(x, y, energy, levels=level, cmap="plasma")
    ax[1].set_aspect("equal")
    ax[1].set_xlabel(r"$x$")
    ax[1].set_ylabel(r"$y$", rotation=0)
    ax[1].set_title(rf"Energy contour ($t$ = {t:.3E})")
    figure.colorbar(image_pcolor, ax=ax[1], location=colorbar_location)

    # wall
    for a in ax.flat:
        plot_wall(a, wall, x, y)

    figure.savefig(f"{name}.{snapshot:03}.energy.png")
    pyplot.close()


def plot_pressure(name, x, y, pressure, t, snapshot, ratio=1, level=20, wall=None):
    colorbar_location = "bottom" if ratio <= 1 else "right"

    figure, ax = pyplot.subplots(
        nrows=1,
        ncols=2,
        figsize=(10 * 2, 12 * ratio),
        dpi=200,
        constrained_layout=True,
    )

    if wall is not None:
        pressure = numpy.ma.masked_where(~wall[GHOST:-GHOST, GHOST:-GHOST], pressure)

    # pcolor
    image_pcolor = ax[0].pcolormesh(x, y, pressure, shading="gouraud", cmap="plasma")
    ax[0].set_aspect("equal")
    ax[0].set_xlabel(r"$x$")
    ax[0].set_ylabel(r"$y$", rotation=0)
    ax[0].set_title(rf"Pressure ($t$ = {t:.3E})")
    figure.colorbar(image_pcolor, ax=ax[0], location=colorbar_location)

    # contour
    image_contour = ax[1].contour(x, y, pressure, levels=level, cmap="plasma")
    ax[1].set_aspect("equal")
    ax[1].set_xlabel(r"$x$")
    ax[1].set_ylabel(r"$y$", rotation=0)
    ax[1].set_title(rf"Pressure contour ($t$ = {t:.3E})")
    figure.colorbar(image_pcolor, ax=ax[1], location=colorbar_location)

    # wall
    for a in ax.flat:
        plot_wall(a, wall, x, y)

    figure.savefig(f"{name}.{snapshot:03}.pressure.png")
    pyplot.close()


def plot_velocity(
    name,
    x,
    y,
    velocity_x,
    velocity_y,
    t,
    snapshot,
    ratio=1,
    wall=None,
):
    colorbar_location = "bottom" if ratio <= 1 else "right"

    figure, ax = pyplot.subplots(
        nrows=1,
        ncols=1,
        figsize=(10, 10 * ratio),
        dpi=200,
        constrained_layout=True,
    )

    if wall is not None:
        velocity_x = numpy.ma.masked_where(
            ~wall[GHOST:-GHOST, GHOST:-GHOST], velocity_x
        )
        velocity_y = numpy.ma.masked_where(
            ~wall[GHOST:-GHOST, GHOST:-GHOST], velocity_y
        )

    stream_x, stream_y = numpy.meshgrid(x[:, 0], y[0, :])
    image_stream = ax.streamplot(
        stream_x,
        stream_y,
        numpy.transpose(velocity_x),
        numpy.transpose(velocity_y),
        color="k",
    )
    ax.set_title(rf"Velocity ($t$ = {t:.3E})")

    # wall
    plot_wall(ax, wall, x, y)

    figure.savefig(f"{name}.{snapshot:03}.velocity.png")
    pyplot.close()


def plot_entropy(name, x, y, entropy, t, snapshot, ratio=1, level=20, wall=None):
    colorbar_location = "bottom" if ratio <= 1 else "right"

    figure, ax = pyplot.subplots(
        nrows=1,
        ncols=2,
        figsize=(10 * 2, 12 * ratio),
        dpi=200,
        constrained_layout=True,
    )

    if wall is not None:
        entropy = numpy.ma.masked_where(~wall[GHOST:-GHOST, GHOST:-GHOST], entropy)

    # pcolor
    image_pcolor = ax[0].pcolormesh(x, y, entropy, shading="gouraud", cmap="plasma")
    ax[0].set_aspect("equal")
    ax[0].set_xlabel(r"$x$")
    ax[0].set_ylabel(r"$y$", rotation=0)
    ax[0].set_title(rf"Entropy ($t$ = {t:.3E})")
    figure.colorbar(image_pcolor, ax=ax[0], location=colorbar_location)

    # contour
    image_contour = ax[1].contour(x, y, entropy, levels=level, cmap="plasma")
    ax[1].set_aspect("equal")
    ax[1].set_xlabel(r"$x$")
    ax[1].set_ylabel(r"$y$", rotation=0)
    ax[1].set_title(rf"Entropy contour ($t$ = {t:.3E})")
    figure.colorbar(image_pcolor, ax=ax[1], location=colorbar_location)

    # wall
    for a in ax.flat:
        plot_wall(a, wall, x, y)

    figure.savefig(f"{name}.{snapshot:03}.entropy.png")
    pyplot.close()


def plot_vorticity(
    name, x, y, velocity_x, velocity_y, t, snapshot, ratio=1, level=20, wall=None
):
    colorbar_location = "bottom" if ratio <= 1 else "right"

    figure, ax = pyplot.subplots(
        nrows=1,
        ncols=2,
        figsize=(10 * 2, 12 * ratio),
        dpi=200,
        constrained_layout=True,
    )

    if wall is not None:
        velocity_x = numpy.ma.masked_where(
            ~wall[GHOST:-GHOST, GHOST:-GHOST], velocity_x
        )
        velocity_y = numpy.ma.masked_where(
            ~wall[GHOST:-GHOST, GHOST:-GHOST], velocity_y
        )

    vorticity = (velocity_y[2:, 1:-1] - velocity_y[:-2, 1:-1]) / (
        x[2:, 1:-1] - x[:-2, 1:-1]
    ) - (velocity_x[1:-1, 2:] - velocity_x[1:-1, :-2]) / (y[1:-1, 2:] - y[1:-1, :-2])

    x = x[1:-1, 1:-1]
    y = y[1:-1, 1:-1]

    # pcolor
    image_pcolor = ax[0].pcolormesh(x, y, vorticity, shading="gouraud", cmap="plasma")
    ax[0].set_aspect("equal")
    ax[0].set_xlabel(r"$x$")
    ax[0].set_ylabel(r"$y$", rotation=0)
    figure.colorbar(image_pcolor, ax=ax[0], location=colorbar_location)
    ax[0].set_title(rf"Vorticity ($t$ = {t:.3E})")

    # contour
    image_contour = ax[1].contour(x, y, vorticity, levels=level, cmap="plasma")
    ax[1].set_aspect("equal")
    ax[1].set_xlabel(r"$x$")
    ax[1].set_ylabel(r"$y$", rotation=0)
    ax[1].set_title(rf"Vorticity contour ($t$ = {t:.3E})")
    figure.colorbar(image_pcolor, ax=ax[1], location=colorbar_location)

    # wall
    if wall is not None:
        for a in ax.flat:
            plot_wall(a, wall[1:-1, 1:-1], x, y)

    figure.savefig(f"{name}.{snapshot:03}.vorticity.png")
    pyplot.close()


def plot_temperature(
    name, x, y, density, pressure, t, snapshot, ratio=1, level=20, wall=None
):
    colorbar_location = "bottom" if ratio <= 1 else "right"

    figure, ax = pyplot.subplots(
        nrows=1,
        ncols=2,
        figsize=(10 * 2, 12 * ratio),
        dpi=200,
        constrained_layout=True,
    )

    if wall is not None:
        density = numpy.ma.masked_where(~wall[GHOST:-GHOST, GHOST:-GHOST], density)
        pressure = numpy.ma.masked_where(~wall[GHOST:-GHOST, GHOST:-GHOST], pressure)
    temperature = pressure / density

    # pcolor
    image_pcolor = ax[0].pcolormesh(x, y, temperature, shading="gouraud", cmap="plasma")
    ax[0].set_aspect("equal")
    ax[0].set_xlabel(r"$x$")
    ax[0].set_ylabel(r"$y$", rotation=0)
    ax[0].set_title(rf"Temperature ($t$ = {t:.3E})")
    figure.colorbar(image_pcolor, ax=ax[0], location=colorbar_location)

    # contour
    image_contour = ax[1].contour(x, y, temperature, levels=level, cmap="plasma")
    ax[1].set_aspect("equal")
    ax[1].set_xlabel(r"$x$")
    ax[1].set_ylabel(r"$y$", rotation=0)
    ax[1].set_title(rf"Temperature contour ($t$ = {t:.3E})")
    figure.colorbar(image_pcolor, ax=ax[1], location=colorbar_location)

    # wall
    for a in ax.flat:
        plot_wall(a, wall, x, y)

    figure.savefig(f"{name}.{snapshot:03}.temperature.png")
    pyplot.close()


def plot_wall(ax, wall, x, y):
    if wall is None:
        return

    wall = wall[GHOST:-GHOST, GHOST:-GHOST]
    ax.contourf(x, y, ~wall, levels=[0.5, 1.5], colors=["#cccccc"])
    ax.contour(
        x,
        y,
        wall.astype(float),
        levels=[0.5],
        colors="black",
        linewidths=3,
    )
