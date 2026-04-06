import h5py


def save_snapshot(name, arg, problem, density, momentum, energy, velocity, pressure):
    with h5py.File(f"{name}.h5", "w") as file:
        group = file.create_group("data")

        group_input = group.create_group("input")
        group_input.create_dataset("gamma", data=problem.gamma)
        group_input.create_dataset("method", data=arg.method)
        group_input.create_dataset("nx", data=arg.nx)

        group_output = group.create_group("output")
        group_output.create_dataset("t_stop", data=problem.t_stop)

        group_mesh = group.create_group("mesh")
        group_mesh.create_dataset("x", data=problem.x)

        group_profile = group.create_group("profile")
        group_profile.create_dataset("density", data=density)
        group_profile.create_dataset("momentum", data=momentum)
        group_profile.create_dataset("energy", data=energy)
        group_profile.create_dataset("velocity", data=velocity)
        group_profile.create_dataset("pressure", data=pressure)
