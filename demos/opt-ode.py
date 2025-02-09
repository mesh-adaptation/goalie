import matplotlib.pyplot as plt
from firedrake import *

from goalie import *

dt = 0.2
times = np.arange(0, 1.01, dt)

end_time = 1
time_partition = TimeInterval(end_time, dt, "u")


def get_function_spaces(mesh):
    return {"u": FunctionSpace(mesh, "R", 0)}


def get_initial_condition(point_seq):
    fs = point_seq.function_spaces["u"][0]
    return {"u": Function(fs).assign(1.0)}


def get_solver_theta(point_seq):
    def solver(index):
        tp = point_seq.time_partition
        u, u_ = point_seq.fields["u"]
        R = point_seq.function_spaces["u"][index]
        dt = Function(R).assign(tp.timesteps[index])
        v = TestFunction(R)

        # Initialise control to zero
        theta = Function(R).assign(0.0)
        # TODO: Avoid such hackiness
        point_seq._control = theta
        F = (u - u_ - dt * (theta * u + (1 - theta) * u_)) * v * dx

        sp = {"ksp_type": "preonly", "pc_type": "jacobi"}
        dt = tp.timesteps[index]
        t_start, t_end = tp.subintervals[index]
        t = t_start
        while t < t_end - 1.0e-05:
            solve(F == 0, u, solver_parameters=sp)
            yield

            u_.assign(u)
            t += dt

    return solver


point_seq = PointSeq(
    time_partition,
    get_function_spaces=get_function_spaces,
    get_initial_condition=get_initial_condition,
    get_solver=get_solver_theta,
)

solutions = point_seq.solve_forward()["u"]["forward"]

forward_euler_trajectory = [float(get_initial_condition(point_seq)["u"])]
forward_euler_trajectory += [
    float(sol) for subinterval in solutions for sol in subinterval
]

# Plot the trajectory and compare it against the analytical solution. ::

fig, axes = plt.subplots()
axes.plot(times, np.exp(times), "--x", label="Analytical solution")
axes.plot(times, forward_euler_trajectory, "--+", label="Forward Euler")
axes.set_xlabel(r"Time, $t$")
axes.set_ylabel(r"$u$")
axes.grid(True)
axes.legend()
plt.tight_layout()
plt.savefig("opt-ode-forward_euler.jpg")

# TODO: Optimisation of theta parameter
