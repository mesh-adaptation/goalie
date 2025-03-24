import matplotlib.pyplot as plt
from firedrake import *

from goalie_adjoint import *

dt = 0.2
times = np.arange(0, 1.01, dt)

end_time = 1
fields = [Field("u"), Field("theta", unsteady=False, solved_for=False)]
time_partition = TimeInterval(end_time, dt, fields)


def get_initial_condition(point_seq):
    return {
        "u": Function(point_seq.function_spaces["u"][0]).assign(1.0),
        "theta": Function(point_seq.function_spaces["theta"][0]).assign(0.0),
    }


def get_solver_theta(point_seq):
    def solver(index):
        u, u_ = point_seq.field_data["u"]
        theta = point_seq.field_data["theta"]
        R = point_seq.function_spaces["u"][index]
        tp = point_seq.time_partition
        dt = Function(R).assign(tp.timesteps[index])
        v = TestFunction(R)

        F = (u - u_ - dt * (theta * u + (1 - theta) * u_)) * v * dx

        sp = {"ksp_type": "preonly", "pc_type": "jacobi"}
        t_start, t_end = tp.subintervals[index]
        dt = tp.timesteps[index]
        t = t_start
        while t < t_end - 1.0e-05:
            solve(F == 0, u, solver_parameters=sp, ad_block_tag="u")
            yield

            u_.assign(u)
            t += dt

    return solver


def get_qoi(point_seq, index):
    def end_time_qoi():
        sol = exp(1.0)
        u = point_seq.field_data["u"][0]
        return inner(u - sol, u - sol) * dx

    return end_time_qoi


point_seq = AdjointMeshSeq(
    time_partition,
    VertexOnlyMesh(UnitIntervalMesh(1), [[0.5]]),
    get_initial_condition=get_initial_condition,
    get_solver=get_solver_theta,
    get_qoi=get_qoi,
    qoi_type="end_time",
)

# Print initial field values
ics = get_initial_condition(point_seq)
for fieldname, field in ics.items():
    if isinstance(field, tuple):
        print(f"{fieldname}_0 = {float(field[0]):.4e}")
    else:
        print(f"{fieldname}_0 = {float(field):.4e}")

solutions = point_seq.solve_adjoint(compute_gradient=True)

# Print QoI value
J = point_seq.J
print(f"J = {J:.4e}")

forward_euler_trajectory = [float(get_initial_condition(point_seq)["u"])]
forward_euler_trajectory += [
    float(sol) for subinterval in solutions["u"]["forward"] for sol in subinterval
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
plt.savefig("opt-ode_forward_euler.jpg", bbox_inches="tight")

# Print gradient values
for fieldname, gradient in point_seq.gradient.items():
    print(f"dJ/d{fieldname} = {float(gradient):.4e}")

parameters = OptimisationParameters({"lr": 0.5, "maxiter": 100})
print(parameters)

optimiser = QoIOptimiser(point_seq, "theta", parameters, method="gradient_descent")
optimiser.minimise()

fig, axes = plt.subplots()
axes.plot(optimiser.progress["control"], "--x")
axes.set_xlabel("Iteration")
axes.set_ylabel("Control")
axes.grid(True)
plt.savefig("opt-ode_control.jpg", bbox_inches="tight")

fig, axes = plt.subplots()
axes.plot(optimiser.progress["qoi"], "--x")
axes.set_xlabel("Iteration")
axes.set_ylabel("QoI")
axes.grid(True)
plt.savefig("opt-ode_qoi.jpg", bbox_inches="tight")

# Plot optimised trajectory
solutions = point_seq.solve_adjoint()
J = point_seq.J
print(f"Optimised QoI: {J:.4e}")
print(f"Optimised control: {float(point_seq._control):.4f}")
opt_trajectory = [float(get_initial_condition(point_seq)["u"])]
opt_trajectory += [
    float(sol) for subinterval in solutions["u"]["forward"] for sol in subinterval
]
fig, axes = plt.subplots()
axes.plot(times, np.exp(times), "--x", label="Analytical solution")
axes.plot(times, forward_euler_trajectory, "--+", label="Forward Euler")
axes.plot(times, opt_trajectory, "--+", label="Optimised")
axes.set_xlabel(r"Time, $t$")
axes.set_ylabel(r"$u$")
axes.grid(True)
axes.legend()
plt.tight_layout()
plt.savefig("opt-ode_optimised.jpg", bbox_inches="tight")
