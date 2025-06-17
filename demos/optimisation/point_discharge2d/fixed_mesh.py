# TODO: text

import matplotlib.pyplot as plt
from firedrake import *
from goalie import *
from firedrake.pyplot import *

from setup import *

# TODO: text

n = 2
mesh = RectangleMesh(100 * n, 20 * n, 50, 10)

# TODO: text

time_partition = TimeInstant(fields)
mesh_seq = AdjointMeshSeq(
    time_partition,
    mesh,
    get_initial_condition=get_initial_condition,
    get_solver=get_solver,
    get_qoi=get_qoi,
    qoi_type="steady",
)

solutions = mesh_seq.solve_adjoint(compute_gradient=True)

# Plot the solution fields and check the initial QoI and gradient. ::

plot_kwargs = {
    "levels": np.linspace(0, 1.65, 33),
    "figsize": (10, 3),
    "cmap": "coolwarm",
}
fig, axes, tcs = plot_snapshots(
    solutions, time_partition, "c", "forward", **plot_kwargs
)
fig.colorbar(tcs[0][0], orientation="horizontal", pad=0.2)
axes.set_title("Unoptimised forward solution")
fig.savefig(f"fixed_mesh_{n}_forward_unoptimised.jpg")

# Plot the adjoint solution, too, which has a different scale. ::

plot_kwargs = {"figsize": (10, 3), "cmap": "coolwarm", "levels": 33}
fig, axes, tcs = plot_snapshots(
    solutions, time_partition, "c", "adjoint", **plot_kwargs
)
fig.colorbar(tcs[0][0], orientation="horizontal", pad=0.2)
axes.set_title("Adjoint solution")
fig.savefig(f"fixed_mesh_{n}_adjoint.jpg")

J = mesh_seq.J
print(f"Initial control: {float(mesh_seq.field_functions['yc']):.8f}")
print(f"Initial QoI: {J:.4e}")
print(f"Initial gradient: {float(mesh_seq.gradient['r']):.4e}")

# TODO: text

# Run a Taylor test to check we are happy with the gradients. ::

# FIXME: Sub-quadratic convergence
# mesh_seq.taylor_test("yc")

# Now run the optimisation routine and plot the results. ::

parameters = OptimisationParameters({"lr": 0.001, "maxiter": 100})
optimiser = QoIOptimiser(mesh_seq, "yc", parameters, method="gradient_descent")
optimiser.minimise()

# TODO: text

fig, axes = plt.subplots()
axes.plot(optimiser.progress["count"], optimiser.progress["control"], "--x")
axes.set_xlabel("Iteration")
axes.set_ylabel("Control")
axes.grid(True)
plt.savefig(f"fixed_mesh_{n}_control.jpg", bbox_inches="tight")

fig, axes = plt.subplots()
axes.plot(optimiser.progress["count"], optimiser.progress["qoi"], "--x")
axes.set_xlabel("Iteration")
axes.set_ylabel("QoI")
axes.grid(True)
plt.savefig(f"fixed_mesh_{n}_qoi.jpg", bbox_inches="tight")

# .. figure:: fixed_mesh_2_control.jpg
#    :figwidth: 80%
#    :align: center
#
# .. figure:: fixed_mesh_2_qoi.jpg
#    :figwidth: 80%
#    :align: center
#
# TODO: text

solutions = mesh_seq.solve_forward()
print(f"Optimised control: {float(mesh_seq.controls['yc'].tape_value()):.8f}")

plot_kwargs = {
    "levels": np.linspace(0, 1.65, 33),
    "figsize": (10, 3),
    "cmap": "coolwarm",
}
fig, axes, tcs = plot_snapshots(
    solutions, time_partition, "c", "forward", **plot_kwargs
)
fig.colorbar(tcs[0][0], orientation="horizontal", pad=0.2)
axes.set_title("Optimised forward solution")
fig.savefig(f"fixed_mesh_{n}_forward_optimised.jpg")

# .. figure:: fixed_mesh_2_forward_optimised.jpg
#    :figwidth: 80%
#    :align: center
