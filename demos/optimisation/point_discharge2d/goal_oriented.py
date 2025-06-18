# TODO: text

from matplotlib import ticker
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from animate import *
from firedrake import *
from goalie import *
from firedrake.pyplot import *

from setup import *

# TODO: text

n = 2
mesh = RectangleMesh(100 * n, 20 * n, 50, 10)

# TODO: text

time_partition = TimeInstant(fields)
mesh_seq = GoalOrientedMeshSeq(
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
fig.savefig(f"goal_oriented_{n}_forward_unoptimised.jpg")

# Plot the adjoint solution, too, which has a different scale. ::

plot_kwargs = {"figsize": (10, 3), "cmap": "coolwarm", "levels": 33}
fig, axes, tcs = plot_snapshots(
    solutions, time_partition, "c", "adjoint", **plot_kwargs
)
fig.colorbar(tcs[0][0], orientation="horizontal", pad=0.2)
axes.set_title("Adjoint solution")
fig.savefig(f"goal_oriented_{n}_adjoint.jpg")

J = mesh_seq.J
print(f"Initial control: {float(mesh_seq.field_functions['yc']):.8f}")
print(f"Initial QoI: {J:.4e}")
print(f"Initial gradient: {float(mesh_seq.gradient['r']):.4e}")

# TODO: text

# Run a Taylor test to check we are happy with the gradients. ::

# FIXME: Sub-quadratic convergence
# mesh_seq.taylor_test("yc")


def adaptor(mesh_seq, solutions, indicators):
    P1_ten = TensorFunctionSpace(mesh_seq[0], "CG", 1)

    # Recover the Hessian of the forward solution
    hessian = RiemannianMetric(P1_ten)
    hessian.compute_hessian(solutions["c"]["forward"][0][0])

    # Ramp the target metric complexity from 400 to 1000 over the first few iterations
    metric = RiemannianMetric(P1_ten)
    base, target, iteration = 400, 1000, mesh_seq.fp_iteration
    mp = {"dm_plex_metric_target_complexity": ramp_complexity(base, target, iteration)}
    metric.set_parameters(mp)

    # Deduce an anisotropic metric from the error indicator field and the Hessian
    metric.compute_anisotropic_dwr_metric(indicators["c"][0][0], hessian)
    complexity = metric.complexity()

    # Adapt the mesh
    mesh_seq[0] = adapt(mesh_seq[0], metric)
    num_dofs = mesh_seq.count_vertices()[0]
    num_elem = mesh_seq.count_elements()[0]
    pyrint(
        f"{iteration + 1:2d}, complexity: {complexity:4.0f}"
        f", dofs: {num_dofs:4d}, elements: {num_elem:4d}"
    )

    # Plot each intermediate adapted mesh
    fig, axes = plt.subplots(figsize=(10, 2))
    mesh_seq.plot(fig=fig, axes=axes, interior_kw={"linewidth": 0.5})
    axes.set_title(f"Mesh at iteration {iteration + 1}")
    fig.savefig(f"aniso_go_mesh{iteration + 1}.jpg")
    plt.close()

    # Plot error indicator on intermediate meshes
    plot_kwargs["norm"] = mcolors.LogNorm()
    plot_kwargs["locator"] = ticker.LogLocator()
    fig, axes, tcs = plot_indicator_snapshots(
        indicators, time_partition, "c", **plot_kwargs
    )
    axes.set_title(f"Indicator at iteration {mesh_seq.fp_iteration + 1}")
    fig.colorbar(tcs[0][0], orientation="horizontal", pad=0.2)
    fig.savefig(f"aniso_go_indicator{mesh_seq.fp_iteration + 1}.jpg")
    plt.close()

    # Check whether the target complexity has been (approximately) reached. If not,
    # return ``True`` to indicate that convergence checks should be skipped until the
    # next fixed point iteration.
    continue_unconditionally = complexity < 0.95 * target
    return [continue_unconditionally]


adapt_parameters = GoalOrientedAdaptParameters(
    {
        "element_rtol": 0.005,
        "qoi_rtol": 0.005,
        "maxiter": 35,
    }
)

# Now run the optimisation routine and plot the results. ::

opt_parameters = OptimisationParameters({"lr": 0.001, "maxiter": 100})
optimiser = QoIOptimiser(
    mesh_seq, "yc", opt_parameters, method="gradient_descent", adaptor=adaptor
)
optimiser.minimise(adaptation_parameters=adapt_parameters)

# TODO: text

fig, axes = plt.subplots()
axes.plot(optimiser.progress["count"], optimiser.progress["control"], "--x")
axes.set_xlabel("Iteration")
axes.set_ylabel("Control")
axes.grid(True)
plt.savefig(f"goal_oriented_{n}_control.jpg", bbox_inches="tight")

fig, axes = plt.subplots()
axes.plot(optimiser.progress["count"], optimiser.progress["qoi"], "--x")
axes.set_xlabel("Iteration")
axes.set_ylabel("QoI")
axes.grid(True)
plt.savefig(f"goal_oriented_{n}_qoi.jpg", bbox_inches="tight")

# .. figure:: goal_oriented_2_control.jpg
#    :figwidth: 80%
#    :align: center
#
# .. figure:: goal_oriented_2_qoi.jpg
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
fig.savefig(f"goal_oriented_{n}_forward_optimised.jpg")

# .. figure:: goal_oriented_2_forward_optimised.jpg
#    :figwidth: 80%
#    :align: center
