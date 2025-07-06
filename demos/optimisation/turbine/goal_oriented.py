import argparse
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from animate.metric import RiemannianMetric
from animate.adapt import adapt
from firedrake.pyplot import tricontourf
from firedrake.function import Function
from firedrake.functionspace import TensorFunctionSpace
from firedrake.utility_meshes import RectangleMesh

from goalie.go_mesh_seq import GoalOrientedMeshSeq
from goalie.log import pyrint
from goalie.metric import ramp_complexity
from goalie.plot import plot_indicator_snapshots
from goalie.optimisation import QoIOptimiser
from goalie.options import OptimisationParameters
from goalie.time_partition import TimeInstant

from setup import fields, get_initial_condition, get_solver, get_qoi

# Add argparse for command-line arguments
parser = argparse.ArgumentParser(description="Plot progress of controls and QoIs.")
parser.add_argument("--n", type=int, default=4, help="Initial mesh resolution.")
args = parser.parse_args()

# Use parsed arguments
n = args.n

anisotropic = False
aniso_str = "aniso" if anisotropic else "iso"

# Set up the GoalOrientedMeshSeq
mesh_seq = GoalOrientedMeshSeq(
    TimeInstant(fields),
    RectangleMesh(12 * n, 5 * n, 1200, 500),
    get_initial_condition=get_initial_condition,
    get_solver=get_solver,
    get_qoi=get_qoi,
    qoi_type="steady",
)

# # FIXME: Sub-quadratic convergence
# # Run a Taylor test to check the gradient is computed correctly
# mesh_seq.taylor_test("yc")

# Solve the adjoint problem, computing gradients, and plot the x-velocity component of
# both the forward and adjoint solutions
solutions = mesh_seq.solve_adjoint(compute_gradient=True)
u, eta = solutions["solution_2d"]["forward"][0][0].subfunctions
fig, axes = plt.subplots(figsize=(12, 5))
axes.set_title(r"Forward $x$-velocity")
fig.colorbar(tricontourf(u.subfunctions[0], axes=axes, cmap="coolwarm"), ax=axes)
plt.savefig(f"goal_oriented_{n}_forward.jpg", bbox_inches="tight")
fig, axes = plt.subplots(figsize=(12, 5))
axes.set_title(r"Adjoint $x$-velocity")
u_star, eta_star = solutions["solution_2d"]["adjoint"][0][0].subfunctions
fig.colorbar(tricontourf(u_star.subfunctions[0], axes=axes, cmap="coolwarm"), ax=axes)
plt.savefig(f"goal_oriented_{n}_adjoint.jpg", bbox_inches="tight")

J = mesh_seq.J
print(f"J = {J:.4e}")

parameters = OptimisationParameters({"lr": 10.0})
print(parameters)


def adaptor(mesh_seq, solutions, indicators):
    P1_ten = TensorFunctionSpace(mesh_seq[0], "CG", 1)
    metric = RiemannianMetric(P1_ten)

    # Ramp the target metric complexity from 400 to 1000 over the first few iterations
    base, target, iteration, num_iterations = 400, 1000, mesh_seq.fp_iteration, 3
    mp = {
        "dm_plex_metric_target_complexity": ramp_complexity(
            base, target, iteration, num_iterations=num_iterations
        ),
        # "dm_plex_metric_no_insert": True,
    }
    metric.set_parameters(mp)

    if anisotropic:
        # Recover the Hessian of the forward solution
        hessians = {key: RiemannianMetric(P1_ten) for key in ("u", "v", "eta")}
        (u, v), eta = solutions["solution_2d"]["forward"][0][0].subfunctions
        hessians["u"].compute_hessian(u)
        hessians["v"].compute_hessian(v)
        hessians["eta"].compute_hessian(eta)
        hessian = hessians["u"].intersect(hessians["v"], hessians["eta"])

        # Deduce an anisotropic metric from the error indicator field and the Hessian
        metric.compute_anisotropic_dwr_metric(indicators["solution_2d"][0][0], hessian)
    else:
        # Deduce an isotropic metric from the error indicator field
        metric.compute_isotropic_dwr_metric(indicators["solution_2d"][0][0])
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
    fig, axes = plt.subplots(figsize=(12, 5))
    interior_kw = {"edgecolor": "k", "linewidth": 0.5}
    mesh_seq.plot(fig=fig, axes=axes, interior_kw=interior_kw)
    axes.set_title(f"Mesh at iteration {iteration + 1}")
    fig.savefig(f"{aniso_str}_go_mesh{iteration + 1}.jpg")
    plt.close()

    # Plot error indicator on intermediate meshes
    plot_kwargs = {"figsize": (12, 5)}
    plot_kwargs["norm"] = mcolors.LogNorm()
    plot_kwargs["locator"] = ticker.LogLocator()
    fig, axes, tcs = plot_indicator_snapshots(
        indicators, mesh_seq.time_partition, "solution_2d", **plot_kwargs
    )
    axes.set_title(f"Indicator at iteration {mesh_seq.fp_iteration + 1}")
    fig.colorbar(tcs[0][0], orientation="horizontal", pad=0.2)
    fig.savefig(f"{aniso_str}_go_indicator{mesh_seq.fp_iteration + 1}.jpg")
    plt.close()

    # Check whether the target complexity has been (approximately) reached. If not,
    # return ``True`` to indicate that convergence checks should be skipped until the
    # next fixed point iteration.
    continue_unconditionally = complexity < 0.95 * target
    return [continue_unconditionally]


optimiser = QoIOptimiser(
    mesh_seq, "yc", parameters, method="gradient_descent", adaptor=adaptor
)
optimiser.minimise()

np.save(f"goal_oriented_{aniso_str}_{n}_control.npy", optimiser.progress["control"])
np.save(f"goal_oriented_{aniso_str}_{n}_qoi.npy", optimiser.progress["qoi"])
